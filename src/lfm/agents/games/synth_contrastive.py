"""Contrastive expression game over a frozen ``SynthLM`` voice box.

Mirrors the ``ContrastiveGame`` pattern (anchor + distractors → InfoNCE
contrastive loss + diversity regularisers + n-gram blocking) but with the
SynthLM substrate:

  * Generator    = ``PrefixProjector``   (instead of ``DiffusionZGenerator``)
  * Voice box    = frozen ``SynthLM``    (instead of frozen ``DepTreeVAE``)
  * Re-encoder   = the same Qwen body, mean-pooled  (instead of ``MessageEncoder``
                   on top of an ``IPAEncoder``)

The structural advantage of this setup: re-encoding generated alien tokens
goes through the *same* Qwen body that generated them, so the round-trip
``Qwen(alien_gen) ≈ source_embedding`` lives in a single representational
geometry by construction.

Training signal:
  loss = info_nce(alien_emb, source_emb, distractors)
       + λ_topology * cosine(alien_emb, source_emb)
       + λ_bigram_kl * BigramKL(generated || corpus_bigrams)
       + λ_adj_div  * AdjacencyDiversity(generated)
       + λ_diversity * cross-batch token-set diversity (anti-collapse)

The last term is the lesson from DepTreeVAE: a high contrastive accuracy
achieved via degenerate repeating codes is a Pyrrhic victory. We add
explicit pressure for *non-degenerate* generation so the contrastive metric
correlates with linguistic discrimination, not collapse-discrimination.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from lfm.agents.components import embed_tokens_straight_through
from lfm.agents.config import CurriculumConfig
from lfm.agents.games.base import (
    AdjDiversityLoss,
    NgramKLLoss,
    _AdjPartial,
    _NgramBlocker,
    _NgramPartial,
)
from lfm.config.base import LFMBaseConfig
from lfm.synth.config import SynthConfig
from lfm.synth.model import SynthLM


# ─── config ────────────────────────────────────────────────────────────────


class SynthContrastiveGameConfig(LFMBaseConfig):
    """Full game + training config. AgentTrainer reads training fields directly
    from this object (batch_size, num_distractors, curriculum, output_dir, etc.).
    """

    # ── SynthLM source paths ─────────────────────────────────────────────
    synth_config: str = "configs/synth_local_qwen.yaml"  # SynthConfig YAML
    phase1_checkpoint: str = ""                           # frozen Phase 1 checkpoint

    # ── Source / generation ─────────────────────────────────────────────
    source_dim: int = 896                # per-position dim of last_k_concat
    n_source_positions: int = 8          # K of last_k_concat
    n_prefix_tokens: int = 8             # PrefixProjector output positions
    max_gen_len: int = 64                # AR generation cap (in alien tokens)

    # ── Contrastive loss ─────────────────────────────────────────────────
    contrastive_temperature: float = 0.07
    embedding_dim: int = 896             # message space dim (== Qwen hidden)

    # ── Loss weights ─────────────────────────────────────────────────────
    info_nce_weight: float = 1.0
    topology_weight: float = 0.1
    adj_diversity_weight: float = 0.05

    # ── n-gram KL anti-collapse: each entry is (n, weight, npz_path) ─────
    # e.g. [{"n": 2, "weight": 0.05, "path": ".../bigram_topk.npz"},
    #       {"n": 3, "weight": 0.05, "path": ".../trigram_topk.npz"}]
    # Higher orders constrain deeper sequential structure.
    # Backwards-compat: if `bigram_kl_path` is set and `ngram_kl` is empty,
    # a single n=2 entry is auto-constructed at weight `bigram_kl_weight`.
    ngram_kl: list[dict] = []
    bigram_kl_weight: float = 0.05       # back-compat; used iff ngram_kl empty
    bigram_kl_path: str | None = None    # back-compat; used iff ngram_kl empty

    # ── Diversity / anti-degenerate ──────────────────────────────────────
    adj_diversity_target: float = 0.3
    ngram_block: list[int] = []          # e.g. [3, 4] blocks during AR generation

    # ── Generation control ───────────────────────────────────────────────
    generation_temperature: float = 1.0  # >0 enables sampling
    eos_token_id: int = 3
    pad_token_id: int = 0

    # ── Anchor curriculum (for the round-trip's contrastive ramp) ────────
    distractor_curriculum_start: int = 0
    distractor_curriculum_end: int = 5000   # ramp distractors over these steps

    # ── Online hard-negative mining (ANCE-style) ─────────────────────────
    # Once the curriculum-warmup phase ends, periodically re-encode all
    # passages and rebuild a per-passage top-K nearest-neighbor index in
    # the *current* model's alien-emb space. Distractors are then sampled
    # from this self-paced confusion set instead of the static KMeans
    # clusters — the model always trains against its current weakness.
    # Set ``hard_neg_refresh_every: 0`` to disable.
    hard_neg_refresh_every: int = 0
    hard_neg_topk: int = 100
    hard_neg_warmup: int = 2000     # don't start mining until this step

    # ── Trainer fields (read directly by AgentTrainer) ───────────────────
    embedding_store_dir: str = "data/embeddings_qwen"
    batch_size: int = 16
    gradient_accumulation_steps: int = 1
    steps: int = 5000
    num_distractors: int = 7
    max_grad_norm: float = 1.0
    contrastive_scoring: bool = True     # batch-wide InfoNCE for accuracy reporting

    # Per-component LRs (the AgentTrainer reads trainable_param_groups() to
    # build the optimizer, so these are referenced inside the game class)
    projector_lr: float = 1e-4
    head_lr: float = 1e-4
    log_temperature_lr: float = 1e-3

    # Trainer logging / checkpointing
    log_every: int = 25
    checkpoint_every: int = 100
    output_dir: str = "data/synth_contrastive_game"
    device: str = "cuda"
    seed: int = 42

    curriculum: CurriculumConfig = CurriculumConfig()


# ─── output dataclass ──────────────────────────────────────────────────────


@dataclass
class SynthExpressionOutput:
    """What flows from generation to loss computation."""

    alien_emb: Tensor      # (B, P, D)     positionally-binned re-encoded alien (with grad)
    token_ids: Tensor      # (B, S)        generated alien token IDs (no grad)
    valid_mask: Tensor     # (B, S)        positions before/at first EOS
    surface_emb: Tensor    # (B, S, D)     straight-through embeddings of generated tokens
    hidden: Tensor         # (B, S, D)     body hidden states from re-encoding pass
    probs: Tensor          # (B, S, V)     output probabilities (for ngram_kl / adj_div)


# ─── game class ────────────────────────────────────────────────────────────


class SynthContrastiveGame(nn.Module):
    """Contrastive expression game over a frozen SynthLM.

    The PrefixProjector + LengthHead from ``synth_lm`` are the trainable
    components; the body, alien_emb, alien_head, and the auxiliary heads
    (coherence, ending) are frozen.

    AgentTrainer interface:
      * ``forward(anchor, distractors, *, step) -> dict[str, Tensor]``
      * ``trainable_param_groups() -> list[dict]``
      * ``checkpoint_state() -> dict`` / ``load_checkpoint_state(ckpt: dict)``
    """

    def __init__(
        self,
        config: SynthContrastiveGameConfig,
        synth_lm: SynthLM,
        synth_config: SynthConfig,
    ) -> None:
        super().__init__()
        self.config = config
        self.synth_config = synth_config

        # Freeze the entire SynthLM. Only PrefixProjector + the new
        # discriminator head below are trainable.
        self.synth_lm = self._freeze(synth_lm)
        self._d_model = synth_lm.backend.d_model
        self._vocab_size = synth_lm.backend._alien_emb.num_embeddings
        self._eos_id = config.eos_token_id
        self._pad_id = config.pad_token_id

        # Trainable: projector + a small discriminator head that turns the
        # round-trip alien_emb into a contrastive message vector.
        self.message_head = nn.Linear(self._d_model, config.embedding_dim)
        self.target_proj = nn.Linear(config.source_dim, config.embedding_dim)
        self.log_temperature = nn.Parameter(
            torch.tensor(math.log(config.contrastive_temperature))
        )

        # Diversity / anti-collapse
        # Build the list of n-gram KL losses from config. Back-compat: if the
        # old single-bigram_kl_path/weight is set and ngram_kl is empty,
        # construct one n=2 entry from the legacy fields.
        ngram_specs = list(config.ngram_kl)
        if not ngram_specs and config.bigram_kl_path:
            ngram_specs = [{
                "n": 2,
                "weight": config.bigram_kl_weight,
                "path": config.bigram_kl_path,
            }]
        self._ngram_weights: list[float] = [float(s["weight"]) for s in ngram_specs]
        self._ngram_orders: list[int] = [int(s["n"]) for s in ngram_specs]
        self.ngram_kls = nn.ModuleList([
            NgramKLLoss(
                Path(s["path"]) if s.get("path") else None,
                self._vocab_size,
                n=int(s["n"]),
            )
            for s in ngram_specs
        ])
        self.adj_diversity = AdjDiversityLoss(config.adj_diversity_target)
        self._ngram_blocker = _NgramBlocker(config.ngram_block or [])

    # ─── freeze ────────────────────────────────────────────────────────────

    @staticmethod
    def _freeze(synth_lm: SynthLM) -> SynthLM:
        synth_lm.eval()
        for p in synth_lm.parameters():
            p.requires_grad_(False)
        # PrefixProjector is part of synth_lm but we WILL train it. Re-enable.
        for p in synth_lm.projector.parameters():
            p.requires_grad_(True)
        return synth_lm

    # ─── AgentTrainer interface ────────────────────────────────────────────

    def trainable_param_groups(self) -> list[dict]:
        cfg = self.config
        return [
            {"params": list(self.synth_lm.projector.parameters()), "lr": cfg.projector_lr},
            {"params": list(self.message_head.parameters()),       "lr": cfg.head_lr},
            {"params": list(self.target_proj.parameters()),        "lr": cfg.head_lr},
            {"params": [self.log_temperature],                     "lr": cfg.log_temperature_lr},
        ]

    def checkpoint_state(self) -> dict:
        return {
            "projector":     self.synth_lm.projector.state_dict(),
            "message_head":  self.message_head.state_dict(),
            "target_proj":   self.target_proj.state_dict(),
            "log_temp":      self.log_temperature.detach().clone(),
        }

    def load_checkpoint_state(self, ckpt: dict) -> None:
        self.synth_lm.projector.load_state_dict(ckpt["projector"])
        self.message_head.load_state_dict(ckpt["message_head"])
        self.target_proj.load_state_dict(ckpt["target_proj"])
        with torch.no_grad():
            self.log_temperature.copy_(ckpt["log_temp"])

    # ─── main forward ─────────────────────────────────────────────────────

    def forward(
        self,
        anchor: Tensor,                              # (B, [n_source,] source_dim) — flat or multi-position
        distractors: Tensor,                         # (B, K, [n_source,] source_dim)
        *,
        step: int = 0,
        candidate_indices: Tensor | None = None,    # accepted for AgentTrainer compatibility; unused here
    ) -> dict[str, Tensor]:
        del candidate_indices
        ratio = self._distractor_curriculum_ratio(step)
        n_dist = int(round(ratio * distractors.size(1)))
        active_distractors = distractors[:, :n_dist] if n_dist > 0 else None

        expr = self._generate_round_trip(anchor)
        terms, contrastive_logits = self._compute_loss_terms(expr, anchor, active_distractors)
        total = self._aggregate(terms)

        # Accuracy: argmax over (B + n_dist) candidates matches the diagonal
        target_idx = torch.arange(contrastive_logits.size(0), device=contrastive_logits.device)
        accuracy = (contrastive_logits.argmax(dim=1) == target_idx).float().mean()

        # Generation diversity diagnostic — anti-collapse signal
        ttr = self._batch_ttr(expr.token_ids, expr.valid_mask)

        out = {
            "loss":            total,
            "accuracy":        accuracy,
            "ttr":             ttr,
            **terms,
            "n_distractors":   torch.tensor(float(n_dist), device=total.device),
        }
        # Internal (prefixed _) tensors aren't logged by AgentTrainer
        out["_token_ids"] = expr.token_ids
        out["_valid_mask"] = expr.valid_mask
        return out

    @staticmethod
    def _batch_ttr(token_ids: Tensor, valid_mask: Tensor) -> Tensor:
        """Type-token ratio across the batch's valid generated tokens.
        Anti-collapse diagnostic: low TTR means generations are repetitive
        across the batch (the DepTreeVAE failure mode)."""
        flat = token_ids[valid_mask]
        if flat.numel() == 0:
            return torch.zeros((), device=token_ids.device)
        n_unique = torch.unique(flat).numel()
        return torch.tensor(n_unique / flat.numel(), device=token_ids.device)

    # ─── round-trip generation ─────────────────────────────────────────────

    def _generate_round_trip(self, anchor: Tensor) -> SynthExpressionOutput:
        """Project source → AR-generate alien tokens (no_grad, for memory) →
        re-encode [prefix | tokens] through body in a single forward pass with
        gradient flow → positionally bin the alien-token hidden states into
        ``n_source_positions`` buckets matching the source embedding shape.

        Memory rationale: full-grad AR over T steps accumulates O(B·T²·D·L)
        activations. Detaching AR keeps memory at one-forward cost. Gradient
        flows: loss → positional alien_emb → body → prefix → projector.

        Why positional (not mean) pool: mean-pooling collapses sequence
        position into a bag-of-tokens, so the contrastive loss is invariant
        to permutation — the model has zero gradient for grammatical word
        order, long-range dependencies, or sequence-level structure. Binning
        into P=n_source_positions buckets matches the source's last-k_concat
        shape and forces the alien's first 1/P of tokens to encode the
        source's first position, etc. — restoring positional gradient.
        """
        B = anchor.size(0)
        device = anchor.device

        # 1) Project to prefix tokens. (Trainable; gradient flows.)
        prefix = self.synth_lm.projector(anchor).to(self.synth_lm.backend.dtype)

        # 2) Detached autoregressive generation. Only emits token IDs.
        with torch.no_grad():
            prefix_d = prefix.detach()
            context = prefix_d.clone()
            gen_ids: list[Tensor] = []
            done = torch.zeros(B, dtype=torch.bool, device=device)
            temp = max(self.config.generation_temperature, 1e-6)
            for t in range(self.config.max_gen_len):
                hidden = self.synth_lm.backend.forward_hidden(context)
                logits = self.synth_lm.backend.alien_logits(hidden[:, -1:]).squeeze(1)
                logits = self._apply_blockers(logits.unsqueeze(1), gen_ids, t).squeeze(1)
                if temp == 1.0 or self.config.generation_temperature == 0:
                    next_id = logits.argmax(dim=-1) if self.config.generation_temperature == 0 \
                              else torch.multinomial(F.softmax(logits, dim=-1), 1).squeeze(-1)
                else:
                    next_id = torch.multinomial(F.softmax(logits / temp, dim=-1), 1).squeeze(-1)
                gen_ids.append(next_id)
                done = done | (next_id == self._eos_id)
                if done.all():
                    break
                next_emb = self.synth_lm.backend.embed_alien(next_id.unsqueeze(1))
                context = torch.cat([context, next_emb], dim=1)

        token_ids = torch.stack(gen_ids, dim=1)                           # (B, S)
        valid_mask = self._build_valid_mask(token_ids)                    # (B, S) bool

        # 3) Re-encode pass WITH gradient: feed [prefix | embed(tokens)] through
        #    the body once. Tokens are detached IDs; their embeddings come from
        #    the frozen alien_emb table, so gradient flow is:
        #        loss → alien_emb_pool → body → prefix → projector
        #    Gradient-checkpointed so only inputs are saved during forward;
        #    the body forward is recomputed during backward. Bounds peak
        #    activation memory regardless of B*T size.
        token_embs = self.synth_lm.backend.embed_alien(token_ids).to(prefix.dtype)
        prefix_len = prefix.size(1)

        def _re_encode(prefix_, token_embs_):
            full = torch.cat([prefix_, token_embs_], dim=1)
            return self.synth_lm.backend.forward_hidden(full)

        full_hidden = torch.utils.checkpoint.checkpoint(
            _re_encode, prefix, token_embs, use_reentrant=False,
        )
        alien_hidden = full_hidden[:, prefix_len:]                        # (B, S, D)

        # 4) Positional binning: (B, S, D) + (B, S) → (B, P, D) where P =
        #    n_source_positions. Restores positional gradient that mean-pool
        #    destroyed.
        alien_emb = self._positional_pool(
            alien_hidden, valid_mask, n_pos=self.config.n_source_positions,
        )

        # 5) Probs over generated positions — used by bigram_kl and adj_diversity.
        #    We keep these IN the autograd graph (no no_grad wrap) so the diversity
        #    losses provide real gradient signal back through alien_logits → body →
        #    prefix → projector. Without grad flow these are merely diagnostic
        #    measurements, leaving the only real anti-collapse pressure to come
        #    indirectly from info_nce — which was the DepTreeVAE failure pattern.
        probs_seq = F.softmax(
            self.synth_lm.backend.alien_logits(alien_hidden).float(), dim=-1,
        )

        return SynthExpressionOutput(
            alien_emb=alien_emb,
            token_ids=token_ids,
            valid_mask=valid_mask,
            surface_emb=token_embs,
            hidden=alien_hidden,
            probs=probs_seq,
        )

    @torch.no_grad()
    def encode_for_hard_neg_mining(self, anchor: Tensor) -> Tensor:
        """Per-anchor signature for online hard-negative mining.

        Returns a ``(B, source_dim)`` tensor: mean-pooled-over-positions
        of the round-trip alien_emb. This is the projection of each
        passage in the current model's alien space — passages whose
        signatures are nearest are the model's current confusion set.

        Inference-only: uses the KV-cached ``generate`` path for AR
        (O(T) per batch) plus a single re-encode forward (vs the
        ``_generate_round_trip`` AR loop which is O(T²) without cache).
        For 50K passages at batch=128 this is the difference between
        ~10 minutes and ~30 seconds.
        """
        backend = self.synth_lm.backend

        # 1) KV-cached AR generation → token_ids, valid_mask
        token_ids, valid_mask = self.generate(anchor)

        # 2) Single re-encode pass through the body (no grad needed; mining
        #    is purely a similarity-search step).
        prefix = self.synth_lm.projector(anchor).to(backend.dtype)
        token_embs = backend.embed_alien(token_ids).to(prefix.dtype)
        full = torch.cat([prefix, token_embs], dim=1)
        full_hidden = backend.forward_hidden(full)
        alien_hidden = full_hidden[:, prefix.size(1):]

        # 3) Positional pool to (B, P, D), then mean across P for the
        #    kNN signature (mean is a good proxy and 8x faster than flat).
        alien_emb = self._positional_pool(
            alien_hidden, valid_mask, n_pos=self.config.n_source_positions,
        )
        return alien_emb.mean(dim=1).float()

    @torch.no_grad()
    def generate(self, anchor: Tensor) -> tuple[Tensor, Tensor]:
        """Inference-only AR generation with KV caching.

        Same sampling/blocking as ``_generate_round_trip``'s AR loop but uses
        ``past_key_values`` so each step is O(1) in sequence length, and
        skips the grad-flowing re-encode + probs computation entirely.

        Returns ``(token_ids, valid_mask)`` shaped ``(B, S)``.
        """
        B = anchor.size(0)
        device = anchor.device
        backend = self.synth_lm.backend

        prefix = self.synth_lm.projector(anchor).to(backend.dtype)
        hidden, past = backend.forward_hidden(prefix, use_cache=True)

        gen_ids: list[Tensor] = []
        done = torch.zeros(B, dtype=torch.bool, device=device)
        temp = max(self.config.generation_temperature, 1e-6)
        for t in range(self.config.max_gen_len):
            logits = backend.alien_logits(hidden[:, -1:]).squeeze(1)
            logits = self._apply_blockers(logits.unsqueeze(1), gen_ids, t).squeeze(1)
            if self.config.generation_temperature == 0:
                next_id = logits.argmax(dim=-1)
            else:
                next_id = torch.multinomial(F.softmax(logits / temp, dim=-1), 1).squeeze(-1)
            gen_ids.append(next_id)
            done = done | (next_id == self._eos_id)
            if done.all():
                break
            next_emb = backend.embed_alien(next_id.unsqueeze(1))
            hidden, past = backend.forward_hidden(
                next_emb, past_key_values=past, use_cache=True,
            )

        token_ids = torch.stack(gen_ids, dim=1)
        return token_ids, self._build_valid_mask(token_ids)

    def _build_valid_mask(self, token_ids: Tensor) -> Tensor:
        """Mark positions up to and including the first EOS; 0 after."""
        is_eos = token_ids == self._eos_id
        first_eos_or_end = is_eos.cumsum(dim=1) >= 1
        # valid: positions strictly before first EOS, plus the EOS position itself
        before_first_eos = first_eos_or_end.cumsum(dim=1) <= 1
        return before_first_eos

    def _apply_blockers(
        self, logits: Tensor, gen_ids_so_far: list[Tensor], t: int,
    ) -> Tensor:
        """Optional n-gram repeat suppression during AR."""
        if not self.config.ngram_block or t < 2:
            return logits
        history = torch.stack(gen_ids_so_far, dim=1) if gen_ids_so_far else None
        if history is None or history.size(1) < max(self.config.ngram_block):
            return logits
        return self._ngram_blocker(logits.squeeze(1), history, t).unsqueeze(1)

    # ─── positional pooling ───────────────────────────────────────────────

    def _positional_pool(
        self, hidden: Tensor, mask: Tensor, n_pos: int,
    ) -> Tensor:
        """Aggregate (B, S, D) into (B, n_pos, D) by uniformly binning the
        valid-position prefix into ``n_pos`` buckets and mean-pooling within
        each bucket. Empty buckets fall back to the doc's overall mean.

        Concretely: for batch element b with V[b] valid tokens, position s
        goes to bucket floor(s * n_pos / V[b]) when s < V[b], else discarded.
        """
        B, S, D = hidden.shape
        device = hidden.device
        dtype = hidden.dtype

        valid_count = mask.sum(dim=1).clamp(min=1).long()                 # (B,)
        positions = torch.arange(S, device=device).unsqueeze(0).expand(B, -1)  # (B, S)
        # bucket index: floor(s * n_pos / V[b]) clamped to [0, n_pos-1]
        bin_idx = (positions * n_pos // valid_count.unsqueeze(-1)).clamp(max=n_pos - 1)
        # invalid positions go to a "trash" bin n_pos which we discard
        bin_idx = bin_idx.masked_fill(~mask, n_pos)                       # (B, S)

        # scatter-add into (B, n_pos+1, D); last index is the trash bin
        ext_sum = torch.zeros(B, n_pos + 1, D, device=device, dtype=dtype)
        ext_sum.scatter_add_(
            1, bin_idx.unsqueeze(-1).expand(-1, -1, D), hidden,
        )
        ext_count = torch.zeros(B, n_pos + 1, device=device, dtype=dtype)
        ones = torch.ones_like(bin_idx, dtype=dtype)
        ext_count.scatter_add_(1, bin_idx, ones)

        binned = ext_sum[:, :n_pos]                                       # (B, n_pos, D)
        cnt = ext_count[:, :n_pos]                                        # (B, n_pos)
        pooled = binned / cnt.clamp(min=1).unsqueeze(-1)

        # Empty-bin fallback: doc-level mean of valid positions
        m = mask.unsqueeze(-1).to(dtype)
        doc_mean = (hidden * m).sum(dim=1) / m.sum(dim=1).clamp(min=1)    # (B, D)
        empty = (cnt == 0).unsqueeze(-1)                                  # (B, n_pos, 1)
        pooled = torch.where(empty, doc_mean.unsqueeze(1).expand_as(pooled), pooled)
        # Cast to float32 for downstream Linear heads (which are float32).
        return pooled.to(torch.float32)

    # ─── loss terms ───────────────────────────────────────────────────────

    def _compute_loss_terms(
        self,
        expr: SynthExpressionOutput,
        anchor: Tensor,                       # (B, P, source_dim) — never mean-pooled now
        distractors: Tensor | None,           # (B, K, P, source_dim) or None
    ) -> tuple[dict[str, Tensor], Tensor]:
        # Project alien_emb (B, P, D) and source (B, P, source_dim) through
        # trainable heads into message space. Linear broadcasts over leading dims.
        msg_alien = self.message_head(expr.alien_emb)                     # (B, P, E)
        msg_source = self.target_proj(anchor)                             # (B, P, E)

        info_nce, logits = self._info_nce(msg_alien, msg_source, distractors)
        # Topology — cosine averaged across positions and batch
        topology = 1.0 - F.cosine_similarity(
            msg_alien, msg_source, dim=-1,
        ).mean()

        terms: dict[str, Tensor] = {
            "info_nce": info_nce,
            "topology": topology,
        }
        # n-gram KL losses (each registered NgramKLLoss)
        ngram_kl_total = msg_alien.new_zeros(())
        for i, (kl_mod, w, n) in enumerate(zip(
            self.ngram_kls, self._ngram_weights, self._ngram_orders,
        )):
            if not kl_mod.is_loaded or w <= 0:
                continue
            part = kl_mod.partial(expr.probs, expr.valid_mask)
            kl = kl_mod.finalize(part)
            terms[f"ngram_kl_n{n}"] = kl
            ngram_kl_total = ngram_kl_total + w * kl
        terms["ngram_kl"] = ngram_kl_total

        # Adjacency diversity (operates on probs sequence)
        if self.config.adj_diversity_weight > 0:
            ap = self.adj_diversity.partial(expr.probs, expr.valid_mask)
        else:
            ap = self.adj_diversity.zero_partial(msg_alien.device)
        terms["adj_diversity"] = self.adj_diversity.finalize(ap)

        return terms, logits

    def _info_nce(
        self,
        msg_alien: Tensor,                  # (B, P, D)
        msg_source: Tensor,                 # (B, P, D)
        distractors: Tensor | None,         # (B, K, P, source_dim) or None
    ) -> tuple[Tensor, Tensor]:
        """Per-position symmetric InfoNCE, summed over P positions.

        For each position p ∈ [0, P), we run an independent in-batch
        InfoNCE between alien_p and source_p, with optional explicit
        distractor negatives. The per-position losses are averaged.

        Returns (loss, logits) where `logits` is the position-summed
        (B, B+K) score matrix used for accuracy reporting (diagonal =
        correct match across all positions jointly).
        """
        msg_alien = F.normalize(msg_alien, dim=-1)                        # (B, P, D)
        msg_source = F.normalize(msg_source, dim=-1)                      # (B, P, D)
        temp = self.log_temperature.exp()
        B, P, _ = msg_alien.shape
        labels = torch.arange(B, device=msg_alien.device)

        if distractors is not None:
            d_proj = self.target_proj(distractors)                        # (B, K, P, E)
            d_proj = F.normalize(d_proj, dim=-1)
        else:
            d_proj = None

        loss_total = msg_alien.new_zeros(())
        logits_sum: Tensor | None = None
        for p in range(P):
            a = msg_alien[:, p]                                           # (B, D)
            s = msg_source[:, p]                                          # (B, D)
            logits_p = a @ s.t() / temp                                   # (B, B)
            loss_a2s = F.cross_entropy(logits_p, labels)
            loss_s2a = F.cross_entropy(logits_p.t(), labels)
            loss_p = 0.5 * (loss_a2s + loss_s2a)
            full_p = logits_p

            if d_proj is not None:
                d_p = d_proj[:, :, p, :]                                  # (B, K, D)
                d_logits_p = (a.unsqueeze(1) * d_p).sum(-1) / temp        # (B, K)
                full_p = torch.cat([logits_p, d_logits_p], dim=1)         # (B, B+K)
                loss_p = loss_p + F.cross_entropy(full_p, labels)
                loss_p = loss_p / 2.0

            loss_total = loss_total + loss_p
            logits_sum = full_p if logits_sum is None else logits_sum + full_p

        return loss_total / P, logits_sum

    # ─── aggregation ──────────────────────────────────────────────────────

    def _aggregate(self, terms: dict[str, Tensor]) -> Tensor:
        """Sum of differentiable loss terms.

        bigram_kl + adj_diversity carry real gradient now that probs are
        in the autograd graph; bigram_kl penalises divergence from corpus
        natural bigram distribution (catches BOTH cycling collapse AND
        excessive uniform diversity). adj_diversity penalises adjacent-position
        similarity (anti-repetition).

        cross_diversity (TTR-based) is computed from discrete argmax token
        IDs and is non-differentiable — kept only as a diagnostic in the
        output dict (not in the loss). Its `1 - TTR` formulation would also
        push toward TTR=1.0 which is anti-natural (natural English TTR for
        our batch sizes is ~0.4-0.5).
        """
        cfg = self.config
        # ngram_kl is already weighted internally (sum of w_i * KL_i for each
        # registered n-gram order); just add it directly.
        return (
            cfg.info_nce_weight        * terms["info_nce"]
            + cfg.topology_weight      * terms["topology"]
            + terms["ngram_kl"]
            + cfg.adj_diversity_weight * terms["adj_diversity"]
        )

    # ─── curriculum ───────────────────────────────────────────────────────

    def _distractor_curriculum_ratio(self, step: int) -> float:
        cfg = self.config
        if step <= cfg.distractor_curriculum_start:
            return 0.0
        if step >= cfg.distractor_curriculum_end:
            return 1.0
        progress = (step - cfg.distractor_curriculum_start) / (
            cfg.distractor_curriculum_end - cfg.distractor_curriculum_start
        )
        return float(progress)

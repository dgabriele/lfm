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

import logging
import math
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import Tensor, nn

logger = logging.getLogger(__name__)

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
    info_nce_weight: float = 1.0           # per-sentence InfoNCE (stability)
    info_nce_aggregate_weight: float = 0.0  # aggregate-of-K InfoNCE (composition).
    residual_weight: float = 0.0           # boosting-style monotonic-improvement loss.
                                           # For each sub-aggregate of the first k
                                           # sentences (k=1..K), compute InfoNCE.
                                           # Penalise non-improvement: each new
                                           # sentence's contribution must reduce
                                           # aggregate InfoNCE by at least
                                           # ``residual_margin``.
    residual_margin: float = 0.05
                                           # Set > 0 with n_paragraphs > 1 to enable
                                           # compositional training: each sentence is
                                           # conditioned on a learned encoding of
                                           # previous sentences via ctx_proj, and the
                                           # K sentences' aggregate alien_emb must
                                           # match the source — rewards K sentences
                                           # for encoding *complementary* aspects.
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

    # ── Surgery C: multi-paragraph generation per anchor + coherence ─────
    # Each anchor yields ``n_paragraphs`` independent alien expressions
    # via per-paragraph learnable prefix offsets. Coherence loss penalises
    # low pairwise cosine similarity between paragraphs of the same anchor.
    # Targets the discourse-level structure that contrastive-only
    # objectives can't induce (low burstiness, flat class grammar).
    n_paragraphs: int = 1                # 1 = legacy single-paragraph game
    coherence_weight: float = 0.0
    coherence_target_cos: float = 0.5

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

        # Per-paragraph prefix offsets (Surgery C). Small random noise so
        # they diverge during training; with d_model ~896, randn vectors
        # are nearly orthogonal so this gives K usefully-distinct biases
        # to start from.
        if config.n_paragraphs > 1:
            self.paragraph_offsets = nn.Parameter(
                0.02 * torch.randn(
                    config.n_paragraphs, config.n_prefix_tokens, self._d_model,
                ),
            )
            # Compositional training context projector (Surgery D / v5).
            # Maps the pooled alien-hidden mean of *previously-generated*
            # sentences into a prefix-shaped additive term for the next
            # sentence's prefix. Initialised to zero so a fresh-init game
            # starts identical to v4 single-sentence behaviour, and a
            # resumed v3/v4 checkpoint loads cleanly with this layer fresh
            # at zero (no behaviour change until training pulls it off zero).
            self.ctx_proj = nn.Linear(self._d_model, self._d_model)
            nn.init.zeros_(self.ctx_proj.weight)
            nn.init.zeros_(self.ctx_proj.bias)
        else:
            self.register_parameter("paragraph_offsets", None)
            self.ctx_proj = None

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
        groups = [
            {"params": list(self.synth_lm.projector.parameters()), "lr": cfg.projector_lr},
            {"params": list(self.message_head.parameters()),       "lr": cfg.head_lr},
            {"params": list(self.target_proj.parameters()),        "lr": cfg.head_lr},
            {"params": [self.log_temperature],                     "lr": cfg.log_temperature_lr},
        ]
        if self.paragraph_offsets is not None:
            groups.append(
                {"params": [self.paragraph_offsets], "lr": cfg.head_lr},
            )
        if self.ctx_proj is not None:
            groups.append(
                {"params": list(self.ctx_proj.parameters()), "lr": cfg.head_lr},
            )
        return groups

    def checkpoint_state(self) -> dict:
        state = {
            "projector":     self.synth_lm.projector.state_dict(),
            "message_head":  self.message_head.state_dict(),
            "target_proj":   self.target_proj.state_dict(),
            "log_temp":      self.log_temperature.detach().clone(),
        }
        if self.paragraph_offsets is not None:
            state["paragraph_offsets"] = self.paragraph_offsets.detach().clone()
        if self.ctx_proj is not None:
            state["ctx_proj"] = self.ctx_proj.state_dict()
        return state

    def load_checkpoint_state(self, ckpt: dict) -> None:
        self.synth_lm.projector.load_state_dict(ckpt["projector"])
        self.message_head.load_state_dict(ckpt["message_head"])
        self.target_proj.load_state_dict(ckpt["target_proj"])
        with torch.no_grad():
            self.log_temperature.copy_(ckpt["log_temp"])
            # Surgery C: paragraph offsets are new in v4 — silently skip
            # if missing from older checkpoints (they'll keep their fresh
            # random init). Also handle K-bump (resume from a smaller-K
            # ckpt into a larger-K config): copy what we have into the
            # first n_load positions, leave the rest at fresh init.
            if self.paragraph_offsets is not None and "paragraph_offsets" in ckpt:
                saved = ckpt["paragraph_offsets"]
                if saved.shape == self.paragraph_offsets.shape:
                    self.paragraph_offsets.copy_(saved)
                elif saved.dim() == 3 and saved.shape[1:] == self.paragraph_offsets.shape[1:] \
                        and saved.shape[0] <= self.paragraph_offsets.shape[0]:
                    n_load = saved.shape[0]
                    self.paragraph_offsets[:n_load].copy_(saved)
                    logger.info(
                        "paragraph_offsets: loaded %d/%d from ckpt; rest at fresh init",
                        n_load, self.paragraph_offsets.shape[0],
                    )
                else:
                    logger.warning(
                        "paragraph_offsets: ckpt shape %s incompatible with model shape %s — leaving at fresh init",
                        tuple(saved.shape), tuple(self.paragraph_offsets.shape),
                    )
        # Surgery D: ctx_proj is new in v5 — silently skip from older
        # checkpoints (it stays at its zero init, which makes v5 with
        # untouched ctx_proj behaviourally identical to v4).
        if self.ctx_proj is not None and "ctx_proj" in ckpt:
            self.ctx_proj.load_state_dict(ckpt["ctx_proj"])

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
        cfg = self.config
        ratio = self._distractor_curriculum_ratio(step)
        n_dist = int(round(ratio * distractors.size(1)))
        active_distractors = distractors[:, :n_dist] if n_dist > 0 else None
        K = max(1, cfg.n_paragraphs)

        # Compositional chain (Surgery D): sentence k is conditioned on a
        # learned encoding of sentences <k via ctx_proj, summed into the
        # prefix. K=1 collapses to legacy single-sentence behavior.
        # K>1 with ctx_proj at zero (uninitialised / fresh resume) is
        # equivalent to v4 independent multi-paragraph.
        #
        # Context pool is the *running mean* of previous sentences' hidden
        # means (not running sum). The earlier sum-based version had
        # magnitude growing linearly with k, which drove later sentences
        # into a stereotyped narrow attractor (only 8% unique pos-3
        # opening tokens at K=4 in the diagnostic). Mean keeps the input
        # to ctx_proj at a stable scale regardless of K.
        exprs: list[SynthExpressionOutput] = []
        context_pool: Tensor | None = None  # (B, D) running mean of sent means, with grad
        sent_mean_sum: Tensor | None = None
        for k in range(K):
            expr_k = self._generate_round_trip(
                anchor, paragraph_idx=k, context_pool=context_pool,
            )
            exprs.append(expr_k)
            m = expr_k.valid_mask.unsqueeze(-1).to(expr_k.hidden.dtype)
            sent_mean = (expr_k.hidden * m).sum(dim=1) / m.sum(dim=1).clamp(min=1)
            sent_mean = sent_mean.float()
            sent_mean_sum = sent_mean if sent_mean_sum is None else sent_mean_sum + sent_mean
            # context for the *next* sentence is mean over all sentences seen so far
            context_pool = sent_mean_sum / float(k + 1)

        # Per-sentence loss terms: each sentence still must individually
        # encode the anchor (stability term). Averaged across K so weights
        # stay invariant to K.
        per_sent_terms: dict[str, Tensor] = {}
        per_sent_logits: list[Tensor] = []
        for expr in exprs:
            terms_k, logits_k = self._compute_loss_terms(expr, anchor, active_distractors)
            per_sent_logits.append(logits_k)
            for name, val in terms_k.items():
                per_sent_terms[name] = (
                    per_sent_terms[name] + val if name in per_sent_terms else val.clone()
                )
        for name in list(per_sent_terms):
            per_sent_terms[name] = per_sent_terms[name] / K

        # Aggregate-level InfoNCE (Surgery D): mean of K sentence alien_embs
        # must match the source. Rewards the K sentences for encoding
        # *complementary* aspects of the anchor — composition, not
        # redundancy. K identical sentences satisfy per-sentence InfoNCE
        # equally well; only complementary-content sentences also win at
        # aggregate.
        #
        # Surgery E (residual): if residual_weight > 0, also compute the
        # InfoNCE of every sub-aggregate (mean of first k sentences for
        # k=1..K) and penalise non-monotonic-improvement: each new
        # sentence must reduce sub-aggregate InfoNCE by at least
        # residual_margin. This is the "boosting" view of composition —
        # every member of the ensemble must reduce ensemble error.
        compute_aggregate = K > 1 and (
            cfg.info_nce_aggregate_weight > 0 or cfg.residual_weight > 0
        )
        if compute_aggregate:
            msg_source = self.target_proj(anchor)
            stacked_aliens = torch.stack(
                [e.alien_emb for e in exprs], dim=0,
            )                                                       # (K, B, P, D)
            sub_nces: list[Tensor] = []
            agg_logits_list: list[Tensor] = []
            for k in range(1, K + 1):
                # mean of first k sentences' alien_embs
                sub_agg = stacked_aliens[:k].mean(dim=0)            # (B, P, D)
                msg_alien_sub = self.message_head(sub_agg)
                sub_nce_k, sub_logits_k = self._info_nce(
                    msg_alien_sub, msg_source, active_distractors,
                )
                sub_nces.append(sub_nce_k)
                agg_logits_list.append(sub_logits_k)
            # Final-aggregate metrics (k=K)
            agg_alien = stacked_aliens.mean(dim=0)                  # (B, P, D)
            msg_alien_agg = self.message_head(agg_alien)
            agg_topology = 1.0 - F.cosine_similarity(
                msg_alien_agg, msg_source, dim=-1,
            ).mean()
            per_sent_terms["info_nce_agg"] = sub_nces[-1]
            per_sent_terms["topology_agg"] = agg_topology
            agg_logits = agg_logits_list[-1]

            # Residual / monotonic-improvement penalty.
            if cfg.residual_weight > 0:
                margin = float(cfg.residual_margin)
                residual_terms = []
                for k in range(1, K):
                    # sub_nces[k-1] = nce(agg of first k)
                    # sub_nces[k]   = nce(agg of first k+1)
                    # Want sub_nces[k] <= sub_nces[k-1] - margin.
                    # Penalise positive part of (sub_nces[k] - sub_nces[k-1] + margin).
                    delta = sub_nces[k] - sub_nces[k - 1] + margin
                    residual_terms.append(F.relu(delta))
                per_sent_terms["residual"] = torch.stack(residual_terms).mean()
            else:
                per_sent_terms["residual"] = exprs[0].alien_emb.new_zeros(())
        else:
            agg_logits = per_sent_logits[0]
            per_sent_terms["info_nce_agg"] = exprs[0].alien_emb.new_zeros(())
            per_sent_terms["topology_agg"] = exprs[0].alien_emb.new_zeros(())
            per_sent_terms["residual"] = exprs[0].alien_emb.new_zeros(())

        # Cross-sentence diversity (Surgery C): penalise paragraphs whose
        # signatures are too similar — keeps composition meaningful.
        if K > 1 and cfg.coherence_weight > 0:
            per_sent_terms["coherence"] = self._coherence_loss(exprs)
        else:
            per_sent_terms["coherence"] = exprs[0].alien_emb.new_zeros(())

        total = self._aggregate(per_sent_terms)

        # Accuracy: prefer the aggregate logits when active (the real signal
        # of compositional encoding); else use per-sentence-0's logits.
        target_idx = torch.arange(agg_logits.size(0), device=agg_logits.device)
        accuracy = (agg_logits.argmax(dim=1) == target_idx).float().mean()

        # Lexical diagnostics (per-batch, no_grad) — track the surface-token
        # signals that the offline diagnostic flagged: within-doc lexical
        # coherence and positional collapse.
        lex_coh, pos_div_last = self._lexical_diagnostics(exprs)

        # Generation diversity diagnostic — across all paragraphs' tokens.
        # Paragraphs may end at different AR lengths (each stops at its own
        # done.all() boundary); right-pad to common max length before concat.
        max_S = max(e.token_ids.size(1) for e in exprs)
        pad_id = int(self.config.pad_token_id)
        toks_padded = []
        masks_padded = []
        for e in exprs:
            S = e.token_ids.size(1)
            if S < max_S:
                pad_len = max_S - S
                toks_padded.append(F.pad(e.token_ids, (0, pad_len), value=pad_id))
                masks_padded.append(F.pad(e.valid_mask, (0, pad_len), value=False))
            else:
                toks_padded.append(e.token_ids)
                masks_padded.append(e.valid_mask)
        all_tok = torch.cat(toks_padded, dim=0)
        all_mask = torch.cat(masks_padded, dim=0)
        ttr = self._batch_ttr(all_tok, all_mask)

        out = {
            "loss":            total,
            "accuracy":        accuracy,
            "ttr":             ttr,
            "lex_coh":         lex_coh,
            "pos_div_last":    pos_div_last,
            **per_sent_terms,
            "n_distractors":   torch.tensor(float(n_dist), device=total.device),
            "n_paragraphs":    torch.tensor(float(K), device=total.device),
        }
        out["_token_ids"] = all_tok
        out["_valid_mask"] = all_mask
        return out

    def _coherence_loss(
        self, exprs: list[SynthExpressionOutput],
    ) -> Tensor:
        """Cross-paragraph DIVERSITY loss: hinge-penalty on pairwise
        cosine similarity between paragraph signatures being too HIGH.

        Per-paragraph InfoNCE already pulls every paragraph toward its
        anchor's source; without an opposing force, paragraphs collapse
        to near-identical text and Surgery C delivers nothing. This loss
        pushes paragraph signatures *apart* so the K paragraphs encode
        the same anchor via *different* surface forms — yielding the
        within-document topical recurrence + lexical variety needed for
        natural-corpus burstiness.

        Penalty fires when cos > ``coherence_target_cos`` (the maximum
        allowed similarity).
        """
        target = float(self.config.coherence_target_cos)
        sigs = [F.normalize(e.alien_emb.mean(dim=1), dim=-1) for e in exprs]
        K = len(sigs)
        loss = sigs[0].new_zeros(())
        n_pairs = 0
        for i in range(K):
            for j in range(i + 1, K):
                cos = (sigs[i] * sigs[j]).sum(dim=-1)              # (B,)
                loss = loss + F.relu(cos - target).mean()
                n_pairs += 1
        return loss / max(n_pairs, 1)

    @torch.no_grad()
    def _lexical_diagnostics(
        self, exprs: list[SynthExpressionOutput],
    ) -> tuple[Tensor, Tensor]:
        """Per-batch surface-token diagnostics (no_grad, no gradient flow).

        Returns ``(lex_coh, pos_div_last)``:
          * ``lex_coh``  = mean within-doc Jaccard / mean across-doc Jaccard
            over the batch. >1 means same-doc sentences share more tokens
            than random pairs (real topical coherence). =1 means K
            sentences are functionally lexically independent.
          * ``pos_div_last`` = (unique first-tokens at last sentence
            position) / B. <1 indicates positional collapse — the model
            has stereotyped how to start its last sentence.

        Cost: O(B*K^2) set ops + O(K_cross) cross-pair samples; negligible
        per step.
        """
        K = len(exprs)
        device = exprs[0].token_ids.device
        if K < 2:
            zero = torch.zeros((), device=device)
            return zero, zero
        B = exprs[0].token_ids.size(0)

        # Build per-(b, k) token sets from valid positions.
        token_sets: list[list[set[int]]] = []
        for b in range(B):
            sets_for_b: list[set[int]] = []
            for k in range(K):
                ids = exprs[k].token_ids[b].tolist()
                valid = exprs[k].valid_mask[b].tolist()
                sets_for_b.append({int(t) for t, v in zip(ids, valid) if v})
            token_sets.append(sets_for_b)

        def _jacc(a: set[int], b: set[int]) -> float:
            if not a and not b:
                return 1.0
            return len(a & b) / max(len(a | b), 1)

        within = []
        for b in range(B):
            for i in range(K):
                for j in range(i + 1, K):
                    within.append(_jacc(token_sets[b][i], token_sets[b][j]))
        within_mean = sum(within) / max(len(within), 1)

        # Across-doc: random sentence pairs from different docs in the batch.
        across = []
        n_cross = min(B * K, 64)
        for _ in range(n_cross):
            a_idx = int(torch.randint(0, B, (1,)).item())
            b_idx = int(torch.randint(0, B, (1,)).item())
            if B > 1:
                while b_idx == a_idx:
                    b_idx = int(torch.randint(0, B, (1,)).item())
            ka = int(torch.randint(0, K, (1,)).item())
            kb = int(torch.randint(0, K, (1,)).item())
            across.append(_jacc(token_sets[a_idx][ka], token_sets[b_idx][kb]))
        across_mean = sum(across) / max(len(across), 1)

        lex_coh = within_mean / max(across_mean, 1e-12)

        # Positional diversity at the last sentence position.
        last_firsts: list[int] = []
        last = exprs[-1]
        for b in range(B):
            ids = last.token_ids[b].tolist()
            valid = last.valid_mask[b].tolist()
            for t, v in zip(ids, valid):
                if v:
                    last_firsts.append(int(t))
                    break
        pos_div = len(set(last_firsts)) / max(len(last_firsts), 1) if last_firsts else 0.0

        return (
            torch.tensor(lex_coh, device=device, dtype=torch.float32),
            torch.tensor(pos_div, device=device, dtype=torch.float32),
        )

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

    def _build_prefix(
        self, anchor: Tensor, paragraph_idx: int = 0,
        context_pool: Tensor | None = None,
    ) -> Tensor:
        """Construct the conditioning prefix for a single AR pass.

        Composition:
            prefix = projector(anchor)
                   + paragraph_offsets[paragraph_idx]   (Surgery C)
                   + ctx_proj(context_pool).unsqueeze(1)  (Surgery D)

        ``context_pool`` is the trained additive context term — typically
        the running sum of mean-pooled alien hidden states from previously
        generated sentences. ``ctx_proj`` is initialised to zero so the
        first sentence (no prior context) and any sentence with
        ``context_pool=None`` behaves exactly as v3/v4.
        """
        prefix = self.synth_lm.projector(anchor).to(self.synth_lm.backend.dtype)
        if self.paragraph_offsets is not None:
            prefix = prefix + self.paragraph_offsets[paragraph_idx].to(prefix.dtype)
        if context_pool is not None and self.ctx_proj is not None:
            ctx_term = self.ctx_proj(context_pool).to(prefix.dtype)
            prefix = prefix + ctx_term.unsqueeze(1)
        return prefix

    def _round_trip_from_prefix(self, prefix: Tensor) -> SynthExpressionOutput:
        """Detached AR + grad-flowing re-encode + positional pool from a
        pre-built prefix. Factored out so multi-sentence compositional
        training (Surgery D) can build context-conditional prefixes per
        sentence and reuse this function for the round-trip work.

        Memory rationale: full-grad AR over T steps accumulates O(B·T²·D·L)
        activations. Detaching AR keeps memory at one-forward cost. Gradient
        flows: loss → positional alien_emb → body → prefix → (projector,
        paragraph_offsets, ctx_proj).
        """
        backend = self.synth_lm.backend
        B = prefix.size(0)
        device = prefix.device

        # 1) Detached autoregressive generation. Only emits token IDs.
        with torch.no_grad():
            context = prefix.detach().clone()
            gen_ids: list[Tensor] = []
            done = torch.zeros(B, dtype=torch.bool, device=device)
            temp = max(self.config.generation_temperature, 1e-6)
            for t in range(self.config.max_gen_len):
                hidden = backend.forward_hidden(context)
                logits = backend.alien_logits(hidden[:, -1:]).squeeze(1)
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
                next_emb = backend.embed_alien(next_id.unsqueeze(1))
                context = torch.cat([context, next_emb], dim=1)

        token_ids = torch.stack(gen_ids, dim=1)
        valid_mask = self._build_valid_mask(token_ids)

        # 2) Re-encode pass WITH gradient — gradient-checkpointed.
        token_embs = backend.embed_alien(token_ids).to(prefix.dtype)
        prefix_len = prefix.size(1)

        def _re_encode(prefix_, token_embs_):
            full = torch.cat([prefix_, token_embs_], dim=1)
            return backend.forward_hidden(full)

        full_hidden = torch.utils.checkpoint.checkpoint(
            _re_encode, prefix, token_embs, use_reentrant=False,
        )
        alien_hidden = full_hidden[:, prefix_len:]

        # 3) Positional binning: (B, S, D) + (B, S) → (B, P, D).
        alien_emb = self._positional_pool(
            alien_hidden, valid_mask, n_pos=self.config.n_source_positions,
        )

        # 4) Probs in the autograd graph for ngram_kl + adj_diversity.
        probs_seq = F.softmax(
            backend.alien_logits(alien_hidden).float(), dim=-1,
        )

        return SynthExpressionOutput(
            alien_emb=alien_emb,
            token_ids=token_ids,
            valid_mask=valid_mask,
            surface_emb=token_embs,
            hidden=alien_hidden,
            probs=probs_seq,
        )

    def _generate_round_trip(
        self, anchor: Tensor, paragraph_idx: int = 0,
        context_pool: Tensor | None = None,
    ) -> SynthExpressionOutput:
        """Convenience: build prefix and run the round-trip. K=1 callers and
        the K>1 chain in ``forward`` both go through here.
        """
        prefix = self._build_prefix(anchor, paragraph_idx, context_pool)
        return self._round_trip_from_prefix(prefix)

    @torch.no_grad()
    def encode_for_hard_neg_mining(self, anchor: Tensor) -> Tensor:
        """Per-anchor signature for online hard-negative mining.

        Returns a ``(B, n_source_positions * embedding_dim)`` tensor: the
        per-position InfoNCE projection (``message_head`` output) of the
        round-trip alien_emb, L2-normalised per position and then flattened.

        Why this geometry: the InfoNCE loss scores each (alien_i, source_j)
        pair as the sum across positions of per-position cosine similarities
        between ``message_head(alien_emb_i)`` and ``target_proj(source_j)``.
        Two anchors that the contrastive task confuses must therefore have
        similar *per-position* InfoNCE projections — and after per-position
        L2 norm + flatten, their dot product equals the sum-of-per-position
        cosines, which is the exact InfoNCE confusion ranking.

        Earlier mean-pool signature (B, source_dim) collapsed positional
        structure that the per-position contrastive loss specifically uses,
        so its kNN found "near in mean-pool space" pairs that the actual
        contrastive task discriminated trivially. This signature lives in
        the same geometry as the loss and produces *genuinely* hard
        negatives.

        Inference-only: KV-cached generate + single re-encode (vs the
        slower no-cache AR loop in _generate_round_trip).
        """
        backend = self.synth_lm.backend

        # 1) KV-cached AR generation → token_ids, valid_mask
        token_ids, valid_mask = self.generate(anchor)

        # 2) Single re-encode pass through the body
        prefix = self.synth_lm.projector(anchor).to(backend.dtype)
        token_embs = backend.embed_alien(token_ids).to(prefix.dtype)
        full = torch.cat([prefix, token_embs], dim=1)
        full_hidden = backend.forward_hidden(full)
        alien_hidden = full_hidden[:, prefix.size(1):]

        # 3) Positional pool → (B, P, D)
        alien_emb = self._positional_pool(
            alien_hidden, valid_mask, n_pos=self.config.n_source_positions,
        )

        # 4) Project to InfoNCE space → (B, P, embedding_dim)
        msg_alien = self.message_head(alien_emb)

        # 5) Per-position L2 norm + flatten → (B, P * embedding_dim).
        msg_alien = F.normalize(msg_alien, dim=-1)
        return msg_alien.flatten(start_dim=1).float()

    @torch.no_grad()
    def generate(self, anchor: Tensor, paragraph_idx: int = 0) -> tuple[Tensor, Tensor]:
        """Inference-only AR generation with KV caching.

        Same sampling/blocking as ``_generate_round_trip``'s AR loop but uses
        ``past_key_values`` so each step is O(1) in sequence length, and
        skips the grad-flowing re-encode + probs computation entirely.

        For multi-paragraph models (Surgery C) ``paragraph_idx`` selects which
        per-paragraph offset is added to the prefix; default 0 = first
        paragraph. K=1 models ignore the index.

        Returns ``(token_ids, valid_mask)`` shaped ``(B, S)``.
        """
        B = anchor.size(0)
        device = anchor.device
        backend = self.synth_lm.backend

        prefix = self.synth_lm.projector(anchor).to(backend.dtype)
        if self.paragraph_offsets is not None:
            prefix = prefix + self.paragraph_offsets[paragraph_idx].to(prefix.dtype)
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

    @torch.no_grad()
    def chain_generate(
        self, anchor: Tensor, n_sentences: int,
    ) -> list[tuple[Tensor, Tensor]]:
        """Inference-only chain generation (Alternative 1 to Surgery C).

        Generates ``n_sentences`` alien sentences anchored to the same
        source. Each subsequent sentence's prefix is augmented by a
        pooled encoding of the previously-generated sentences — so
        sentence k sees what sentences 0..k-1 produced and can condition
        on them.

        The pooled history is sentence_0_mean + sentence_1_mean + ...
        (sum, not average — gives growing context weight as more is said).
        Added uniformly to all ``n_prefix`` prefix positions before AR.

        For ``k`` beyond the number of trained paragraph_offsets, cycle
        back: ``offsets[k % K_trained]``. The chain conditioning carries
        the actual discourse signal regardless.

        Returns a list of ``n_sentences`` ``(token_ids, valid_mask)``
        pairs.
        """
        backend = self.synth_lm.backend
        device = anchor.device
        B = anchor.size(0)

        base_prefix = self.synth_lm.projector(anchor).to(backend.dtype)  # (B, n_prefix, D)
        n_prefix = base_prefix.size(1)

        history_pool: Tensor | None = None    # (B, D), float32

        sentences: list[tuple[Tensor, Tensor]] = []
        K_trained = self.paragraph_offsets.size(0) if self.paragraph_offsets is not None else 0
        for k in range(n_sentences):
            prefix = base_prefix.clone()
            if K_trained > 0:
                offset_k = self.paragraph_offsets[k % K_trained].to(prefix.dtype)
                prefix = prefix + offset_k
            if history_pool is not None:
                prefix = prefix + history_pool.unsqueeze(1).to(prefix.dtype)

            # AR generate (KV-cached). Accumulate alien-token hidden states
            # into a running sum during the loop — no full hidden tensor is
            # ever materialised, so memory stays O(B*D) regardless of S.
            hidden, past = backend.forward_hidden(prefix, use_cache=True)
            D = hidden.size(-1)
            running_sum = torch.zeros(B, D, device=device, dtype=torch.float32)
            running_count = torch.zeros(B, device=device, dtype=torch.float32)
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
                # Mask BEFORE updating done: include hidden iff element hadn't
                # EOS'd yet, i.e., position t is at-or-before first EOS.
                weight = (~done).float()                       # (B,)
                done = done | (next_id == self._eos_id)
                if done.all():
                    break
                next_emb = backend.embed_alien(next_id.unsqueeze(1))
                hidden, past = backend.forward_hidden(
                    next_emb, past_key_values=past, use_cache=True,
                )
                # hidden is (B, 1, D); accumulate weighted into running pool.
                running_sum = running_sum + hidden.squeeze(1).float() * weight.unsqueeze(-1)
                running_count = running_count + weight
            token_ids = torch.stack(gen_ids, dim=1)
            valid_mask = self._build_valid_mask(token_ids)
            sentences.append((token_ids, valid_mask))

            # Per-element mean over valid hidden states.
            sent_mean = running_sum / running_count.clamp(min=1).unsqueeze(-1)
            history_pool = sent_mean if history_pool is None else history_pool + sent_mean

        return sentences

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
        total = (
            cfg.info_nce_weight        * terms["info_nce"]
            + cfg.topology_weight      * terms["topology"]
            + terms["ngram_kl"]
            + cfg.adj_diversity_weight * terms["adj_diversity"]
        )
        # Surgery D — compositional aggregate terms
        if "info_nce_agg" in terms and cfg.info_nce_aggregate_weight > 0:
            total = total + cfg.info_nce_aggregate_weight * terms["info_nce_agg"]
            if "topology_agg" in terms:
                total = total + cfg.topology_weight * terms["topology_agg"]
        # Surgery E — residual / monotonic-improvement
        if "residual" in terms and cfg.residual_weight > 0:
            total = total + cfg.residual_weight * terms["residual"]
        # Surgery C — cross-sentence diversity
        if "coherence" in terms and cfg.coherence_weight > 0:
            total = total + cfg.coherence_weight * terms["coherence"]
        return total

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

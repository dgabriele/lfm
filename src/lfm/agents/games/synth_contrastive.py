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
from lfm.agents.games.base import (
    AdjDiversityLoss,
    BigramKLLoss,
    _AdjPartial,
    _BigramPartial,
    _NgramBlocker,
)
from lfm.synth.config import SynthConfig
from lfm.synth.model import SynthLM


# ─── config ────────────────────────────────────────────────────────────────


@dataclass
class SynthContrastiveGameConfig:
    """Game-specific knobs. SynthConfig drives the underlying voice box."""

    # Source / generation
    source_dim: int = 896                # per-position dim of last_k_concat
    n_source_positions: int = 8          # K of last_k_concat
    n_prefix_tokens: int = 8             # PrefixProjector output positions
    max_gen_len: int = 64                # AR generation cap (in alien tokens)

    # Contrastive loss
    contrastive_temperature: float = 0.07
    embedding_dim: int = 896             # message space dim (== Qwen hidden)

    # Loss weights
    info_nce_weight: float = 1.0
    topology_weight: float = 0.1
    bigram_kl_weight: float = 0.05
    adj_diversity_weight: float = 0.05
    cross_batch_diversity_weight: float = 0.10  # anti-collapse — see DepTreeVAE lesson

    # Diversity / anti-degenerate
    bigram_kl_path: str | None = None    # corpus reference; None disables
    adj_diversity_target: float = 0.3
    ngram_block: list[int] | None = None  # e.g. [3, 4] blocks 3- and 4-gram repeats during AR

    # Generation control
    generation_temperature: float = 1.0  # 0 = argmax; >0 enables sampling
    eos_token_id: int = 3
    pad_token_id: int = 0

    # Curriculum
    distractor_curriculum_start: int = 0
    distractor_curriculum_end: int = 5000   # ramp to full distractor count over these steps


# ─── output dataclass ──────────────────────────────────────────────────────


@dataclass
class SynthExpressionOutput:
    """What flows from generation to loss computation."""

    alien_emb: Tensor      # (B, D)        re-encoded alien text in Qwen space (with grad)
    token_ids: Tensor      # (B, S)        generated alien token IDs (no grad)
    valid_mask: Tensor     # (B, S)        positions before/at first EOS
    surface_emb: Tensor    # (B, S, D)     straight-through embeddings of generated tokens
    hidden: Tensor         # (B, S, D)     body hidden states from re-encoding pass
    probs: Tensor          # (B, S, V)     output probabilities (for bigram_kl / adj_div)


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
        self.bigram_kl = BigramKLLoss(
            Path(config.bigram_kl_path) if config.bigram_kl_path else None,
            self._vocab_size,
        )
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
        return [
            {"params": list(self.synth_lm.projector.parameters()), "lr": 1e-4},
            {"params": list(self.message_head.parameters()),       "lr": 1e-4},
            {"params": list(self.target_proj.parameters()),        "lr": 1e-4},
            {"params": [self.log_temperature],                     "lr": 1e-3},
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
        """Project source → AR-generate alien tokens with straight-through grads
        → re-encode via the same frozen body → mean-pool the alien-token portion.

        The single body forward at re-encoding closes the round-trip in one pass
        rather than re-running through a separate encoder. Straight-through
        embeddings during AR keep gradient flow from the round-trip cosine all
        the way back to the projector.
        """
        B = anchor.size(0)
        device = anchor.device

        # 1) Project to prefix tokens. PrefixProjector accepts (B, n_source, source_dim)
        #    and returns (B, n_prefix, d_model).
        prefix = self.synth_lm.projector(anchor).to(self.synth_lm.backend.dtype)

        # 2) Autoregressive generation with straight-through embedding.
        context = prefix.clone()
        gen_embs: list[Tensor] = []
        gen_ids: list[Tensor] = []
        gen_probs: list[Tensor] = []
        done = torch.zeros(B, dtype=torch.bool, device=device)

        for t in range(self.config.max_gen_len):
            hidden = self.synth_lm.backend.forward_hidden(context)
            last_hidden = hidden[:, -1:]                                  # (B, 1, D)
            logits = self.synth_lm.backend.alien_logits(last_hidden)      # (B, 1, V)
            logits = self._apply_blockers(logits, gen_ids, t)
            probs = F.softmax(logits.float() / max(self.config.generation_temperature, 1e-6), dim=-1)
            # Straight-through embed
            hard_id = probs.argmax(dim=-1)                                # (B, 1)
            hard_onehot = F.one_hot(hard_id, self._vocab_size).float()
            st = (hard_onehot - probs).detach() + probs                   # (B, 1, V)
            next_emb = (st.to(self.synth_lm.backend._alien_emb.weight.dtype)
                        @ self.synth_lm.backend._alien_emb.weight).to(context.dtype)
            gen_embs.append(next_emb.squeeze(1))
            gen_ids.append(hard_id.squeeze(1))
            gen_probs.append(probs.squeeze(1))

            done = done | (hard_id.squeeze(1) == self._eos_id)
            if done.all():
                break
            context = torch.cat([context, next_emb], dim=1)

        token_ids = torch.stack(gen_ids, dim=1)                           # (B, S)
        surface_emb = torch.stack(gen_embs, dim=1)                        # (B, S, D)
        probs_seq = torch.stack(gen_probs, dim=1)                         # (B, S, V)
        valid_mask = self._build_valid_mask(token_ids)                    # (B, S) bool

        # 3) Re-encode: feed [prefix | surface_emb] through the body once more
        #    so the alien-token positions get hidden states that reflect their
        #    final context (not their generation-time context).
        full = torch.cat([prefix, surface_emb], dim=1)
        full_hidden = self.synth_lm.backend.forward_hidden(full)
        alien_hidden = full_hidden[:, prefix.size(1):]                    # (B, S, D)

        # 4) Mean-pool the alien portion under the valid mask → alien_emb
        m = valid_mask.float().unsqueeze(-1)
        alien_emb = (alien_hidden * m).sum(dim=1) / m.sum(dim=1).clamp(min=1.0)

        return SynthExpressionOutput(
            alien_emb=alien_emb,
            token_ids=token_ids,
            valid_mask=valid_mask,
            surface_emb=surface_emb,
            hidden=alien_hidden,
            probs=probs_seq,
        )

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

    # ─── loss terms ───────────────────────────────────────────────────────

    def _compute_loss_terms(
        self,
        expr: SynthExpressionOutput,
        anchor: Tensor,
        distractors: Tensor | None,
    ) -> tuple[dict[str, Tensor], Tensor]:
        # Project alien_emb and source through trainable heads into message space
        msg_alien = self.message_head(expr.alien_emb)
        msg_source = self.target_proj(self._pool_source(anchor))

        info_nce, logits = self._info_nce(msg_alien, msg_source, distractors)
        topology = 1.0 - F.cosine_similarity(msg_alien, msg_source, dim=-1).mean()

        # Bigram KL & adjacency diversity computed on probs over generated positions
        bp = self.bigram_kl.zero_partial(msg_alien.device)
        ap = self.adj_diversity.zero_partial(msg_alien.device)
        if self.bigram_kl.is_loaded:
            bp = self.bigram_kl.partial(expr.probs, expr.valid_mask)
        if self.config.adj_diversity_weight > 0:
            ap = self.adj_diversity.partial(expr.probs, expr.valid_mask)
        bigram_kl = self.bigram_kl.finalize(bp)
        adj_diversity = self.adj_diversity.finalize(ap)

        # Cross-batch token diversity — anti-collapse safeguard inspired by
        # DepTreeVAE's degenerate-repeating-code failure mode. Penalises low
        # type-token ratio across the batch's generated tokens.
        cross_diversity = self._cross_batch_diversity_loss(expr)

        return {
            "info_nce":        info_nce,
            "topology":        topology,
            "bigram_kl":       bigram_kl,
            "adj_diversity":   adj_diversity,
            "cross_diversity": cross_diversity,
        }, logits

    def _pool_source(self, anchor: Tensor) -> Tensor:
        """Reduce (B, n_source, source_dim) → (B, source_dim) via mean over positions."""
        if anchor.dim() == 2:
            return anchor
        return anchor.mean(dim=1)

    def _info_nce(
        self,
        msg_alien: Tensor,                  # (B, D)
        msg_source: Tensor,                 # (B, D)
        distractors: Tensor | None,         # (B, K, n_source, source_dim) or None
    ) -> tuple[Tensor, Tensor]:
        """Symmetric in-batch InfoNCE plus optional explicit distractors.
        Returns (loss, logits) where logits is the (B, B+K) score matrix
        used for accuracy reporting (diagonal = correct match)."""
        msg_alien = F.normalize(msg_alien, dim=-1)
        msg_source = F.normalize(msg_source, dim=-1)
        temp = self.log_temperature.exp()

        # In-batch contrastive: B alien embeddings vs B source embeddings
        logits = msg_alien @ msg_source.t() / temp                         # (B, B)
        labels = torch.arange(msg_alien.size(0), device=msg_alien.device)
        loss_a2s = F.cross_entropy(logits, labels)
        loss_s2a = F.cross_entropy(logits.t(), labels)
        loss = 0.5 * (loss_a2s + loss_s2a)
        full_logits = logits

        if distractors is not None:
            # Add explicit distractors: each anchor i scores against its B-1
            # in-batch negatives PLUS K curriculum distractors
            d_proj = self.target_proj(self._pool_source(distractors))      # (B, K, D)
            d_proj = F.normalize(d_proj, dim=-1)
            d_logits = (msg_alien.unsqueeze(1) * d_proj).sum(-1) / temp    # (B, K)
            full_logits = torch.cat([logits, d_logits], dim=1)             # (B, B+K)
            loss = loss + F.cross_entropy(full_logits, labels)
            loss = loss / 2.0

        return loss, full_logits

    def _cross_batch_diversity_loss(self, expr: SynthExpressionOutput) -> Tensor:
        """Penalise low type-token ratio across batch's valid generated tokens.

        Direct counter to the DepTreeVAE failure: if all batch members produce
        the same tokens, this loss is high. If each produces distinct tokens,
        it's near zero.
        """
        ids = expr.token_ids
        mask = expr.valid_mask
        flat = ids[mask]
        if flat.numel() == 0:
            return torch.zeros((), device=ids.device)
        n_unique = torch.unique(flat).numel()
        n_total = flat.numel()
        ttr = n_unique / max(n_total, 1)
        # Penalise low TTR; loss = (1 - ttr)
        return torch.tensor(1.0 - ttr, device=ids.device)

    # ─── aggregation ──────────────────────────────────────────────────────

    def _aggregate(self, terms: dict[str, Tensor]) -> Tensor:
        cfg = self.config
        return (
            cfg.info_nce_weight              * terms["info_nce"]
            + cfg.topology_weight            * terms["topology"]
            + cfg.bigram_kl_weight           * terms["bigram_kl"]
            + cfg.adj_diversity_weight       * terms["adj_diversity"]
            + cfg.cross_batch_diversity_weight * terms["cross_diversity"]
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

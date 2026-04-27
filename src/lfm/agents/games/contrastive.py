"""Contrastive expression game over a frozen ``DepTreeVAE`` decoder.

Inherits all VAE-loading, generation, and decode infrastructure from
:class:`~lfm.agents.games.base.BaseExpressionGame`.  This subclass adds
the InfoNCE discrimination heads (hidden-state + surface) and four
information-theoretically-motivated regularisers (bigram-KL,
adj-diversity, decoder-fluency, LLM-pressure).

Loss
----

  total = α·hidden_NCE + β·surface_NCE + γ·topology
        + δ·z_diversity + ε·bigram_KL + ζ·adj_diversity + η·z_prior
        + θ·decoder_fluency + ι·LLM_pressure
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass  # still used for LogitAggregations
from pathlib import Path
from typing import Optional

import torch
import torch.nn.functional as F  # noqa: N812
from torch import Tensor, nn

from lfm.agents.components import MessageEncoder
from lfm.agents.config import MessageEncoderConfig
from lfm.agents.games.base import (
    AdjDiversityLoss,
    BaseExpressionGame,
    BaseExpressionGameConfig,
    BigramKLLoss,
    ExpressionOutput,
    _AdjPartial,
    _BigramPartial,
)
from lfm.agents.llm_pressure import LLMPressureScorer
from lfm.generator.dep_tree_vae.model import DepTreeVAE

logger = logging.getLogger(__name__)


# ===========================================================================
# Config
# ===========================================================================


class ContrastiveGameConfig(BaseExpressionGameConfig):
    """Configuration for the dep_tree_vae contrastive game."""

    # Surface InfoNCE encoder.
    encoder: MessageEncoderConfig = MessageEncoderConfig()

    # Loss weights specific to the contrastive game.
    hidden_infonce_weight: float = 1.0
    surface_infonce_weight: float = 1.0
    bigram_kl_weight: float = 0.05
    adj_diversity_weight: float = 0.05
    adj_diversity_target: float = 0.30
    llm_pressure_weight: float = 0.0

    # Shared InfoNCE temperature.
    contrastive_temperature: float = 0.07

    # Top-K corpus bigram .npz produced by compute_corpus_bigram.py.
    bigram_kl_path: Optional[str] = None

    # Frozen LLM scorer for llm_pressure (loaded only if weight > 0).
    llm_model_name: str = "Qwen/Qwen2.5-0.5B"
    llm_gumbel_tau: float = 1.0

    # Trainer reads this to set the B-way chance baseline.
    contrastive_scoring: bool = True

    # Override defaults from base — contrastive game uses distractors.
    num_distractors: int = 15
    output_dir: str = "data/contrastive_dep_tree_v1"


# ===========================================================================
# Output dataclasses
# ===========================================================================


@dataclass
class LogitAggregations:
    """What the chunked output_head pass produces.

    The (B, S, V) tensors logits/probs are NEVER returned: they live
    only inside the chunk loop and are released between chunks.
    """

    surface_emb: Tensor      # (B, S, H) straight-through token embeddings
    bigram_kl: Tensor        # scalar
    adj_diversity: Tensor    # scalar
    decoder_fluency: Tensor  # scalar — mean CE of greedy tokens under decoder


# ===========================================================================
# Game
# ===========================================================================


class ContrastiveGame(BaseExpressionGame):
    """Contrastive expression game over a frozen ``DepTreeVAE``.

    Public interface used by ``AgentTrainer``:
      * ``forward(anchor, distractors, *, step, candidate_indices)``
      * ``render_surface(token_ids, mask, eos_id, output_mode)``
      * ``trainable_param_groups()``
      * ``checkpoint_state()`` / ``load_checkpoint_state(ckpt)``
    """

    def __init__(self, config: ContrastiveGameConfig, vae: DepTreeVAE) -> None:
        super().__init__(config, vae)

        self.hidden_proj = nn.Linear(self._hidden_dim, config.embedding_dim)
        self.surface_encoder = MessageEncoder(
            self._hidden_dim, config.embedding_dim,
            num_heads=config.encoder.num_heads,
            num_layers=config.encoder.num_layers,
        )
        self.log_temperature = nn.Parameter(
            torch.tensor(math.log(config.contrastive_temperature)),
        )

        self.bigram_kl = BigramKLLoss(
            Path(config.bigram_kl_path) if config.bigram_kl_path else None,
            self._vocab_size,
        )
        self.adj_diversity = AdjDiversityLoss(config.adj_diversity_target)
        self.llm_pressure = self._build_llm_pressure()

    def _build_llm_pressure(self) -> Optional[LLMPressureScorer]:
        cfg = self.config
        if cfg.llm_pressure_weight <= 0:
            return None
        return LLMPressureScorer(
            spm_model=self._sp,
            spm_vocab_size=self._vocab_size,
            llm_model_name=cfg.llm_model_name,
        )

    # ---- AgentTrainer interface ----

    def trainable_param_groups(self) -> list[dict]:
        cfg = self.config
        groups = [
            {"params": list(self.z_gen.parameters()), "lr": cfg.gru_lr},
            {"params": list(self.target_proj.parameters()), "lr": cfg.receiver_lr},
            {"params": list(self.hidden_proj.parameters()), "lr": cfg.receiver_lr},
            {"params": list(self.surface_encoder.parameters()), "lr": cfg.receiver_lr},
            {"params": [self.log_temperature], "lr": cfg.receiver_lr},
        ]
        if self.llm_pressure is not None:
            groups.append({"params": [self.llm_pressure.projection], "lr": cfg.receiver_lr})
        return groups

    def checkpoint_state(self) -> dict:
        state = {
            "z_gen": self.z_gen.state_dict(),
            "target_proj": self.target_proj.state_dict(),
            "hidden_proj": self.hidden_proj.state_dict(),
            "surface_encoder": self.surface_encoder.state_dict(),
            "log_temperature": self.log_temperature.data,
            "version": 6,
        }
        if self.llm_pressure is not None:
            state["llm_pressure_projection"] = self.llm_pressure.projection.data.cpu()
        return state

    def load_checkpoint_state(self, ckpt: dict) -> None:
        self.z_gen.load_state_dict(ckpt["z_gen"])
        for key, mod in (
            ("target_proj", self.target_proj),
            ("hidden_proj", self.hidden_proj),
            ("surface_encoder", self.surface_encoder),
        ):
            if key in ckpt:
                mod.load_state_dict(ckpt[key])
        if "log_temperature" in ckpt:
            self.log_temperature.data.copy_(ckpt["log_temperature"])
        if self.llm_pressure is not None and "llm_pressure_projection" in ckpt:
            saved = ckpt["llm_pressure_projection"]
            if saved.shape == self.llm_pressure.projection.shape:
                self.llm_pressure.projection.data.copy_(
                    saved.to(self.llm_pressure.projection.device),
                )

    def _distractor_curriculum_ratio(self, step: int) -> float:
        c = self.config.curriculum
        if not c.enabled or c.warmup_steps <= 0:
            return 1.0
        return min(1.0, float(step) / c.warmup_steps)

    # ---- Forward ----

    def forward(
        self,
        anchor: Tensor,
        distractors: Tensor,
        *,
        step: int = 0,
        candidate_indices: Optional[Tensor] = None,
    ) -> dict[str, Tensor]:
        del candidate_indices

        ratio = self._distractor_curriculum_ratio(step)
        n_dist = int(round(ratio * distractors.size(1)))
        active_distractors = distractors[:, :n_dist, :] if n_dist > 0 else None

        expr = self._generate(anchor)
        agg = self._chunked_logit_pass(expr)

        terms, hidden_logits = self._compute_loss_terms(expr, anchor, agg, active_distractors)
        total = self._aggregate(terms)
        self._raise_if_nonfinite(terms)

        out = self._build_output(total, terms, hidden_logits, expr)
        out["n_distractors"] = torch.tensor(float(n_dist))
        return out

    # ---- Chunked output_head pass ----

    def _chunked_logit_pass(self, expr: ExpressionOutput) -> LogitAggregations:
        """Iterate the (B, S, V) output_head pass in chunks along B."""
        device = expr.hidden.device
        B = expr.hidden.size(0)
        chunk = max(self.config.logit_chunk, 1)
        compute_fluency = self.config.decoder_fluency_weight > 0

        bp = self.bigram_kl.zero_partial(device)
        ap = self.adj_diversity.zero_partial(device)
        fluency_sum = torch.zeros((), device=device)
        fluency_count = torch.zeros((), device=device)
        emb_chunks: list[Tensor] = []

        for s in range(0, B, chunk):
            e = min(s + chunk, B)
            h = expr.hidden[s:e]
            m = expr.mask[s:e]
            logits = self.vae.output_head(h)
            probs = F.softmax(logits, dim=-1)

            emb_chunks.append(self._straight_through(logits, probs))
            if self.bigram_kl.is_loaded:
                p = self.bigram_kl.partial(probs, m)
                bp = _BigramPartial(bp.topk_sum + p.topk_sum, bp.n_pairs + p.n_pairs)
            if self.config.adj_diversity_weight > 0:
                p = self.adj_diversity.partial(probs, m)
                ap = _AdjPartial(ap.excess_sum + p.excess_sum, ap.n_pairs + p.n_pairs)
            if compute_fluency:
                tgt = expr.tokens_cpu[s:e].to(device)
                ce = F.cross_entropy(
                    logits.reshape(-1, logits.size(-1)),
                    tgt.reshape(-1),
                    reduction="none",
                ).reshape(e - s, -1)
                fluency_sum = fluency_sum + (ce * m.float()).sum()
                fluency_count = fluency_count + m.float().sum()
            del logits, probs

        decoder_fluency = (
            fluency_sum / fluency_count.clamp(min=1.0)
            if compute_fluency
            else torch.zeros((), device=device)
        )

        return LogitAggregations(
            surface_emb=torch.cat(emb_chunks, dim=0),
            bigram_kl=self.bigram_kl.finalize(bp),
            adj_diversity=self.adj_diversity.finalize(ap),
            decoder_fluency=decoder_fluency,
        )

    # ---- Loss term computation ----

    def _compute_loss_terms(
        self,
        expr: ExpressionOutput,
        anchor: Tensor,
        agg: LogitAggregations,
        distractors: Optional[Tensor] = None,
    ) -> tuple[dict[str, Tensor], Tensor]:
        hidden_msg = self.hidden_proj(self._pool_hidden(expr))
        surface_msg = self._encode_surface(agg.surface_emb, expr.mask)

        hidden_loss, hidden_logits = self._info_nce(hidden_msg, anchor, distractors)
        surface_loss, _ = self._info_nce(surface_msg, anchor, distractors)

        terms: dict[str, Tensor] = {
            "hidden_nce":   hidden_loss,
            "surface_nce":  surface_loss,
            "topology":     self._topology(hidden_msg, anchor),
            "z_div_loss":   self._z_diversity_loss(expr),
            "bigram_kl":    agg.bigram_kl,
            "adj_div":      agg.adj_diversity,
            "dec_fluency":  agg.decoder_fluency,
            "llm_pressure": self._llm_pressure_loss(expr),
            "z_prior":      self._z_prior_loss(expr),
        }
        return terms, hidden_logits

    def _encode_surface(self, surface_emb: Tensor, mask: Tensor) -> Tensor:
        empty = ~mask.any(dim=1)
        if empty.any():
            mask = mask.clone()
            mask[empty, 0] = True
        return self.surface_encoder(surface_emb, mask)

    def _info_nce(
        self,
        msg: Tensor,
        anchor: Tensor,
        distractors: Optional[Tensor] = None,
    ) -> tuple[Tensor, Tensor]:
        msg_n = F.normalize(msg, dim=-1)
        tgt_n = F.normalize(anchor.detach(), dim=-1)
        temperature = self.log_temperature.exp().clamp(min=0.01, max=100.0)

        if distractors is not None:
            B, D, E = distractors.shape
            neg_n = F.normalize(distractors.detach().reshape(B * D, E), dim=-1)
            all_tgt = torch.cat([tgt_n, neg_n], dim=0)
            logits = (msg_n @ all_tgt.t()) / temperature
            labels = torch.arange(msg_n.size(0), device=logits.device)
            loss = F.cross_entropy(logits, labels)
        else:
            logits = (msg_n @ tgt_n.t()) / temperature
            labels = torch.arange(logits.size(0), device=logits.device)
            loss = 0.5 * (F.cross_entropy(logits, labels) + F.cross_entropy(logits.t(), labels))

        return loss, logits

    def _llm_pressure_loss(self, expr: ExpressionOutput) -> Tensor:
        if self.llm_pressure is None:
            return expr.hidden.new_zeros(())
        logits = self.vae.output_head(expr.hidden)
        return self.llm_pressure(
            agent_logits=logits, mask=expr.mask, tau=self.config.llm_gumbel_tau,
        )

    # ---- Aggregation + output ----

    def _loss_weights(self) -> dict[str, float]:
        cfg = self.config
        return {
            "hidden_nce":   cfg.hidden_infonce_weight,
            "surface_nce":  cfg.surface_infonce_weight,
            "topology":     cfg.topology_weight,
            "z_div_loss":   cfg.z_diversity_weight,
            "bigram_kl":    cfg.bigram_kl_weight,
            "adj_div":      cfg.adj_diversity_weight,
            "dec_fluency":  cfg.decoder_fluency_weight,
            "llm_pressure": cfg.llm_pressure_weight,
            "z_prior":      cfg.z_prior_weight,
        }

    def _aggregate(self, terms: dict[str, Tensor]) -> Tensor:
        weights = self._loss_weights()
        return sum(weights[k] * terms[k] for k in weights)

    def _build_output(
        self,
        total: Tensor,
        terms: dict[str, Tensor],
        hidden_logits: Tensor,
        expr: ExpressionOutput,
    ) -> dict[str, Tensor]:
        with torch.no_grad():
            target_idx = torch.arange(hidden_logits.size(0), device=hidden_logits.device)
            accuracy = (hidden_logits.argmax(1) == target_idx).float().mean()
            msg_lengths = expr.mask.float().sum(dim=1).mean()
            surface_unique = self._surface_uniqueness(expr)
            uniq_tok = self._mean_unique_tokens(expr)

        out: dict[str, Tensor] = {
            "loss": total,
            "accuracy": accuracy,
            "msg_lengths": msg_lengths.detach(),
            "num_phrases": expr.num_phrases.mean().detach(),
            "surface_unique": torch.tensor(surface_unique),
            "uniq_tok": torch.tensor(float(uniq_tok)),
            "_tokens": expr.tokens_cpu,
            "_gen_mask": expr.gen_mask_cpu,
        }
        for k, v in terms.items():
            out[k] = v.detach()
        return out

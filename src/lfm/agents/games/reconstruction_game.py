"""Reconstruction expression game over a frozen ``DepTreeVAE`` decoder.

Loss hierarchy
--------------

1. **Primary — reconstruction** (cosine):
   InverseDecoder reads straight-through token embeddings → recovers the
   source embedding.  Mean cosine distance to anchor is the main training
   signal.

2. **Discriminative — surface InfoNCE** (batch NCE):
   Surface encoder tries to identify each anchor from the batch given its
   expression.  Prevents private codes: the expression must carry
   discriminable information, not just recoverable-by-the-decoder information.

3. **Hard constraint — decoder fluency** (cross-entropy):
   CE of Phase-1 greedy tokens under Phase-2 teacher-forced logits.
   Keeps z on the decoder manifold.

4. **Surface regularisers — bigram KL + adj diversity**:
   Penalise deviation from natural bigram distribution and adjacent-token
   cosine similarity hinge.  Prevent degenerate token patterns.

5. **Secondary — topology** (Pearson r):
   Preserves neighbourhood structure in hidden space.

6. **Tertiary — z_diversity + z_prior**:
   Cross-batch z diversity and MMD prior regularisation.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
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
from lfm.generator.dep_tree_vae.model import DepTreeVAE
from lfm.reconstruction.inverse_decoder import InverseDecoder

logger = logging.getLogger(__name__)


# ===========================================================================
# Config
# ===========================================================================


class ReconstructionGameConfig(BaseExpressionGameConfig):
    """Configuration for the reconstruction expression game."""

    # Reconstruction head architecture.
    inverse_decoder_layers: int = 4
    inverse_decoder_heads: int = 8

    # Token concentration penalty (Herfindahl index on per-sample marginal softmax).
    # Penalises sequences where one token dominates the softmax distribution.
    # HHI = sum_v (mean_t p_tv)^2 — minimised at uniform (~1/V), maximised at ~1.0.
    tok_concentration_weight: float = 0.0

    # Surface InfoNCE encoder — prevents private codes.
    encoder: MessageEncoderConfig = MessageEncoderConfig()
    surface_infonce_weight: float = 0.5
    contrastive_temperature: float = 0.07

    # Bigram KL + adj diversity — prevent degenerate surface patterns.
    bigram_kl_path: Optional[str] = None
    bigram_kl_weight: float = 0.05
    adj_diversity_weight: float = 0.05
    adj_diversity_target: float = 0.30

    # Loss weights — reconstruction is the primary driver.
    reconstruction_weight: float = 1.0
    decoder_fluency_weight: float = 0.5
    topology_weight: float = 0.1
    z_diversity_weight: float = 0.1
    z_prior_weight: float = 0.5
    output_dir: str = "data/reconstruction_game_v1"

    # No contrastive distractor pool.
    contrastive_scoring: bool = False


# ===========================================================================
# Internal aggregation dataclass
# ===========================================================================


@dataclass
class _ReconAggregations:
    surface_emb: Tensor       # (B, S, H) straight-through token embeddings
    bigram_kl: Tensor         # scalar
    adj_diversity: Tensor     # scalar
    decoder_fluency: Tensor   # scalar
    phrase_length: Tensor     # scalar
    tok_concentration: Tensor # scalar — Herfindahl index on marginal token softmax


# ===========================================================================
# Game
# ===========================================================================


class ReconstructionGame(BaseExpressionGame):
    """Reconstruction expression game over a frozen ``DepTreeVAE``.

    Public interface used by ``AgentTrainer``:
      * ``forward(anchor, distractors, *, step, candidate_indices)``
      * ``render_surface(token_ids, mask, eos_id, output_mode)``
      * ``trainable_param_groups()``
      * ``checkpoint_state()`` / ``load_checkpoint_state(ckpt)``
    """

    def __init__(self, config: ReconstructionGameConfig, vae: DepTreeVAE) -> None:
        super().__init__(config, vae)

        self.inverse_decoder = InverseDecoder(
            token_dim=self._hidden_dim,
            output_dim=config.embedding_dim,
            num_heads=config.inverse_decoder_heads,
            num_layers=config.inverse_decoder_layers,
        )
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

    # ---- AgentTrainer interface ----

    def trainable_param_groups(self) -> list[dict]:
        cfg = self.config
        return [
            {"params": list(self.z_gen.parameters()), "lr": cfg.gru_lr},
            {"params": (
                list(self.target_proj.parameters())
                + list(self.inverse_decoder.parameters())
                + list(self.surface_encoder.parameters())
                + [self.log_temperature]
            ), "lr": cfg.receiver_lr},
        ]

    def checkpoint_state(self) -> dict:
        return {
            "z_gen": self.z_gen.state_dict(),
            "target_proj": self.target_proj.state_dict(),
            "inverse_decoder": self.inverse_decoder.state_dict(),
            "surface_encoder": self.surface_encoder.state_dict(),
            "log_temperature": self.log_temperature.data,
            "version": 2,
        }

    def load_checkpoint_state(self, ckpt: dict) -> None:
        self.z_gen.load_state_dict(ckpt["z_gen"])
        for key, mod in (
            ("target_proj", self.target_proj),
            ("inverse_decoder", self.inverse_decoder),
            ("surface_encoder", self.surface_encoder),
        ):
            if key in ckpt:
                mod.load_state_dict(ckpt[key])
        if "log_temperature" in ckpt:
            self.log_temperature.data.copy_(ckpt["log_temperature"])

    # ---- Forward ----

    def forward(
        self,
        anchor: Tensor,
        distractors: Tensor,
        *,
        step: int = 0,
        candidate_indices: Optional[Tensor] = None,
    ) -> dict[str, Tensor]:
        del distractors, candidate_indices

        expr = self._generate(anchor)
        terms = self._compute_loss_terms(expr, anchor)
        total = self._aggregate(terms)
        self._raise_if_nonfinite(terms)
        return self._build_output(total, terms, expr)

    # ---- Loss terms ----

    def _compute_loss_terms(
        self,
        expr: ExpressionOutput,
        anchor: Tensor,
    ) -> dict[str, Tensor]:
        agg = self._chunked_output_pass(expr)
        recon_cos = self._cosine_reconstruction(agg.surface_emb, expr.mask, anchor)
        surface_msg = self._encode_surface(agg.surface_emb, expr.mask)
        surface_nce, _ = self._info_nce(surface_msg, anchor)
        pooled = self._pool_hidden(expr)

        return {
            "reconstruction": recon_cos,
            "surface_nce":    surface_nce,
            "dec_fluency":    agg.decoder_fluency,
            "phrase_length":  agg.phrase_length,
            "bigram_kl":      agg.bigram_kl,
            "adj_div":        agg.adj_diversity,
            "tok_conc":       agg.tok_concentration,
            "topology":       self._topology(pooled, anchor),
            "z_div_loss":     self._z_diversity_loss(expr),
            "z_prior":        self._z_prior_loss(expr),
        }

    def _chunked_output_pass(self, expr: ExpressionOutput) -> _ReconAggregations:
        """Run output_head in chunks; accumulate ST embeddings + all surface losses."""
        device = expr.hidden.device
        B = expr.hidden.size(0)
        chunk = max(self.config.logit_chunk, 1)
        cfg = self.config

        K = cfg.max_phrases
        T_actual = expr.hidden.size(1) // K if K > 0 else expr.hidden.size(1)
        compute_fluency = cfg.decoder_fluency_weight > 0
        compute_phr_len = cfg.phrase_length_weight > 0 and cfg.target_tokens_per_phrase > 0

        bp = self.bigram_kl.zero_partial(device)
        ap = self.adj_diversity.zero_partial(device)
        fluency_sum = torch.zeros((), device=device)
        fluency_count = torch.zeros((), device=device)
        plen_sum = torch.zeros((), device=device)
        plen_count = torch.zeros((), device=device)
        hhi_sum = torch.zeros((), device=device)
        hhi_count = 0
        compute_hhi = cfg.tok_concentration_weight > 0
        st_chunks: list[Tensor] = []

        for s in range(0, B, chunk):
            e = min(s + chunk, B)
            h = expr.hidden[s:e]
            m = expr.mask[s:e]
            logits = self.vae.output_head(h)
            probs = F.softmax(logits, dim=-1)

            st_chunks.append(self._straight_through(logits, probs))

            if self.bigram_kl.is_loaded:
                p = self.bigram_kl.partial(probs, m)
                bp = _BigramPartial(bp.topk_sum + p.topk_sum, bp.n_pairs + p.n_pairs)
            if cfg.adj_diversity_weight > 0:
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
            if compute_phr_len:
                c = e - s
                ps, pc = self._phrase_length_loss_chunk(
                    logits.reshape(c * K, T_actual, -1),
                    m.reshape(c * K, T_actual),
                    float(min(cfg.target_tokens_per_phrase, T_actual)),
                )
                plen_sum = plen_sum + ps
                plen_count = plen_count + pc

            if compute_hhi:
                # Herfindahl index on per-sample marginal token distribution.
                # q_i = mean over valid positions of softmax probs for sample i.
                # HHI_i = ||q_i||^2. Minimised at uniform (~1/V), ~1 when all mass
                # on one token. Adding as loss penalises token repetition.
                n_valid = m.float().sum(dim=1, keepdim=True).clamp(min=1)
                q = (probs * m.unsqueeze(-1).float()).sum(dim=1) / n_valid  # (c, V)
                hhi_sum = hhi_sum + (q * q).sum(dim=-1).sum()
                hhi_count += e - s

            del logits, probs

        fluency = (
            fluency_sum / fluency_count.clamp(min=1.0)
            if compute_fluency
            else torch.zeros((), device=device)
        )
        if compute_phr_len:
            mean_len = plen_sum / plen_count.clamp(min=1.0)
            target_len = float(min(cfg.target_tokens_per_phrase, T_actual))
            phr_len_loss = (mean_len - target_len).pow(2)
        else:
            phr_len_loss = torch.zeros((), device=device)

        tok_conc = (
            hhi_sum / max(hhi_count, 1)
            if compute_hhi
            else torch.zeros((), device=device)
        )

        return _ReconAggregations(
            surface_emb=torch.cat(st_chunks, dim=0),
            bigram_kl=self.bigram_kl.finalize(bp),
            adj_diversity=self.adj_diversity.finalize(ap),
            decoder_fluency=fluency,
            phrase_length=phr_len_loss,
            tok_concentration=tok_conc,
        )

    def _cosine_reconstruction(
        self, surface_emb: Tensor, mask: Tensor, anchor: Tensor,
    ) -> Tensor:
        """InverseDecoder on straight-through token embeddings → cosine loss."""
        empty = ~mask.any(dim=1)
        if empty.any():
            mask = mask.clone()
            mask[empty, 0] = True
        reconstructed = self.inverse_decoder(surface_emb, mask)
        recon_n = F.normalize(reconstructed, dim=-1)
        anchor_n = F.normalize(anchor.detach(), dim=-1)
        return 1.0 - (recon_n * anchor_n).sum(dim=-1).mean()

    def _encode_surface(self, surface_emb: Tensor, mask: Tensor) -> Tensor:
        empty = ~mask.any(dim=1)
        if empty.any():
            mask = mask.clone()
            mask[empty, 0] = True
        return self.surface_encoder(surface_emb, mask)

    def _info_nce(self, msg: Tensor, anchor: Tensor) -> tuple[Tensor, Tensor]:
        """Symmetric batch InfoNCE (no external distractors pool)."""
        msg_n = F.normalize(msg, dim=-1)
        tgt_n = F.normalize(anchor.detach(), dim=-1)
        temperature = self.log_temperature.exp().clamp(min=0.01, max=100.0)
        logits = (msg_n @ tgt_n.t()) / temperature
        labels = torch.arange(logits.size(0), device=logits.device)
        loss = 0.5 * (F.cross_entropy(logits, labels) + F.cross_entropy(logits.t(), labels))
        return loss, logits

    def _aggregate(self, terms: dict[str, Tensor]) -> Tensor:
        cfg = self.config
        weights = {
            "reconstruction": cfg.reconstruction_weight,
            "surface_nce":    cfg.surface_infonce_weight,
            "dec_fluency":    cfg.decoder_fluency_weight,
            "phrase_length":  cfg.phrase_length_weight,
            "bigram_kl":      cfg.bigram_kl_weight,
            "adj_div":        cfg.adj_diversity_weight,
            "tok_conc":       cfg.tok_concentration_weight,
            "topology":       cfg.topology_weight,
            "z_div_loss":     cfg.z_diversity_weight,
            "z_prior":        cfg.z_prior_weight,
        }
        return sum(weights[k] * terms[k] for k in weights)

    def _build_output(
        self,
        total: Tensor,
        terms: dict[str, Tensor],
        expr: ExpressionOutput,
    ) -> dict[str, Tensor]:
        with torch.no_grad():
            recon_cos = terms["reconstruction"].detach()
            accuracy = (1.0 - recon_cos).clamp(0.0, 1.0)
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

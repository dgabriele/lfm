"""Skeleton decoder — generates dependency role sequences from z_struct.

Two implementations behind a common interface:
  - ``ParallelSkeletonDecoder``: MLP predicts all positions + length in one shot.
  - ``ARSkeletonDecoder``: Autoregressive transformer decoder.

Selected via ``SkeletonDecoderConfig.mode`` ("parallel" or "ar").
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from lfm.generator.dep_tree_vae.config import (
    NUM_DEP_RELATIONS,
    SkeletonDecoderConfig,
)

# Special tokens for the skeleton vocabulary (dep relation sequence).
SKEL_PAD = 0
SKEL_BOS = NUM_DEP_RELATIONS
SKEL_EOS = NUM_DEP_RELATIONS + 1
SKEL_VOCAB_SIZE = NUM_DEP_RELATIONS + 2


class SkeletonDecoderBase(nn.Module):
    """Common interface for skeleton decoders."""

    def forward(
        self,
        z_struct: Tensor,
        target_roles: Tensor | None = None,
        target_lengths: Tensor | None = None,
    ) -> tuple[Tensor, Tensor | None]:
        """Generate or teacher-force a role sequence.

        Args:
            z_struct: ``(B, struct_dim)``
            target_roles: ``(B, S)`` target role ids (with BOS prefix)
                for teacher forcing. None for free generation.
            target_lengths: ``(B,)`` valid lengths in target_roles.

        Returns:
            logits_or_tokens: Logits ``(B, S, V)`` if training, token
                ids ``(B, S)`` if generating.
            loss: Scalar CE loss if targets provided, else None.
        """
        raise NotImplementedError


class ParallelSkeletonDecoder(SkeletonDecoderBase):
    """One-shot parallel prediction of the full role sequence.

    Predicts all positions simultaneously from z_struct via an MLP.
    Also predicts sequence length.  Fast and sufficient for short
    sequences (~5-30 roles) from a tiny vocabulary (~50 types).
    """

    def __init__(self, cfg: SkeletonDecoderConfig, struct_dim: int) -> None:
        super().__init__()
        h = cfg.hidden_dim
        self.max_roles = cfg.max_roles

        self.role_predictor = nn.Sequential(
            nn.Linear(struct_dim, h),
            nn.GELU(),
            nn.LayerNorm(h),
            nn.Linear(h, h),
            nn.GELU(),
            nn.Linear(h, cfg.max_roles * NUM_DEP_RELATIONS),
        )

        self.length_predictor = nn.Sequential(
            nn.Linear(struct_dim, h),
            nn.GELU(),
            nn.Linear(h, cfg.max_roles),
        )

    def forward(
        self,
        z_struct: Tensor,
        target_roles: Tensor | None = None,
        target_lengths: Tensor | None = None,
    ) -> tuple[Tensor, Tensor | None]:
        b = z_struct.size(0)
        device = z_struct.device

        # (B, max_roles, NUM_DEP_RELATIONS)
        role_logits = self.role_predictor(z_struct).view(
            b, self.max_roles, NUM_DEP_RELATIONS,
        )
        length_logits = self.length_predictor(z_struct)  # (B, max_roles)

        if target_roles is None:
            return self._generate(role_logits, length_logits), None

        # Teacher-forced loss: compare predicted roles to targets.
        # target_roles has BOS prefix — strip it for comparison.
        # The target sequence is: BOS r0 r1 ... rN EOS
        # We predict positions 0..max_roles-1 corresponding to r0..rN.
        targets_no_bos = target_roles[:, 1:]  # strip BOS
        seq_len = targets_no_bos.size(1)

        # Pad or truncate role_logits to match target length
        if seq_len <= self.max_roles:
            logits_slice = role_logits[:, :seq_len, :]
        else:
            logits_slice = F.pad(
                role_logits, (0, 0, 0, seq_len - self.max_roles),
            )

        # Mask out positions beyond actual length
        # target_lengths includes BOS, so actual role count = target_lengths - 1
        # But we also need to account for EOS in the target
        actual_lens = target_lengths - 1 if target_lengths is not None else None

        if actual_lens is not None:
            mask = (
                torch.arange(seq_len, device=device).unsqueeze(0)
                < actual_lens.unsqueeze(1)
            )
            # Only compute loss on EOS-stripped roles (exclude EOS token itself)
            # Clamp targets to valid role range
            clamped_targets = targets_no_bos.clamp(max=NUM_DEP_RELATIONS - 1)
            flat_logits = logits_slice.reshape(-1, NUM_DEP_RELATIONS)
            flat_targets = clamped_targets.reshape(-1)
            flat_mask = mask.reshape(-1)
            role_loss = F.cross_entropy(
                flat_logits[flat_mask], flat_targets[flat_mask],
            )
        else:
            role_loss = F.cross_entropy(
                logits_slice.reshape(-1, NUM_DEP_RELATIONS),
                targets_no_bos.clamp(max=NUM_DEP_RELATIONS - 1).reshape(-1),
            )

        # Length loss
        if actual_lens is not None:
            # actual_lens counts roles (excluding BOS/EOS)
            length_targets = actual_lens.clamp(max=self.max_roles - 1)
            length_loss = F.cross_entropy(length_logits, length_targets)
        else:
            length_loss = torch.tensor(0.0, device=device)

        total_loss = role_loss + 0.1 * length_loss
        return role_logits, total_loss

    @torch.no_grad()
    def _generate(self, role_logits: Tensor, length_logits: Tensor) -> Tensor:
        b = role_logits.size(0)
        device = role_logits.device

        lengths = length_logits.argmax(dim=-1) + 1  # at least 1
        roles = role_logits.argmax(dim=-1)  # (B, max_roles)

        # Build BOS + roles + EOS sequence
        tokens = torch.full(
            (b, self.max_roles + 2), SKEL_PAD,
            dtype=torch.long, device=device,
        )
        tokens[:, 0] = SKEL_BOS
        for i in range(b):
            n = min(lengths[i].item(), self.max_roles)
            tokens[i, 1 : n + 1] = roles[i, :n]
            tokens[i, n + 1] = SKEL_EOS
        return tokens


class ARSkeletonDecoder(SkeletonDecoderBase):
    """Autoregressive transformer decoder for role sequences.

    Generates roles left-to-right conditioned on z_struct as
    cross-attention memory.
    """

    def __init__(self, cfg: SkeletonDecoderConfig, struct_dim: int) -> None:
        super().__init__()
        h = cfg.hidden_dim

        self.role_embedding = nn.Embedding(SKEL_VOCAB_SIZE, h)
        self.pos_embedding = nn.Embedding(cfg.max_roles + 2, h)
        self.z_proj = nn.Linear(struct_dim, h)

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=h,
            nhead=cfg.num_heads,
            dim_feedforward=h * 4,
            dropout=cfg.dropout,
            batch_first=True,
            norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(
            decoder_layer, num_layers=cfg.num_layers,
        )
        self.output_head = nn.Linear(h, SKEL_VOCAB_SIZE)
        self.max_roles = cfg.max_roles

    def forward(
        self,
        z_struct: Tensor,
        target_roles: Tensor | None = None,
        target_lengths: Tensor | None = None,
    ) -> tuple[Tensor, Tensor | None]:
        memory = self.z_proj(z_struct).unsqueeze(1)

        if target_roles is not None:
            return self._teacher_forced(memory, target_roles, target_lengths)
        return self._greedy(memory), None

    def _teacher_forced(
        self,
        memory: Tensor,
        targets: Tensor,
        lengths: Tensor | None,
    ) -> tuple[Tensor, Tensor]:
        b, s = targets.shape
        device = targets.device

        pos = torch.arange(s, device=device).unsqueeze(0)
        causal_mask = nn.Transformer.generate_square_subsequent_mask(
            s, device=device,
        )

        x = self.role_embedding(targets) + self.pos_embedding(pos)
        h = self.decoder(x, memory, tgt_mask=causal_mask)
        logits = self.output_head(h)

        if lengths is not None:
            mask = torch.arange(s, device=device).unsqueeze(0) < lengths.unsqueeze(1)
            flat_logits = logits[:, :-1].reshape(-1, logits.size(-1))
            flat_targets = targets[:, 1:].reshape(-1)
            flat_mask = mask[:, 1:].reshape(-1)
            loss = F.cross_entropy(
                flat_logits[flat_mask], flat_targets[flat_mask],
            )
        else:
            loss = F.cross_entropy(
                logits[:, :-1].reshape(-1, logits.size(-1)),
                targets[:, 1:].reshape(-1),
            )
        return logits, loss

    @torch.no_grad()
    def _greedy(self, memory: Tensor) -> Tensor:
        b = memory.size(0)
        device = memory.device

        tokens = torch.full(
            (b, 1), SKEL_BOS, dtype=torch.long, device=device,
        )
        for _ in range(self.max_roles):
            s = tokens.size(1)
            pos = torch.arange(s, device=device).unsqueeze(0)
            causal_mask = nn.Transformer.generate_square_subsequent_mask(
                s, device=device,
            )
            x = self.role_embedding(tokens) + self.pos_embedding(pos)
            h = self.decoder(x, memory, tgt_mask=causal_mask)
            next_tok = self.output_head(h[:, -1:]).argmax(dim=-1)
            tokens = torch.cat([tokens, next_tok], dim=1)
            if (next_tok == SKEL_EOS).all():
                break
        return tokens


def build_skeleton_decoder(
    cfg: SkeletonDecoderConfig, struct_dim: int,
) -> SkeletonDecoderBase:
    """Factory: build the configured skeleton decoder variant."""
    if cfg.mode == "parallel":
        return ParallelSkeletonDecoder(cfg, struct_dim)
    elif cfg.mode == "ar":
        return ARSkeletonDecoder(cfg, struct_dim)
    else:
        raise ValueError(f"Unknown skeleton mode: {cfg.mode!r}")

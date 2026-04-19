"""Skeleton decoder — generates dependency role sequences from z_struct."""

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


class SkeletonDecoder(nn.Module):
    """Lightweight autoregressive decoder for dependency role sequences.

    Takes z_struct as a cross-attention memory and generates a sequence
    of dependency relation tokens: ``[BOS] nsubj root obj prep det pobj [EOS]``.

    This is a small model (~200K params) since the role vocabulary is
    tiny (~50 relations) and sequences are short (max ~40 roles).
    """

    def __init__(self, cfg: SkeletonDecoderConfig, struct_dim: int) -> None:
        super().__init__()
        h = cfg.hidden_dim

        self.role_embedding = nn.Embedding(SKEL_VOCAB_SIZE, h)
        self.pos_embedding = nn.Embedding(cfg.max_roles + 1, h)

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
        """Forward pass with optional teacher forcing.

        Args:
            z_struct: ``(B, struct_dim)`` structural latent.
            target_roles: ``(B, S)`` target role ids for teacher forcing.
                Includes BOS prefix but not EOS suffix.
            target_lengths: ``(B,)`` valid lengths in target_roles.

        Returns:
            logits: ``(B, S, SKEL_VOCAB_SIZE)``
            loss: scalar CE loss if targets provided, else None.
        """
        memory = self.z_proj(z_struct).unsqueeze(1)  # (B, 1, h)

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

        # Shift: predict next role from current prefix
        # logits[:, t, :] should predict targets[:, t+1]
        # But targets already has BOS prepended, so we predict
        # the role sequence shifted by one.
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

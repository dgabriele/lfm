"""Phrase-level z projector for DepTreeVAE.

Derives a per-word latent from z_content conditioned on the
dependency role and position.  The frozen PhraseDecoder receives
this projected latent as its memory/conditioning signal.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor

from lfm.generator.dep_tree_vae.config import NUM_DEP_RELATIONS


class PhraseZProjector(nn.Module):
    """Map (z_content, role, position) → per-position decoder memory.

    For each token position in the output sequence, the projector
    combines the global content latent with a role embedding and a
    position embedding to produce a conditioning vector for the
    PhraseDecoder's cross-attention.

    This allows the same z_content to produce different words
    depending on syntactic role — a verb in the ROOT slot vs
    a noun in the nsubj slot.
    """

    def __init__(
        self,
        content_dim: int,
        decoder_hidden_dim: int,
        max_seq_len: int = 80,
    ) -> None:
        super().__init__()
        self.role_embedding = nn.Embedding(NUM_DEP_RELATIONS, content_dim)
        self.pos_embedding = nn.Embedding(max_seq_len, content_dim)
        self.proj = nn.Sequential(
            nn.Linear(content_dim * 3, decoder_hidden_dim),
            nn.GELU(),
            nn.Linear(decoder_hidden_dim, decoder_hidden_dim),
        )

    def forward(
        self,
        z_content: Tensor,
        role_ids: Tensor,
        positions: Tensor,
    ) -> Tensor:
        """Project z_content into per-position decoder memory.

        Args:
            z_content: ``(B, content_dim)`` content latent.
            role_ids: ``(B, S)`` dependency role id per position.
            positions: ``(B, S)`` position indices.

        Returns:
            memory: ``(B, S, decoder_hidden_dim)`` — cross-attention
                memory for the PhraseDecoder, one vector per output
                position.
        """
        b, s = role_ids.shape

        role_emb = self.role_embedding(role_ids)           # (B, S, content_dim)
        pos_emb = self.pos_embedding(positions)            # (B, S, content_dim)
        z_exp = z_content.unsqueeze(1).expand(-1, s, -1)   # (B, S, content_dim)

        combined = torch.cat([z_exp, role_emb, pos_emb], dim=-1)
        return self.proj(combined)

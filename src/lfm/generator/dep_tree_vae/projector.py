"""Role-level memory projector for DepTreeVAE.

Produces one memory vector per dependency role in the skeleton.
The PhraseDecoder cross-attends to this role memory during
autoregressive generation, dynamically selecting which role
it's currently filling — no forced positional alignment needed.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor

from lfm.generator.dep_tree_vae.config import NUM_DEP_RELATIONS


class PhraseZProjector(nn.Module):
    """Map (z_content, role_ids) → per-role decoder memory.

    Each role in the skeleton gets a memory vector that combines
    the global content latent with a role-specific embedding.
    The decoder cross-attends to these role memories, learning
    to shift attention as it fills successive roles.

    Output shape: ``(B, num_roles, decoder_hidden_dim)`` — one
    vector per role, not per output token.
    """

    def __init__(
        self,
        content_dim: int,
        decoder_hidden_dim: int,
        max_roles: int = 40,
    ) -> None:
        super().__init__()
        self.role_embedding = nn.Embedding(NUM_DEP_RELATIONS, content_dim)
        self.pos_embedding = nn.Embedding(max_roles, content_dim)
        self.proj = nn.Sequential(
            nn.Linear(content_dim * 3, decoder_hidden_dim),
            nn.GELU(),
            nn.Linear(decoder_hidden_dim, decoder_hidden_dim),
        )

    def forward(
        self,
        z_content: Tensor,
        role_ids: Tensor,
        role_mask: Tensor | None = None,
    ) -> Tensor:
        """Project z_content into per-role decoder memory.

        Args:
            z_content: ``(B, content_dim)`` content latent.
            role_ids: ``(B, R)`` dependency role id per skeleton position.
            role_mask: ``(B, R)`` optional bool mask (True = valid role).

        Returns:
            memory: ``(B, R, decoder_hidden_dim)`` — one vector per role
                for the PhraseDecoder to cross-attend to.
        """
        b, r = role_ids.shape
        device = role_ids.device

        positions = torch.arange(r, device=device).unsqueeze(0).expand(b, -1)
        role_emb = self.role_embedding(role_ids.clamp(max=NUM_DEP_RELATIONS - 1))
        pos_emb = self.pos_embedding(positions)
        z_exp = z_content.unsqueeze(1).expand(-1, r, -1)

        combined = torch.cat([z_exp, role_emb, pos_emb], dim=-1)
        return self.proj(combined)

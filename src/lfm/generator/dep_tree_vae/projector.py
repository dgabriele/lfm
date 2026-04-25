"""Role-level memory projector for DepTreeVAE.

Produces one memory vector per dependency role in the skeleton.
The PhraseDecoder cross-attends to this role memory during
autoregressive generation, dynamically selecting which role
it's currently filling — no forced positional alignment needed.

Content and structure are projected through separate pathways
then combined additively, so the content signal has a dedicated
gradient path from z_content.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor

from lfm.generator.dep_tree_vae.config import NUM_DEP_RELATIONS


class PhraseZProjector(nn.Module):
    """Map (z_content, role_ids) → per-role decoder memory.

    Separate pathways for content and structure:
      - content_proj: z_content → per-role content signal
      - role_proj + pos_proj: role/position → structural conditioning

    Combined additively so gradients from content losses flow
    cleanly to z_content without being diluted by role/position.
    """

    def __init__(
        self,
        content_dim: int,
        decoder_hidden_dim: int,
        max_roles: int = 40,
    ) -> None:
        super().__init__()
        self.role_embedding = nn.Embedding(NUM_DEP_RELATIONS, decoder_hidden_dim)
        self.pos_embedding = nn.Embedding(max_roles, decoder_hidden_dim)
        self.content_proj = nn.Sequential(
            nn.Linear(content_dim, decoder_hidden_dim),
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

        content_mem = self.content_proj(z_content).unsqueeze(1).expand(-1, r, -1)
        role_cond = self.role_embedding(role_ids.clamp(max=NUM_DEP_RELATIONS - 1))
        pos_cond = self.pos_embedding(positions)

        return content_mem + role_cond + pos_cond

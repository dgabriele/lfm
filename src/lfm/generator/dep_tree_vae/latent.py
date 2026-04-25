"""Latent space management for DepTreeVAE."""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor

from lfm.generator.dep_tree_vae.config import LatentConfig


class LatentSpace(nn.Module):
    """Single latent vector exposed to all downstream modules.

    The historical struct/content split has been collapsed; both the
    SkeletonDecoder and PhraseZProjector now read the full ``z``. The
    ``split`` API is preserved (returning ``(z, z)``) so call sites that
    still destructure into ``z_struct, z_content`` remain valid — both
    names alias the same tensor.
    """

    def __init__(self, cfg: LatentConfig) -> None:
        super().__init__()
        self.struct_dim = cfg.struct_dim
        self.content_dim = cfg.content_dim
        self.total_dim = cfg.total_dim

    def reparameterize(self, mu: Tensor, logvar: Tensor) -> Tensor:
        """Sample z from N(mu, sigma) using the reparameterization trick."""
        if self.training:
            std = (0.5 * logvar).exp()
            eps = torch.randn_like(std)
            return mu + eps * std
        return mu

    def split(self, z: Tensor) -> tuple[Tensor, Tensor]:
        """Return the full latent twice — split is now a no-op."""
        return z, z

    def forward(
        self, mu: Tensor, logvar: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Reparameterize and return the full latent (aliased twice).

        Returns:
            z_struct: ``(B, total_dim)`` — alias for z.
            z_content: ``(B, total_dim)`` — alias for z.
            z: ``(B, total_dim)`` — full vector for KL computation.
        """
        z = self.reparameterize(mu, logvar)
        return z, z, z

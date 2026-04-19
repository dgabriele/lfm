"""Latent space management for DepTreeVAE."""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor

from lfm.generator.dep_tree_vae.config import LatentConfig


class LatentSpace(nn.Module):
    """Split latent vector into structural and content subspaces.

    Handles reparameterization and provides clean accessors for
    z_struct and z_content.
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
        """Split z into (z_struct, z_content)."""
        return z[:, : self.struct_dim], z[:, self.struct_dim :]

    def merge(self, z_struct: Tensor, z_content: Tensor) -> Tensor:
        """Merge z_struct and z_content back into a full z."""
        return torch.cat([z_struct, z_content], dim=-1)

    def forward(
        self, mu: Tensor, logvar: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Reparameterize and split.

        Returns:
            z_struct: ``(B, struct_dim)``
            z_content: ``(B, content_dim)``
            z: ``(B, total_dim)`` — full vector for KL computation.
        """
        z = self.reparameterize(mu, logvar)
        z_struct, z_content = self.split(z)
        return z_struct, z_content, z

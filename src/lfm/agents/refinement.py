"""Refinement denoiser — replace Phase 2 decoder re-run with lightweight diffusion.

Takes token embeddings from Phase 1 AR decode as a "rough draft" and
refines them via iterative denoising conditioned on z memories.  Produces
hidden states with gradients flowing to z, same interface as
``rerun_decoder_multiphrase_with_grad`` but at ~10x lower VRAM.

Supports reverse mode for bidirectional communication: given observed
IPA token embeddings, denoise to recover the z that produced them.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn

from lfm.agents.diffusion import DenoiserBlock, sinusoidal_embedding


class RefinementDenoiser(nn.Module):
    """Refine AR decoder token embeddings via iterative denoising.

    Replaces Phase 2 (``rerun_decoder_multiphrase_with_grad``).  The frozen
    decoder generates tokens in Phase 1; this module produces refined
    hidden representations from those tokens, conditioned on z memories
    via cross-attention.

    Args:
        hidden_dim: Token embedding / hidden state dimensionality.
        num_layers: Denoiser transformer blocks.
        num_heads: Attention heads per block.
        num_steps: Refinement diffusion steps (T).
    """

    def __init__(
        self,
        hidden_dim: int = 512,
        num_layers: int = 4,
        num_heads: int = 8,
        num_steps: int = 4,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_steps = num_steps

        # Timestep embedding
        self.time_embed = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # Denoiser blocks: self-attention across token positions +
        # cross-attention to z memories
        self.layers = nn.ModuleList([
            DenoiserBlock(hidden_dim, num_heads, hidden_dim * 4)
            for _ in range(num_layers)
        ])

        # Output projection (residual from input)
        self.out_norm = nn.LayerNorm(hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)

    def forward(
        self,
        token_embeddings: Tensor,
        z_memories: Tensor,
        mask: Tensor,
    ) -> Tensor:
        """Refine token embeddings conditioned on z memories.

        Args:
            token_embeddings: ``(B, T, hidden_dim)`` from frozen
                token_embedding lookup (detached).
            z_memories: ``(B, M, hidden_dim)`` from
                ``latent_to_decoder(weighted_z)``.  Has gradients.
            mask: ``(B, T)`` boolean mask (True = valid).

        Returns:
            ``(B, T, hidden_dim)`` refined hidden states with gradients
            flowing to z_memories.
        """
        B, T, H = token_embeddings.shape
        device = token_embeddings.device

        # Start from token embeddings (detached — no grad to frozen decoder)
        h = token_embeddings

        # Noise schedule: decreasing noise over T steps
        for step in range(self.num_steps):
            t_val = 1.0 - step / self.num_steps
            noise_scale = 0.1 * t_val

            # Add noise (not at final step)
            if step < self.num_steps - 1:
                h = h + noise_scale * torch.randn_like(h)

            # Timestep embedding broadcast to all positions
            t_tensor = torch.full((B,), t_val, device=device)
            t_embed = self.time_embed(sinusoidal_embedding(t_tensor, H))
            h = h + t_embed.unsqueeze(1)

            # Run through denoiser blocks
            # Self-attention across token positions,
            # cross-attention to z memories
            for layer in self.layers:
                h = layer(h, z_memories)

        # Output projection with residual from input
        h = self.out_norm(h)
        h = token_embeddings + self.out_proj(h)

        return h

    def reverse(
        self,
        token_embeddings: Tensor,
        mask: Tensor,
        num_steps: int = 8,
    ) -> Tensor:
        """Reverse mode: recover z-like representations from tokens.

        Given observed IPA token embeddings, iteratively refine to
        produce representations that can be projected back to z-space.
        This is the bidirectional communication path.

        Args:
            token_embeddings: ``(B, T, hidden_dim)`` observed IPA tokens.
            mask: ``(B, T)`` boolean mask.
            num_steps: Reverse diffusion steps (more = higher quality).

        Returns:
            ``(B, T, hidden_dim)`` refined representations suitable for
            z reconstruction via a learned projection.
        """
        # Future: implement reverse diffusion for IPA → z reconstruction
        raise NotImplementedError(
            "Reverse mode not yet implemented. "
            "See docs/roadmap.md §10 for the design."
        )

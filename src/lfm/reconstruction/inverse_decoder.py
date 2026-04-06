"""Inverse decoder: map IPA token representations back to embeddings.

Reads the straight-through differentiable token sequence produced by
the frozen decoder and reconstructs the original input embedding.
This is the learned inverse of the linguistic bottleneck — it recovers
what was encoded from what the LLM would see.

The architecture mirrors a transformer encoder with learned query
readout, similar to MessageEncoder but serving the distinct purpose
of embedding reconstruction rather than discriminative scoring.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn


class InverseDecoder(nn.Module):
    """Reconstruct input embeddings from IPA token representations.

    Args:
        token_dim: Dimension of input token representations (decoder hidden dim).
        output_dim: Dimension of reconstructed embedding.
        num_heads: Attention heads per layer.
        num_layers: Transformer encoder layers.
    """

    def __init__(
        self,
        token_dim: int,
        output_dim: int,
        num_heads: int = 8,
        num_layers: int = 4,
    ) -> None:
        super().__init__()

        # Self-attention over token sequence
        layer = nn.TransformerEncoderLayer(
            token_dim, num_heads, token_dim * 4,
            batch_first=True, norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers)

        # Learned query readout → fixed-size output
        self.query = nn.Parameter(torch.randn(1, 1, token_dim) * 0.02)
        self.readout = nn.MultiheadAttention(
            token_dim, num_heads, batch_first=True,
        )
        self.readout_norm = nn.LayerNorm(token_dim)

        # Project to embedding space
        self.output_proj = nn.Linear(token_dim, output_dim)

    def forward(self, tokens: Tensor, mask: Tensor) -> Tensor:
        """Reconstruct embedding from token representations.

        Args:
            tokens: ``(B, S, token_dim)`` straight-through token embeddings.
            mask: ``(B, S)`` boolean (True = valid token).

        Returns:
            ``(B, output_dim)`` reconstructed embedding.
        """
        pad_mask = ~mask

        # Self-attention over token sequence
        h = self.encoder(tokens, src_key_padding_mask=pad_mask)

        # Learned query cross-attention readout
        q = self.query.expand(tokens.size(0), -1, -1)
        attended, _ = self.readout(q, h, h, key_padding_mask=pad_mask)
        out = self.readout_norm(attended.squeeze(1))

        return self.output_proj(out)

"""Inverse decoder: map IPA token representations back to embeddings.

Two-stage architecture:

1. **HiddenStatePredictor**: reads surface token embeddings and predicts
   the frozen decoder's hidden states at each position.  Supervised by
   MSE against the actual hidden states from Phase 2.  This learns to
   "undo" the argmax — recovering the rich continuous representation
   from the discrete tokens the LLM would see.

2. **InverseDecoder**: reads the predicted hidden states and reconstructs
   the original input embedding via learned query cross-attention readout.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn


class HiddenStatePredictor(nn.Module):
    """Predict decoder hidden states from surface token embeddings.

    A transformer encoder that maps the discrete token representations
    (what the LLM sees) to the decoder's internal hidden states (the
    rich continuous signal that carries discriminative information).

    Args:
        token_dim: Dimension of token embeddings / hidden states.
        num_heads: Attention heads per layer.
        num_layers: Transformer encoder layers.
    """

    def __init__(
        self,
        token_dim: int,
        num_heads: int = 8,
        num_layers: int = 4,
    ) -> None:
        super().__init__()
        layer = nn.TransformerEncoderLayer(
            token_dim, num_heads, token_dim * 4,
            batch_first=True, norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers)

    def forward(self, tokens: Tensor, mask: Tensor) -> Tensor:
        """Predict hidden states from token embeddings.

        Args:
            tokens: ``(B, S, token_dim)`` surface token embeddings.
            mask: ``(B, S)`` boolean (True = valid token).

        Returns:
            ``(B, S, token_dim)`` predicted hidden states.
        """
        return self.encoder(tokens, src_key_padding_mask=~mask)


class InverseDecoder(nn.Module):
    """Reconstruct input embeddings from hidden state representations.

    Args:
        token_dim: Dimension of input representations (decoder hidden dim).
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

        # Self-attention over hidden state sequence
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

    def forward(self, hidden: Tensor, mask: Tensor) -> Tensor:
        """Reconstruct embedding from hidden state representations.

        Args:
            hidden: ``(B, S, token_dim)`` hidden states (predicted or actual).
            mask: ``(B, S)`` boolean (True = valid token).

        Returns:
            ``(B, output_dim)`` reconstructed embedding.
        """
        pad_mask = ~mask

        h = self.encoder(hidden, src_key_padding_mask=pad_mask)

        q = self.query.expand(hidden.size(0), -1, -1)
        attended, _ = self.readout(q, h, h, key_padding_mask=pad_mask)
        out = self.readout_norm(attended.squeeze(1))

        return self.output_proj(out)

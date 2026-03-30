"""Shared neural components for agent communication games."""

from __future__ import annotations

import torch
from torch import Tensor, nn


class MessageEncoder(nn.Module):
    """Encode variable-length decoder hidden states into a fixed message vector.

    Uses self-attention to process the decoder's multi-scale hidden states,
    then a learned query cross-attention readout to produce a fixed-size
    vector.  This preserves the rich per-position structure from the
    multi-head decoder rather than destroying it with mean-pooling.

    Args:
        hidden_dim: Dimensionality of decoder hidden states.
        output_dim: Dimensionality of the output message vector.
        num_heads: Number of attention heads in self-attention and readout.
        num_layers: Number of self-attention layers.
    """

    def __init__(
        self,
        hidden_dim: int,
        output_dim: int,
        num_heads: int = 8,
        num_layers: int = 2,
    ) -> None:
        super().__init__()
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=num_heads,
                dim_feedforward=hidden_dim * 4,
                batch_first=True,
                dropout=0.1,
            )
            for _ in range(num_layers)
        ])
        self.query = nn.Parameter(torch.randn(1, 1, hidden_dim) * 0.02)
        self.readout = nn.MultiheadAttention(
            embed_dim=hidden_dim, num_heads=num_heads, batch_first=True,
        )
        self.out_norm = nn.LayerNorm(hidden_dim)
        self.proj = nn.Linear(hidden_dim, output_dim)

    def forward(self, hidden_states: Tensor, mask: Tensor) -> Tensor:
        """Encode decoder hidden states into a message vector.

        Args:
            hidden_states: ``(batch, seq_len, hidden_dim)`` decoder states.
            mask: ``(batch, seq_len)`` boolean mask (``True`` = valid).

        Returns:
            ``(batch, output_dim)`` message vector.
        """
        pad_mask = ~mask
        x = hidden_states
        for layer in self.layers:
            x = layer(x, src_key_padding_mask=pad_mask)

        b = x.size(0)
        query = self.query.expand(b, -1, -1)
        readout, _ = self.readout(query, x, x, key_padding_mask=pad_mask)
        readout = self.out_norm(readout.squeeze(1))
        return self.proj(readout)


class Receiver(nn.Module):
    """Score candidates against the message via learned dot-product.

    Args:
        dim: Dimensionality of message and candidate vectors.
    """

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.proj = nn.Linear(dim, dim)

    def forward(self, message: Tensor, candidates: Tensor) -> Tensor:
        """Score candidates.

        Args:
            message: ``(batch, dim)`` message vector.
            candidates: ``(batch, K, dim)`` candidate embeddings.

        Returns:
            ``(batch, K)`` logits.
        """
        projected = self.proj(message)
        return torch.bmm(
            projected.unsqueeze(1), candidates.transpose(1, 2),
        ).squeeze(1)

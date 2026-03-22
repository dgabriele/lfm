"""Structural adversarial discriminator for VAE pretraining.

Judges whether a subword token sequence has the distributional regularities
of natural language — subword transition patterns, syllable-scale smoothness,
punctuation placement, capitalization patterns — without judging semantic
content, vocabulary overlap, or global complexity.

An alien 200-morpheme agglutinative compound is fine as long as its internal
structure follows the distributional regularities learned from real language.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F  # noqa: N812
from torch import Tensor, nn


class StructuralDiscriminator(nn.Module):
    """Multi-scale CNN discriminator for structural language-likeness.

    Operates on subword token sequences via parallel convolutions at
    different kernel sizes (3, 5, 7) to capture structural features at
    trigram (phonotactic), 5-gram (morphological), and 7-gram (phrasal)
    scales.  Uses its own learned embeddings (not shared with the decoder)
    so it develops an independent representation of structural regularity.

    Args:
        vocab_size: Full vocabulary size including BOS/EOS special tokens.
        embed_dim: Discriminator embedding dimensionality.
        hidden_dim: Number of output channels per convolution.
        use_spectral_norm: Apply spectral normalization to convolutions
            for training stability.
    """

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int = 128,
        hidden_dim: int = 256,
        use_spectral_norm: bool = True,
    ) -> None:
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, embed_dim)

        # Multi-scale parallel convolutions
        conv3 = nn.Conv1d(embed_dim, hidden_dim, kernel_size=3, padding=1)
        conv5 = nn.Conv1d(embed_dim, hidden_dim, kernel_size=5, padding=2)
        conv7 = nn.Conv1d(embed_dim, hidden_dim, kernel_size=7, padding=3)

        if use_spectral_norm:
            conv3 = nn.utils.spectral_norm(conv3)
            conv5 = nn.utils.spectral_norm(conv5)
            conv7 = nn.utils.spectral_norm(conv7)

        self.conv3 = conv3
        self.conv5 = conv5
        self.conv7 = conv7

        # Classification head
        self.head = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
        self,
        token_ids: Tensor,
        mask: Tensor,
        soft_probs: Tensor | None = None,
    ) -> Tensor:
        """Score a batch of token sequences for structural regularity.

        Args:
            token_ids: ``(batch, seq_len)`` integer token indices.  Used
                for real data.
            mask: ``(batch, seq_len)`` boolean, ``True`` = valid position.
            soft_probs: Optional ``(batch, seq_len, vocab_size)``
                differentiable Gumbel-Softmax distributions.  When
                provided, used instead of ``token_ids`` for embedding
                lookup (``soft_probs @ embedding.weight``), enabling
                gradient flow to the generator.

        Returns:
            ``(batch,)`` scalar logits — higher = more language-like.
        """
        if soft_probs is not None:
            # Differentiable path: soft probs @ embedding weight
            x = soft_probs @ self.embedding.weight  # (B, S, E)
        else:
            x = self.embedding(token_ids)  # (B, S, E)
        x = x * mask.unsqueeze(-1).float()  # zero out padding
        x = x.transpose(1, 2)  # (B, E, S) for Conv1d

        h3 = F.leaky_relu(self.conv3(x), 0.2)  # (B, H, S)
        h5 = F.leaky_relu(self.conv5(x), 0.2)
        h7 = F.leaky_relu(self.conv7(x), 0.2)

        # Masked mean pool over sequence dimension
        mask_1d = mask.unsqueeze(1).float()  # (B, 1, S)
        denom = mask_1d.sum(dim=2).clamp(min=1)  # (B, 1)

        pool3 = (h3 * mask_1d).sum(dim=2) / denom  # (B, H)
        pool5 = (h5 * mask_1d).sum(dim=2) / denom
        pool7 = (h7 * mask_1d).sum(dim=2) / denom

        pooled = torch.cat([pool3, pool5, pool7], dim=1)  # (B, H*3)
        return self.head(pooled).squeeze(-1)  # (B,)

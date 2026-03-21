"""Abstract base class for phonology modules.

All phonology implementations must subclass ``PhonologyModule`` and implement
the required abstract interface.  The phonology module maps discrete tokens to
continuous surface-form vectors subject to implicit phonotactic constraints
that emerge from communication pressure.
"""

from __future__ import annotations

from abc import abstractmethod
from typing import ClassVar

from torch import Tensor

from lfm._types import TokenEmbeddings, TokenIds
from lfm.core.module import LFMModule


class PhonologyModule(LFMModule):
    """Base class for phonology modules.

    Phonology maps discrete tokens to continuous surface-form vectors
    subject to implicit phonotactic constraints.  No explicit phonological
    categories (vowels, consonants, sonority hierarchies) are encoded —
    structure emerges from smoothness, energy, and diversity pressures.

    Subclasses must implement:
        - ``forward``: full phonological processing returning surface forms,
          energy contour, pronounceability scores, and enriched embeddings.
        - ``to_surface_forms``: mapping from tokens/embeddings to continuous
          surface-form vectors.
    """

    output_prefix: ClassVar[str] = "phonology"

    @abstractmethod
    def forward(self, tokens: TokenIds, embeddings: TokenEmbeddings) -> dict[str, Tensor]:
        """Run phonological processing on a batch of token sequences.

        Args:
            tokens: Integer token indices of shape ``(batch, seq_len)``.
            embeddings: Dense token embeddings of shape
                ``(batch, seq_len, dim)``.

        Returns:
            Dictionary with the following keys:

            - ``surface_forms`` — continuous surface vectors, shape
              ``(batch, seq_len, max_surface_len, surface_dim)``.
            - ``energy_contour`` — scalar energy per surface position,
              shape ``(batch, seq_len, max_surface_len)``.
            - ``pronounceability_score`` — scalar score per token, shape
              ``(batch, seq_len)``.
            - ``embeddings`` — phonologically enriched embeddings, shape
              ``(batch, seq_len, dim)``.
        """
        ...

    @abstractmethod
    def to_surface_forms(self, tokens: TokenIds, embeddings: TokenEmbeddings) -> Tensor:
        """Map tokens to continuous surface-form vectors.

        Args:
            tokens: Integer token indices of shape ``(batch, seq_len)``.
            embeddings: Dense token embeddings of shape
                ``(batch, seq_len, dim)``.

        Returns:
            Surface-form tensor of shape
            ``(batch, seq_len, max_surface_len, surface_dim)``.
        """
        ...

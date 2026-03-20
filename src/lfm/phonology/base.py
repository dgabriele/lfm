"""Abstract base class for phonology modules.

All phonology implementations must subclass ``PhonologyModule`` and implement
the required abstract interface.  The phonology module imposes phonotactic
constraints on discrete token representations, biasing them toward
pronounceable forms with well-defined syllable structure.
"""

from __future__ import annotations

from abc import abstractmethod
from typing import ClassVar

from torch import Tensor

from lfm._types import TokenEmbeddings, TokenIds
from lfm.core.module import LFMModule


class PhonologyModule(LFMModule):
    """Abstract base for phonology modules.

    A phonology module maps discrete tokens and their embeddings to
    phoneme sequences, syllable structures, and pronounceability scores.
    It enriches token embeddings with phonological information for use
    by downstream morphological and syntactic modules.

    Subclasses must implement:
        - ``forward``: full phonological analysis returning phoneme sequences,
          syllable structure, pronounceability scores, and enriched embeddings.
        - ``to_phonemes``: mapping from discrete tokens to phoneme sequences.
    """

    output_prefix: ClassVar[str] = "phonology"

    @abstractmethod
    def forward(self, tokens: TokenIds, embeddings: TokenEmbeddings) -> dict[str, Tensor]:
        """Run phonological analysis on a batch of token sequences.

        Args:
            tokens: Integer token indices of shape ``(batch, seq_len)``.
            embeddings: Dense token embeddings of shape
                ``(batch, seq_len, dim)``.

        Returns:
            Dictionary with the following keys:

            - ``phoneme_sequences`` — phoneme index tensor, shape
              ``(batch, seq_len, max_phonemes)``.
            - ``syllable_structure`` — syllable boundary indicators, shape
              ``(batch, seq_len, max_syllables)``.
            - ``pronounceability_score`` — scalar score per token, shape
              ``(batch, seq_len)``.
            - ``embeddings`` — phonologically enriched embeddings, shape
              ``(batch, seq_len, dim)``.
        """
        ...

    @abstractmethod
    def to_phonemes(self, tokens: TokenIds) -> Tensor:
        """Map discrete tokens to phoneme sequences.

        Args:
            tokens: Integer token indices of shape ``(batch, seq_len)``.

        Returns:
            Phoneme index tensor of shape ``(batch, seq_len, max_phonemes)``.
        """
        ...

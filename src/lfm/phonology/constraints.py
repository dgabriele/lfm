"""Phonotactic constraint module.

Implements onset/nucleus/coda syllable structure rules with sonority
sequencing constraints.  Evaluates token sequences for well-formedness
according to configurable phonotactic rules.
"""

from __future__ import annotations

from torch import Tensor

from lfm._registry import register
from lfm._types import TokenIds
from lfm.phonology.base import PhonologyModule
from lfm.phonology.config import PhonologyConfig


@register("phonology", "constraints")
class PhonotacticConstraints(PhonologyModule):
    """Onset/nucleus/coda syllable structure rules with sonority sequencing.

    Evaluates phoneme sequences against phonotactic rules governing which
    consonant clusters are valid in onset and coda positions, and enforces
    the Sonority Sequencing Principle (SSP): onsets must rise in sonority
    toward the nucleus, and codas must fall.

    Args:
        config: Phonology configuration specifying inventory and syllable
            structure limits.
    """

    def __init__(self, config: PhonologyConfig) -> None:
        super().__init__(config)

        self.max_syllables = config.max_syllables_per_token

    # ------------------------------------------------------------------
    # Abstract method implementations
    # ------------------------------------------------------------------

    def to_phonemes(self, tokens: TokenIds) -> Tensor:
        """Map discrete tokens to phoneme sequences.

        Args:
            tokens: Integer token indices of shape ``(batch, seq_len)``.

        Returns:
            Phoneme index tensor of shape ``(batch, seq_len, max_phonemes)``.
        """
        raise NotImplementedError("PhonotacticConstraints.to_phonemes() not yet implemented")

    def forward(self, tokens: TokenIds, embeddings: Tensor) -> dict[str, Tensor]:
        """Evaluate phonotactic constraints on a batch of token sequences.

        Args:
            tokens: Integer token indices of shape ``(batch, seq_len)``.
            embeddings: Dense token embeddings of shape
                ``(batch, seq_len, dim)``.

        Returns:
            Dictionary with phoneme sequences, syllable structure,
            pronounceability scores, and enriched embeddings.
        """
        raise NotImplementedError("PhonotacticConstraints.forward() not yet implemented")

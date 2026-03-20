"""Pronounceability scorer module.

Provides a differentiable pronounceability scorer that evaluates how
pronounceable a token sequence is, biased toward English phonotactic
patterns by default.  This is the default phonology module.
"""

from __future__ import annotations

from torch import Tensor, nn

from lfm._registry import register
from lfm._types import TokenIds
from lfm.phonology.base import PhonologyModule
from lfm.phonology.config import PhonologyConfig


@register("phonology", "pronounceable")
class PronounceabilityScorer(PhonologyModule):
    """Differentiable pronounceability scorer, English-biased by default.

    Uses a small learned network to produce a scalar pronounceability
    score for each token position.  The scorer is trained jointly with
    the rest of the pipeline, encouraging tokens to adopt phonotactically
    valid surface forms.

    This is the default phonology module (registered name ``"pronounceable"``
    matches ``PhonologyConfig.name``).

    Args:
        config: Phonology configuration specifying inventory and weight
            parameters.
    """

    def __init__(self, config: PhonologyConfig) -> None:
        super().__init__(config)

        self.pronounceability_weight = config.pronounceability_weight
        self.max_syllables = config.max_syllables_per_token

        # Placeholder scoring head — will be wired to embedding dim at
        # build time once upstream dimensions are known.
        self.score_head: nn.Module | None = None

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
        raise NotImplementedError("PronounceabilityScorer.to_phonemes() not yet implemented")

    def forward(self, tokens: TokenIds, embeddings: Tensor) -> dict[str, Tensor]:
        """Score pronounceability for a batch of token sequences.

        Args:
            tokens: Integer token indices of shape ``(batch, seq_len)``.
            embeddings: Dense token embeddings of shape
                ``(batch, seq_len, dim)``.

        Returns:
            Dictionary with phoneme sequences, syllable structure,
            pronounceability scores, and enriched embeddings.
        """
        raise NotImplementedError("PronounceabilityScorer.forward() not yet implemented")

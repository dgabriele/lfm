"""Morpheme composer module.

Implements productive morpheme recombination, taking a segmented morpheme
representation and composing morphemes into novel tokens.  This enables
the system to productively generate new word forms from existing
morphological building blocks.
"""

from __future__ import annotations

from torch import Tensor

from lfm._registry import register
from lfm._types import Mask, TokenIds
from lfm.morphology.base import MorphologyModule
from lfm.morphology.config import MorphologyConfig


@register("morphology", "composer")
class MorphemeComposer(MorphologyModule):
    """Productive morpheme recombination -- composes morphemes into novel tokens.

    Takes morpheme-level representations and recombines them to produce
    token-level embeddings that reflect compositional morphological
    structure.  Supports productive generation of new word forms from
    learned morpheme units.

    Args:
        config: Morphology configuration specifying morpheme counts and
            dimension parameters.
    """

    def __init__(self, config: MorphologyConfig) -> None:
        super().__init__(config)

        self.max_morphemes = config.max_morphemes
        self.morpheme_dim = config.morpheme_dim

    # ------------------------------------------------------------------
    # Abstract method implementations
    # ------------------------------------------------------------------

    def segment(self, tokens: TokenIds) -> tuple[Tensor, Mask]:
        """Segment tokens into morpheme units without recomposition.

        Args:
            tokens: Integer token indices of shape ``(batch, seq_len)``.

        Returns:
            A tuple of morpheme segment indices and a boolean mask
            indicating valid morpheme positions.
        """
        raise NotImplementedError("MorphemeComposer.segment() not yet implemented")

    def forward(self, tokens: TokenIds, embeddings: Tensor) -> dict[str, Tensor]:
        """Compose morpheme representations into token-level embeddings.

        Args:
            tokens: Integer token indices of shape ``(batch, seq_len)``.
            embeddings: Dense token embeddings of shape
                ``(batch, seq_len, dim)``.

        Returns:
            Dictionary with segments, segment mask, composed embeddings,
            and segment log-probabilities.
        """
        raise NotImplementedError("MorphemeComposer.forward() not yet implemented")

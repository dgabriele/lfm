"""Minimum Description Length (MDL) morphological segmenter.

Implements unsupervised morphological segmentation using the Minimum
Description Length principle, inspired by the Morfessor family of models.
Discovers sub-token structure by finding segmentations that minimise the
combined cost of the lexicon and the corpus given the lexicon.
"""

from __future__ import annotations

from torch import Tensor

from lfm._registry import register
from lfm._types import Mask, TokenIds
from lfm.morphology.base import MorphologyModule
from lfm.morphology.config import MorphologyConfig


@register("morphology", "mdl_segmenter")
class MDLSegmenter(MorphologyModule):
    """Unsupervised morphological segmentation using MDL principle (Morfessor-like).

    Learns to segment tokens into morpheme-like units by minimising a
    two-part description length: the cost of storing the morpheme lexicon
    plus the cost of encoding the corpus using that lexicon.

    Args:
        config: Morphology configuration specifying morpheme counts and
            dimension parameters.
    """

    def __init__(self, config: MorphologyConfig) -> None:
        super().__init__(config)

        self.max_morphemes = config.max_morphemes
        self.morpheme_dim = config.morpheme_dim
        self.min_morpheme_len = config.min_morpheme_len
        self.max_morpheme_len = config.max_morpheme_len

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
        raise NotImplementedError("MDLSegmenter.segment() not yet implemented")

    def forward(self, tokens: TokenIds, embeddings: Tensor) -> dict[str, Tensor]:
        """Run MDL-based morphological segmentation and recomposition.

        Args:
            tokens: Integer token indices of shape ``(batch, seq_len)``.
            embeddings: Dense token embeddings of shape
                ``(batch, seq_len, dim)``.

        Returns:
            Dictionary with segments, segment mask, composed embeddings,
            segment log-probabilities, and grammatical_features (learned
            latent grammatical categories per token).
        """
        raise NotImplementedError("MDLSegmenter.forward() not yet implemented")

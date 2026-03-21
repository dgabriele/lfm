"""Hierarchical tokenizer module.

Implements a tree-structured tokenizer that learns hierarchical
character/subword structure, inspired by the TreeTokenizer approach.
Discovers a multi-level decomposition of tokens into sub-units
arranged in a binary tree.
"""

from __future__ import annotations

from torch import Tensor

from lfm._registry import register
from lfm._types import Mask, TokenIds
from lfm.morphology.base import MorphologyModule
from lfm.morphology.config import MorphologyConfig


@register("morphology", "hierarchical")
class HierarchicalTokenizer(MorphologyModule):
    """Learns hierarchical character/subword structure (TreeTokenizer-like).

    Builds a binary tree over the sub-units of each token, learning
    which merges produce meaningful morphological constituents.  The
    tree structure enables hierarchical composition from characters or
    character n-grams up to full token representations.

    Args:
        config: Morphology configuration specifying morpheme counts and
            dimension parameters.
    """

    def __init__(self, config: MorphologyConfig) -> None:
        super().__init__(config)

        self.max_morphemes = config.max_morphemes
        self.morpheme_dim = config.morpheme_dim
        self.max_morpheme_len = config.max_morpheme_len

    # ------------------------------------------------------------------
    # Abstract method implementations
    # ------------------------------------------------------------------

    def segment(self, tokens: TokenIds) -> tuple[Tensor, Mask]:
        """Segment tokens into hierarchical sub-units.

        Args:
            tokens: Integer token indices of shape ``(batch, seq_len)``.

        Returns:
            A tuple of morpheme segment indices and a boolean mask
            indicating valid morpheme positions.
        """
        raise NotImplementedError("HierarchicalTokenizer.segment() not yet implemented")

    def forward(self, tokens: TokenIds, embeddings: Tensor) -> dict[str, Tensor]:
        """Run hierarchical tokenization with tree-structured composition.

        Args:
            tokens: Integer token indices of shape ``(batch, seq_len)``.
            embeddings: Dense token embeddings of shape
                ``(batch, seq_len, dim)``.

        Returns:
            Dictionary with segments, segment mask, composed embeddings,
            segment log-probabilities, and grammatical_features (learned
            latent grammatical categories per token).
        """
        raise NotImplementedError("HierarchicalTokenizer.forward() not yet implemented")

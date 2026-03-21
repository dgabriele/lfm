"""Morphological attention module.

Implements attention biased by morphological feature similarity. Tokens with
compatible grammatical features attend more strongly to each other, creating
implicit phrase structure: words that share case marking, number, or other
features form natural groups without explicit constituency parsing.
"""

from __future__ import annotations

from torch import Tensor

from lfm._registry import register
from lfm._types import GrammaticalFeatures, Mask, TokenEmbeddings
from lfm.syntax.base import SyntaxModule
from lfm.syntax.config import SyntaxConfig


@register("syntax", "morphological_attention")
class MorphologicalAttention(SyntaxModule):
    """Attention biased by morphological feature similarity.

    Tokens with compatible grammatical features attend more strongly to
    each other. This creates implicit phrase structure: words that share
    case marking, number, or other features form natural groups without
    explicit constituency parsing.

    Args:
        config: Syntax configuration specifying agreement heads and latent
            dimensionality.
    """

    def __init__(self, config: SyntaxConfig) -> None:
        super().__init__(config)
        # Placeholder layers for future implementation

    def forward(
        self,
        embeddings: TokenEmbeddings,
        mask: Mask,
        grammatical_features: GrammaticalFeatures | None = None,
    ) -> dict[str, Tensor]:
        """Compute attention scores biased by morphological feature similarity.

        Args:
            embeddings: Token embeddings, shape (batch, seq_len, dim).
            mask: Boolean padding mask, shape (batch, seq_len).
            grammatical_features: Optional morphological feature vectors,
                shape (batch, seq_len, num_features).

        Returns:
            Dictionary with agreement_scores, ordering_scores, and
            structural_features.
        """
        raise NotImplementedError("MorphologicalAttention.forward() not yet implemented")

"""Information-theoretic ordering pressure module.

Learns to score token positions by information content, encouraging consistent
ordering strategies (e.g., high-information tokens first, topic-comment
structure, given-new ordering). This replaces explicit word-order rules with
a learned soft preference that adapts to the agent's communication pressures.
"""

from __future__ import annotations

from torch import Tensor

from lfm._registry import register
from lfm._types import GrammaticalFeatures, Mask, TokenEmbeddings
from lfm.syntax.base import SyntaxModule
from lfm.syntax.config import SyntaxConfig


@register("syntax", "ordering_pressure")
class OrderingPressure(SyntaxModule):
    """Information-theoretic ordering pressure.

    Learns to score token positions by information content, encouraging
    consistent ordering strategies (e.g., high-information tokens first,
    topic-comment structure, given-new ordering).

    This replaces explicit word-order rules with a learned soft preference
    that adapts to the agent's communication pressures.

    Args:
        config: Syntax configuration specifying ordering temperature and
            latent dimensionality.
    """

    def __init__(self, config: SyntaxConfig) -> None:
        super().__init__(config)
        self._temperature = config.ordering_temperature
        # Placeholder layers for future implementation

    def forward(
        self,
        embeddings: TokenEmbeddings,
        mask: Mask,
        grammatical_features: GrammaticalFeatures | None = None,
    ) -> dict[str, Tensor]:
        """Compute ordering pressure scores for token positions.

        Args:
            embeddings: Token embeddings, shape (batch, seq_len, dim).
            mask: Boolean padding mask, shape (batch, seq_len).
            grammatical_features: Optional morphological feature vectors,
                shape (batch, seq_len, num_features).

        Returns:
            Dictionary with agreement_scores, ordering_scores, and
            structural_features.
        """
        raise NotImplementedError("OrderingPressure.forward() not yet implemented")

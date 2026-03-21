"""Agreement-based structural module.

Implements soft agreement constraints between morphological features using
multi-head attention. Each head specializes in a different type of agreement
(e.g., case agreement, number agreement). The agreement is learned, not
predefined -- positions with compatible grammatical features receive high
agreement scores while incompatible features receive low scores.
"""

from __future__ import annotations

from torch import Tensor, nn

from lfm._registry import register
from lfm._types import GrammaticalFeatures, Mask, TokenEmbeddings
from lfm.syntax.base import SyntaxModule
from lfm.syntax.config import SyntaxConfig


@register("syntax", "agreement")
class AgreementModule(SyntaxModule):
    """Learns soft agreement constraints between morphological features.

    Uses multi-head attention where each head specializes in a different
    type of agreement (e.g., one head for case agreement, another for
    number agreement). The agreement is learned, not predefined.

    Grammatical features from the morphology module are projected and
    compared pairwise. Positions with compatible features receive high
    agreement scores; incompatible features receive low scores.

    Args:
        config: Syntax configuration specifying agreement heads and latent
            dimensionality.
    """

    def __init__(self, config: SyntaxConfig) -> None:
        super().__init__(config)
        self._num_heads = config.num_agreement_heads
        self._latent_dim = config.latent_dim
        # Placeholder layers for future implementation
        self.feature_proj = nn.Linear(config.latent_dim, config.latent_dim)
        self.agreement_heads = nn.MultiheadAttention(
            config.latent_dim,
            config.num_agreement_heads,
            batch_first=True,
        )

    def forward(
        self,
        embeddings: TokenEmbeddings,
        mask: Mask,
        grammatical_features: GrammaticalFeatures | None = None,
    ) -> dict[str, Tensor]:
        """Compute agreement scores from morphological features.

        Args:
            embeddings: Token embeddings, shape (batch, seq_len, dim).
            mask: Boolean padding mask, shape (batch, seq_len).
            grammatical_features: Optional morphological feature vectors,
                shape (batch, seq_len, num_features).

        Returns:
            Dictionary with agreement_scores, ordering_scores, and
            structural_features.
        """
        raise NotImplementedError("AgreementModule.forward() not yet implemented")

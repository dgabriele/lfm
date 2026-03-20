"""Structural attention masking module.

Implements Transformer Grammars-style structural attention masking,
where the attention pattern is constrained to follow the syntactic
structure induced over the input sequence.
"""

from __future__ import annotations

from torch import Tensor, nn

from lfm._registry import register
from lfm._types import Mask, TokenEmbeddings
from lfm.syntax.base import SyntaxModule
from lfm.syntax.config import SyntaxConfig


@register("syntax", "structural_attention")
class StructuralAttentionMask(SyntaxModule):
    """Transformer Grammars-style structural attention masking.

    Produces attention masks that enforce syntactic locality: tokens can
    only attend to other tokens within the same syntactic constituent
    or to constituents that are structurally adjacent.  This biases the
    model toward learning hierarchically structured representations.

    Args:
        config: Syntax configuration specifying grammar size and latent
            dimensionality.
    """

    def __init__(self, config: SyntaxConfig) -> None:
        super().__init__(config)

        self.latent_dim = config.latent_dim

        # Placeholder layer for computing structural scores
        self.structure_proj = nn.Linear(config.latent_dim, config.latent_dim)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, embeddings: TokenEmbeddings, mask: Mask) -> dict[str, Tensor]:
        """Compute structural attention masks from token embeddings.

        Args:
            embeddings: Dense token embeddings of shape
                ``(batch, seq_len, dim)``.
            mask: Boolean padding mask of shape ``(batch, seq_len)``.

        Returns:
            Dictionary with tree log-probabilities, attention mask,
            constituent representations, and parse depth.
        """
        raise NotImplementedError("StructuralAttentionMask.forward() not yet implemented")

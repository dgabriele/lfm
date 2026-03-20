"""Ordered Neurons module.

Implements ON-LSTM-style ordered neuron integration for implicit syntactic
hierarchy.  Neuron activations are ordered so that higher-level (more
slowly changing) syntactic constituents are represented by higher-indexed
neurons, providing an implicit tree structure without explicit parsing.
"""

from __future__ import annotations

from torch import Tensor, nn

from lfm._registry import register
from lfm._types import Mask, TokenEmbeddings
from lfm.syntax.base import SyntaxModule
from lfm.syntax.config import SyntaxConfig


@register("syntax", "ordered_neurons")
class OrderedNeurons(SyntaxModule):
    """ON-LSTM-style ordered neuron integration for implicit hierarchy.

    Uses ordered neuron gates (master forget and master input gates with
    cumulative softmax) to enforce a hierarchy among hidden dimensions.
    Lower-indexed neurons encode short-range (leaf-level) information
    while higher-indexed neurons encode long-range (phrase/sentence-level)
    structure.

    Args:
        config: Syntax configuration specifying grammar size and latent
            dimensionality.
    """

    def __init__(self, config: SyntaxConfig) -> None:
        super().__init__(config)

        self.latent_dim = config.latent_dim

        # Placeholder layers for ordered neuron gates
        self.master_forget = nn.Linear(config.latent_dim, config.latent_dim)
        self.master_input = nn.Linear(config.latent_dim, config.latent_dim)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, embeddings: TokenEmbeddings, mask: Mask) -> dict[str, Tensor]:
        """Compute implicit hierarchical structure via ordered neurons.

        Args:
            embeddings: Dense token embeddings of shape
                ``(batch, seq_len, dim)``.
            mask: Boolean padding mask of shape ``(batch, seq_len)``.

        Returns:
            Dictionary with tree log-probabilities, attention mask,
            constituent representations, and parse depth.
        """
        raise NotImplementedError("OrderedNeurons.forward() not yet implemented")

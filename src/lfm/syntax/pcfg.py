"""Neural Probabilistic Context-Free Grammar (PCFG) module.

Implements compound PCFG-style neural grammar induction, learning a
probabilistic context-free grammar over token sequences in a fully
differentiable manner.  Nonterminal and preterminal rule probabilities
are parameterized by neural networks.
"""

from __future__ import annotations

from torch import Tensor, nn

from lfm._registry import register
from lfm._types import Mask, TokenEmbeddings
from lfm.syntax.base import SyntaxModule
from lfm.syntax.config import SyntaxConfig


@register("syntax", "neural_pcfg")
class NeuralPCFG(SyntaxModule):
    """Compound PCFG-style neural grammar induction.

    Learns a probabilistic context-free grammar by parameterizing rule
    probabilities with neural networks.  The inside algorithm is used
    to compute tree log-probabilities in a differentiable manner,
    enabling end-to-end training of the grammar alongside the rest of
    the pipeline.

    Args:
        config: Syntax configuration specifying grammar size and latent
            dimensionality.
    """

    def __init__(self, config: SyntaxConfig) -> None:
        super().__init__(config)

        self.num_nonterminals = config.num_nonterminals
        self.num_preterminals = config.num_preterminals
        self.latent_dim = config.latent_dim

        # Placeholder layers for rule-probability networks
        self.rule_mlp = nn.Linear(config.latent_dim, config.num_nonterminals)
        self.term_mlp = nn.Linear(config.latent_dim, config.num_preterminals)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, embeddings: TokenEmbeddings, mask: Mask) -> dict[str, Tensor]:
        """Induce a PCFG parse and compute tree log-probabilities.

        Args:
            embeddings: Dense token embeddings of shape
                ``(batch, seq_len, dim)``.
            mask: Boolean padding mask of shape ``(batch, seq_len)``.

        Returns:
            Dictionary with tree log-probabilities, attention mask,
            constituent representations, and parse depth.
        """
        raise NotImplementedError("NeuralPCFG.forward() not yet implemented")

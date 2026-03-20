"""Structural consistency losses.

Loss functions that reward well-formed syntactic structure and penalize
inconsistencies between induced tree structures and attention patterns.
"""

from __future__ import annotations

from torch import Tensor

from lfm._registry import register
from lfm.core.loss import LFMLoss


@register("loss", "tree_consistency")
class TreeConsistencyLoss(LFMLoss):
    """Penalizes inconsistency between induced tree structure and attention.

    Measures the divergence between the attention patterns implied by the
    induced parse tree and the actual attention weights observed in the
    model.  Encourages the model to attend in ways that are consistent
    with the syntactic structure it induces.

    Args:
        config: Optional loss configuration.
        weight: Multiplicative weight for this loss term.
    """

    def __init__(self, config: object = None, weight: float = 1.0) -> None:
        super().__init__(config, weight)

    def forward(
        self,
        outputs: dict[str, Tensor],
        targets: dict[str, Tensor] | None = None,
    ) -> Tensor:
        """Compute tree-attention consistency loss.

        Args:
            outputs: Combined pipeline output dictionary.
            targets: Optional ground-truth tensors (unused).

        Returns:
            Scalar loss tensor.
        """
        raise NotImplementedError("TreeConsistencyLoss.forward() not yet implemented")


@register("loss", "well_formedness")
class WellFormednessLoss(LFMLoss):
    """Rewards sequences that parse cleanly under induced grammar.

    Encourages the model to produce token sequences that have high
    log-probability under the induced grammar, rewarding well-formed
    syntactic structure.

    Args:
        config: Optional loss configuration.
        weight: Multiplicative weight for this loss term.
    """

    def __init__(self, config: object = None, weight: float = 1.0) -> None:
        super().__init__(config, weight)

    def forward(
        self,
        outputs: dict[str, Tensor],
        targets: dict[str, Tensor] | None = None,
    ) -> Tensor:
        """Compute well-formedness loss (negative tree log-probability).

        Args:
            outputs: Combined pipeline output dictionary.
            targets: Optional ground-truth tensors (unused).

        Returns:
            Scalar loss tensor.
        """
        raise NotImplementedError("WellFormednessLoss.forward() not yet implemented")

"""Compositionality losses.

Loss functions that encourage compositional reuse and productive
combination of morphological units.
"""

from __future__ import annotations

from torch import Tensor

from lfm._registry import register
from lfm.core.loss import LFMLoss


@register("loss", "morpheme_reuse")
class MorphemeReuseLoss(LFMLoss):
    """Pressures reuse of morphemes across different tokens.

    Encourages the system to discover a compact set of morphemes that
    are productively reused across many different token forms, rather
    than memorising a unique decomposition for each token.

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
        """Compute morpheme reuse pressure loss.

        Args:
            outputs: Combined pipeline output dictionary.
            targets: Optional ground-truth tensors (unused).

        Returns:
            Scalar loss tensor.
        """
        raise NotImplementedError("MorphemeReuseLoss.forward() not yet implemented")


@register("loss", "productive_combination")
class ProductiveCombinationLoss(LFMLoss):
    """Encourages novel productive combinations of existing morphemes.

    Rewards the system for generating new token forms by productively
    combining known morphemes in novel ways, rather than relying on
    a fixed set of token-morpheme mappings.

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
        """Compute productive combination encouragement loss.

        Args:
            outputs: Combined pipeline output dictionary.
            targets: Optional ground-truth tensors (unused).

        Returns:
            Scalar loss tensor.
        """
        raise NotImplementedError("ProductiveCombinationLoss.forward() not yet implemented")

"""Diversity losses.

Loss functions that encourage diverse structural realizations and prevent
mode collapse in the generated representations.
"""

from __future__ import annotations

from torch import Tensor

from lfm._registry import register
from lfm.core.loss import LFMLoss


@register("loss", "paraphrastic_diversity")
class ParaphrasticDiversityLoss(LFMLoss):
    """Encourages multiple valid structural realizations.

    Rewards the system for being capable of expressing the same
    underlying meaning through diverse structural forms, encouraging
    paraphrastic capacity in the emergent language.

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
        """Compute paraphrastic diversity loss.

        Args:
            outputs: Combined pipeline output dictionary.
            targets: Optional ground-truth tensors (unused).

        Returns:
            Scalar loss tensor.
        """
        raise NotImplementedError("ParaphrasticDiversityLoss.forward() not yet implemented")


@register("loss", "anti_collapse")
class AntiCollapseLoss(LFMLoss):
    """Prevents mode collapse in generated structures.

    Penalizes low variance in the structural outputs, preventing the
    model from converging to a single degenerate structure for all inputs.

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
        """Compute anti-collapse loss.

        Args:
            outputs: Combined pipeline output dictionary.
            targets: Optional ground-truth tensors (unused).

        Returns:
            Scalar loss tensor.
        """
        raise NotImplementedError("AntiCollapseLoss.forward() not yet implemented")

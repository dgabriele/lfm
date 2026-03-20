"""Information-theoretic losses.

Loss functions based on information theory that regularize codebook
usage and entropy of the discrete representations.
"""

from __future__ import annotations

from torch import Tensor

from lfm._registry import register
from lfm.core.loss import LFMLoss


@register("loss", "codebook_utilization")
class CodebookUtilizationLoss(LFMLoss):
    """Penalizes low codebook utilization (index collapse).

    Encourages uniform usage of codebook entries by penalizing
    distributions over codebook indices that have low entropy.  This
    combats the common failure mode of VQ models where only a small
    fraction of codebook entries are actively used.

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
        """Compute codebook utilization loss.

        Args:
            outputs: Combined pipeline output dictionary.
            targets: Optional ground-truth tensors (unused).

        Returns:
            Scalar loss tensor.
        """
        raise NotImplementedError("CodebookUtilizationLoss.forward() not yet implemented")


@register("loss", "entropy_regularization")
class EntropyRegularizationLoss(LFMLoss):
    """Information-theoretic entropy regularization (VQ-VIB style).

    Applies entropy-based regularization to the quantized representations,
    encouraging an information bottleneck that preserves only the most
    relevant information for downstream tasks.

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
        """Compute entropy regularization loss.

        Args:
            outputs: Combined pipeline output dictionary.
            targets: Optional ground-truth tensors (unused).

        Returns:
            Scalar loss tensor.
        """
        raise NotImplementedError("EntropyRegularizationLoss.forward() not yet implemented")

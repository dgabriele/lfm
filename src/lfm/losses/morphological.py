"""Morphological losses.

Loss functions that reward coherent morphological segmentation and
penalize phonotactically invalid (unpronounceable) word forms.
"""

from __future__ import annotations

from torch import Tensor

from lfm._registry import register
from lfm.core.loss import LFMLoss


@register("loss", "segmentation_coherence")
class SegmentationCoherenceLoss(LFMLoss):
    """Rewards consistent, coherent morphological segmentations.

    Encourages the segmenter to produce stable, repeatable segmentations
    for the same input and to segment morphologically related forms in
    consistent ways (e.g. shared stems receive the same segmentation).

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
        """Compute segmentation coherence loss.

        Args:
            outputs: Combined pipeline output dictionary.
            targets: Optional ground-truth tensors (unused).

        Returns:
            Scalar loss tensor.
        """
        raise NotImplementedError("SegmentationCoherenceLoss.forward() not yet implemented")


@register("loss", "pronounceability")
class PronounceabilityLoss(LFMLoss):
    """Penalizes unpronounceable surface forms.

    Uses the pronounceability scores from the surface phonology module
    to penalize tokens whose surface-form sequences are not smooth —
    encouraging the system to produce surface forms that are well-formed
    within its own emergent phonotactic constraints.

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
        """Compute pronounceability penalty loss.

        Args:
            outputs: Combined pipeline output dictionary.
            targets: Optional ground-truth tensors (unused).

        Returns:
            Scalar loss tensor.
        """
        raise NotImplementedError("PronounceabilityLoss.forward() not yet implemented")

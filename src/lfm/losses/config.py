"""Configuration for loss functions.

Defines ``LossConfig`` for individual loss terms and ``CompositeLossConfig``
for assembling a weighted combination of multiple losses used during training.
"""

from __future__ import annotations

from lfm.config.base import LFMBaseConfig


class LossConfig(LFMBaseConfig):
    """Configuration for a single loss term.

    Attributes:
        name: Registry name of the loss implementation.
        weight: Scalar multiplier applied to this loss term when combining
            it into the composite objective.
    """

    name: str
    weight: float = 1.0


class CompositeLossConfig(LFMBaseConfig):
    """Configuration for the full composite loss function.

    Assembles multiple weighted loss terms into a single training objective.

    Attributes:
        losses: List of individual loss term configurations.
    """

    losses: list[LossConfig] = []

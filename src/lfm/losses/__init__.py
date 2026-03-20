"""Loss subsystem for the LFM framework.

Provides ``LossConfig`` and ``CompositeLossConfig`` for defining and composing
weighted loss terms used during training.
"""

from __future__ import annotations

from lfm.losses.config import CompositeLossConfig, LossConfig

__all__ = ["CompositeLossConfig", "LossConfig"]

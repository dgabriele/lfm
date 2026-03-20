"""Morphology subsystem for the LFM framework.

Provides the abstract ``MorphologyModule`` base class and ``MorphologyConfig``
for imposing sub-token morphological structure on discrete representations.
"""

from __future__ import annotations

from lfm.morphology.base import MorphologyModule
from lfm.morphology.config import MorphologyConfig

__all__ = ["MorphologyConfig", "MorphologyModule"]

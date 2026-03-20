"""Phonology subsystem for the LFM framework.

Provides the abstract ``PhonologyModule`` base class and ``PhonologyConfig``
for imposing phonotactic constraints on token representations.
"""

from __future__ import annotations

from lfm.phonology.base import PhonologyModule
from lfm.phonology.config import PhonologyConfig

__all__ = ["PhonologyConfig", "PhonologyModule"]

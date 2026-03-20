"""Quantization subsystem for the LFM framework.

Provides the abstract ``Quantizer`` base class and ``QuantizationConfig``
for converting continuous agent states into discrete token sequences.
"""

from __future__ import annotations

from lfm.quantization.base import Quantizer
from lfm.quantization.config import QuantizationConfig

__all__ = ["Quantizer", "QuantizationConfig"]

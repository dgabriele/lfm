"""Sentence subsystem for the LFM framework.

Provides the abstract ``SentenceModule`` base class and ``SentenceConfig``
for sentence-type classification and boundary detection.
"""

from __future__ import annotations

from lfm.sentence.base import SentenceModule
from lfm.sentence.config import SentenceConfig

__all__ = ["SentenceConfig", "SentenceModule"]

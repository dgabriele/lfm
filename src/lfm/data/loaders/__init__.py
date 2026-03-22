"""Corpus loader subsystem for the LFM framework.

Provides modular, registry-based corpus loaders for acquiring and balancing
multilingual text data from various sources (Leipzig, OPUS, etc.).
"""

from __future__ import annotations

from lfm.data.loaders.base import CorpusLoader, CorpusLoaderConfig
from lfm.data.loaders.leipzig import LeipzigCorpusConfig, LeipzigCorpusLoader

__all__ = [
    "CorpusLoader",
    "CorpusLoaderConfig",
    "LeipzigCorpusConfig",
    "LeipzigCorpusLoader",
]

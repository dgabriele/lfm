"""Data subsystem for the LFM framework.

Provides dataset classes, corpus loaders, and collation utilities for
data loading, batching, and preprocessing pipelines.
"""

from __future__ import annotations

from lfm.data.collation import variable_length_collate
from lfm.data.config import DataConfig
from lfm.data.corpus import CorpusDataset, MultilingualCorpusDataset

__all__ = [
    "CorpusDataset",
    "DataConfig",
    "MultilingualCorpusDataset",
    "variable_length_collate",
]

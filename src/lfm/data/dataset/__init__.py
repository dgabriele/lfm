"""Dataset generation and management for the LFM framework.

Provides the pipeline for generating pre-processed HDF5 datasets from
corpus sources, and a reader for loading them during pretraining.

Imports of ``DatasetGenerator`` and ``DatasetReader`` are lazy to avoid
requiring ``h5py`` at import time — only needed when actually used.
"""

from __future__ import annotations

from lfm.data.dataset.config import DatasetGenerateConfig, LLMGateConfig, ProcessedSample
from lfm.data.dataset.manifest import DatasetManifest

__all__ = [
    "DatasetGenerateConfig",
    "DatasetGenerator",
    "DatasetManifest",
    "DatasetReader",
    "LLMGateConfig",
    "ProcessedSample",
]


def __getattr__(name: str):
    if name == "DatasetGenerator":
        from lfm.data.dataset.generator import DatasetGenerator

        return DatasetGenerator
    if name == "DatasetReader":
        from lfm.data.dataset.reader import DatasetReader

        return DatasetReader
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

"""Configuration for data loading.

Defines the ``DataConfig`` used to parameterize data loading, batching,
and preprocessing for LFM training and evaluation.
"""

from __future__ import annotations

from lfm.config.base import LFMBaseConfig


class DataConfig(LFMBaseConfig):
    """Configuration for data loading and preprocessing.

    Attributes:
        corpus_paths: List of file or directory paths to load training
            data from.
        batch_size: Number of samples per training batch.
        max_seq_len: Maximum sequence length; longer sequences are
            truncated.
        num_workers: Number of parallel data-loading worker processes.
        pin_memory: Whether to pin data-loader memory for faster
            GPU transfers.
        languages: List of language codes to include.  An empty list
            means all available languages are used.
    """

    corpus_paths: list[str] = []
    batch_size: int = 64
    max_seq_len: int = 128
    num_workers: int = 4
    pin_memory: bool = True
    languages: list[str] = []

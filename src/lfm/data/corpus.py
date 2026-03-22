"""Corpus dataset implementations.

Provides dataset classes for loading text corpora used in structural prior
learning.  Supports both monolingual and multilingual corpus configurations.
"""

from __future__ import annotations

import torch
from torch import Tensor
from torch.utils.data import Dataset

from lfm.data.config import DataConfig


class CorpusDataset(Dataset[dict[str, Tensor]]):
    """Dataset wrapping a text corpus for structural prior learning.

    Loads and tokenizes text data from one or more corpus files, producing
    fixed-length sequences suitable for training the LFM pipeline.

    Args:
        config: Data configuration specifying corpus paths, sequence length,
            and preprocessing parameters.
    """

    def __init__(self, config: DataConfig) -> None:
        self.config = config
        self.corpus_paths = config.corpus_paths
        self.max_seq_len = config.max_seq_len

        # Placeholder for loaded data — populated by a future load() call
        self._data: list[dict[str, Tensor]] = []

    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        return len(self._data)

    def __getitem__(self, index: int) -> dict[str, Tensor]:
        """Return a single sample by index.

        Args:
            index: Sample index.

        Returns:
            Dictionary with tokenized sequence tensors.
        """
        return self._data[index]


class MultilingualCorpusDataset(Dataset[tuple[Tensor, int]]):
    """Pre-tokenized and padded multilingual text sequences.

    Wraps tokenized integer sequences (from any corpus loader + sentencepiece)
    into a PyTorch Dataset.  Each sample is padded or truncated to
    ``max_seq_len`` with an EOS token appended.

    Args:
        token_ids: List of integer token id lists, one per sentence.
        max_seq_len: Maximum sequence length (including EOS).
        eos_id: EOS token index to append to each sequence.
    """

    def __init__(
        self,
        token_ids: list[list[int]],
        max_seq_len: int,
        eos_id: int,
    ) -> None:
        self.data: list[tuple[Tensor, int]] = []
        for ids in token_ids:
            # Append EOS and truncate
            ids_with_eos = ids[: max_seq_len - 1] + [eos_id]
            length = len(ids_with_eos)
            padded = ids_with_eos + [0] * (max_seq_len - length)
            self.data.append((torch.tensor(padded, dtype=torch.long), length))

    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        return len(self.data)

    def __getitem__(self, idx: int) -> tuple[Tensor, int]:
        """Return a single sample: ``(padded_token_ids, length)``.

        Args:
            idx: Sample index.

        Returns:
            Tuple of ``(token_ids, length)`` where ``token_ids`` is a
            padded long tensor of shape ``(max_seq_len,)`` and ``length``
            is the actual sequence length before padding.
        """
        return self.data[idx]

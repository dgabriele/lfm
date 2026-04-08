"""Interleaved multilingual dataset for cross-lingual transfer.

Randomly samples from a primary corpus (Xenoglot) and a secondary
corpus (English) based on a configurable ratio.  The model sees both
languages in the same training distribution — the same mechanism that
enables cross-lingual transfer in multilingual LLMs.

No paired translations.  Just interleaved monolingual data.
"""

from __future__ import annotations

import logging

import numpy as np
import torch
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)


class InterleavedDataset(Dataset):
    """Interleave two tokenized datasets with a configurable ratio.

    Each ``__getitem__`` call randomly selects from either the primary
    or secondary dataset based on the mixing ratio.  This produces a
    training distribution where the model sees both languages without
    explicit pairing.

    Args:
        primary: Main dataset (e.g., Xenoglot).
        secondary: Interleaved dataset (e.g., English).
        secondary_ratio: Fraction of samples drawn from secondary
            (0.0 = primary only, 1.0 = secondary only).
        seed: Random seed for reproducible mixing.
    """

    def __init__(
        self,
        primary: Dataset,
        secondary: Dataset,
        secondary_ratio: float = 0.3,
        seed: int = 42,
    ) -> None:
        self.primary = primary
        self.secondary = secondary
        self.secondary_ratio = secondary_ratio
        self._rng = np.random.default_rng(seed)

        # Total length = primary length (secondary is sampled with replacement)
        self._len = len(primary)

        # Pre-compute which indices are secondary for deterministic ordering
        self._is_secondary = self._rng.random(self._len) < secondary_ratio
        self._secondary_indices = self._rng.integers(
            0, len(secondary), size=int(self._is_secondary.sum()),
        )

        n_sec = int(self._is_secondary.sum())
        logger.info(
            "InterleavedDataset: %d primary, %d secondary (%.0f%%), "
            "%d total",
            self._len - n_sec, n_sec,
            secondary_ratio * 100, self._len,
        )

    def __len__(self) -> int:
        return self._len

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        if self._is_secondary[idx]:
            sec_count = int(self._is_secondary[:idx].sum())
            sec_idx = self._secondary_indices[sec_count % len(self._secondary_indices)]
            return self.secondary[int(sec_idx)]
        return self.primary[idx]

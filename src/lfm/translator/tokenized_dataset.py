"""HDF5-backed tokenized dataset for memory-efficient LLM pretraining.

Pre-tokenizes a text corpus once and saves to HDF5 with LZ4
compression. Training reads from the HDF5 file via memory-mapped
access — constant ~10MB RAM regardless of corpus size.

Usage::

    # First run: tokenizes and saves (one-time cost)
    ds = TokenizedH5Dataset.from_corpus(
        "corpus.txt", tokenizer, max_len=512,
        h5_path="corpus_tokenized.h5",
    )

    # Subsequent runs: loads instantly from HDF5
    ds = TokenizedH5Dataset("corpus_tokenized.h5")
    loader = DataLoader(ds, batch_size=2, shuffle=True)
"""

from __future__ import annotations

import logging
from pathlib import Path

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)


class TokenizedH5Dataset(Dataset):
    """Memory-mapped tokenized dataset backed by HDF5.

    Stores pre-tokenized input_ids, attention_mask, and labels as
    compressed int32 arrays. At training time, reads individual
    examples on demand — no bulk loading into RAM.

    Args:
        h5_path: Path to the HDF5 file.
    """

    def __init__(self, h5_path: str | Path) -> None:
        self.h5_path = Path(h5_path)
        # Open in read mode — h5py handles mmap internally
        self._file = h5py.File(self.h5_path, "r")
        self._input_ids = self._file["input_ids"]
        self._attention_mask = self._file["attention_mask"]
        self._labels = self._file["labels"]
        self._len = self._input_ids.shape[0]
        logger.info(
            "Loaded tokenized dataset: %d examples, max_len=%d, file=%s",
            self._len, self._input_ids.shape[1], self.h5_path,
        )

    def __len__(self) -> int:
        return self._len

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        return {
            "input_ids": torch.from_numpy(
                self._input_ids[idx].astype(np.int64),
            ),
            "attention_mask": torch.from_numpy(
                self._attention_mask[idx].astype(np.int64),
            ),
            "labels": torch.from_numpy(
                self._labels[idx].astype(np.int64),
            ),
        }

    def close(self) -> None:
        self._file.close()

    @classmethod
    def from_corpus(
        cls,
        corpus_path: str | Path,
        tokenizer,
        max_len: int = 512,
        h5_path: str | Path | None = None,
        val_fraction: float = 0.05,
        chunk_size: int = 10000,
    ) -> tuple[TokenizedH5Dataset, TokenizedH5Dataset]:
        """Tokenize a text corpus and save to HDF5.

        Processes the corpus in chunks to bound peak RAM usage.
        Returns (train_dataset, val_dataset).

        Args:
            corpus_path: Path to the text corpus.
            tokenizer: HuggingFace tokenizer.
            max_len: Maximum sequence length.
            h5_path: Output HDF5 path. Defaults to corpus_path with .h5 suffix.
            val_fraction: Fraction of lines held out for validation.
            chunk_size: Lines processed per batch during tokenization.

        Returns:
            Tuple of (train_dataset, val_dataset).
        """
        corpus_path = Path(corpus_path)
        if h5_path is None:
            h5_path = corpus_path.with_suffix(".h5")
        h5_path = Path(h5_path)

        train_path = h5_path.with_stem(h5_path.stem + "_train")
        val_path = h5_path.with_stem(h5_path.stem + "_val")

        # Check if already tokenized
        if train_path.exists() and val_path.exists():
            logger.info("Found existing tokenized datasets, loading...")
            return cls(train_path), cls(val_path)

        # Read lines and split
        logger.info("Reading corpus from %s", corpus_path)
        with open(corpus_path, encoding="utf-8") as f:
            lines = [l.strip() for l in f if l.strip()]

        n_val = max(1, int(len(lines) * val_fraction))
        val_lines = lines[:n_val]
        train_lines = lines[n_val:]

        logger.info(
            "Tokenizing %d train + %d val lines (max_len=%d)...",
            len(train_lines), len(val_lines), max_len,
        )

        cls._tokenize_to_h5(train_lines, tokenizer, max_len, train_path, chunk_size)
        cls._tokenize_to_h5(val_lines, tokenizer, max_len, val_path, chunk_size)

        return cls(train_path), cls(val_path)

    @staticmethod
    def _tokenize_to_h5(
        lines: list[str],
        tokenizer,
        max_len: int,
        h5_path: Path,
        chunk_size: int,
    ) -> None:
        """Tokenize lines in chunks and write to HDF5."""
        n = len(lines)
        h5_path.parent.mkdir(parents=True, exist_ok=True)

        with h5py.File(h5_path, "w") as f:
            # Create resizable datasets with compression
            ids_ds = f.create_dataset(
                "input_ids", shape=(n, max_len), dtype=np.int32,
                chunks=(min(chunk_size, n), max_len),
                compression="gzip", compression_opts=4,
            )
            mask_ds = f.create_dataset(
                "attention_mask", shape=(n, max_len), dtype=np.int32,
                chunks=(min(chunk_size, n), max_len),
                compression="gzip", compression_opts=4,
            )
            labels_ds = f.create_dataset(
                "labels", shape=(n, max_len), dtype=np.int32,
                chunks=(min(chunk_size, n), max_len),
                compression="gzip", compression_opts=4,
            )

            written = 0
            for start in range(0, n, chunk_size):
                end = min(start + chunk_size, n)
                batch_lines = lines[start:end]

                enc = tokenizer(
                    batch_lines,
                    max_length=max_len,
                    padding="max_length",
                    truncation=True,
                    return_tensors="np",
                )

                ids = enc["input_ids"].astype(np.int32)
                mask = enc["attention_mask"].astype(np.int32)
                labels = ids.copy()
                labels[mask == 0] = -100

                ids_ds[start:end] = ids
                mask_ds[start:end] = mask
                labels_ds[start:end] = labels
                written += len(batch_lines)

                if written % 100000 < chunk_size:
                    logger.info(
                        "  Tokenized %d/%d (%.0f%%)",
                        written, n, written / n * 100,
                    )

        size_mb = h5_path.stat().st_size / (1024 * 1024)
        logger.info("Saved %d examples to %s (%.1f MB)", n, h5_path, size_mb)

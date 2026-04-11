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

import json
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


class ChatTokenizedH5Dataset(TokenizedH5Dataset):
    """HDF5-backed chat-format dataset for instruction-tuning on Neuroglot.

    Each line of the corpus is a 4-sentence document (paragraph format).
    The four sentences become four conversation turns wrapped as::

        [system] <chat_system_prompt>
        [user]   T0
        [asst]   T1   ← loss here
        [user]   T2
        [asst]   T3   ← loss here

    Labels are ``-100`` everywhere except the assistant-turn token spans,
    so the base model's English generation capacity in the assistant role
    is preserved — only the Neuroglot content is learned. At inference
    time cross-lingual transfer bridges to English interpretation.

    Storage layout is identical to :class:`TokenizedH5Dataset`, so
    ``__getitem__`` and ``__init__`` are inherited unchanged.
    """

    @classmethod
    def from_corpus(
        cls,
        corpus_path: str | Path,
        tokenizer,
        max_len: int = 768,
        h5_path: str | Path | None = None,
        val_fraction: float = 0.05,
        system_prompt: str = "",
        chunk_size: int = 1000,
    ) -> tuple[ChatTokenizedH5Dataset, ChatTokenizedH5Dataset]:
        """Tokenize a paragraph-format corpus into chat-format examples.

        Args:
            corpus_path: Path to the natural-paragraph corpus (one document
                per line, four sentences separated by ``". "``).
            tokenizer: HuggingFace tokenizer with ``apply_chat_template``.
            max_len: Maximum sequence length for the full conversation.
            h5_path: Output HDF5 path. Defaults to ``corpus_path`` with a
                ``.chat.h5`` suffix so it never collides with the flat
                :class:`TokenizedH5Dataset` cache.
            val_fraction: Fraction of lines held out for validation.
            system_prompt: Text used in the ``system`` message of every
                conversation.
            chunk_size: Examples buffered between HDF5 writes.
        """
        corpus_path = Path(corpus_path)
        if h5_path is None:
            h5_path = corpus_path.with_suffix(".chat.h5")
        h5_path = Path(h5_path)

        train_path = h5_path.with_stem(h5_path.stem + "_train")
        val_path = h5_path.with_stem(h5_path.stem + "_val")

        if train_path.exists() and val_path.exists():
            logger.info("Found existing chat tokenized datasets, loading...")
            return cls(train_path), cls(val_path)

        logger.info("Reading corpus from %s", corpus_path)
        with open(corpus_path, encoding="utf-8") as f:
            lines = [l.strip() for l in f if l.strip()]

        n_val = max(1, int(len(lines) * val_fraction))
        val_lines = lines[:n_val]
        train_lines = lines[n_val:]

        logger.info(
            "Chat-tokenizing %d train + %d val lines (max_len=%d)...",
            len(train_lines), len(val_lines), max_len,
        )

        cls._chat_tokenize_to_h5(
            train_lines, tokenizer, max_len, train_path,
            system_prompt, chunk_size,
        )
        cls._chat_tokenize_to_h5(
            val_lines, tokenizer, max_len, val_path,
            system_prompt, chunk_size,
        )

        return cls(train_path), cls(val_path)

    @staticmethod
    def _parse_turns(line: str) -> list[str] | None:
        """Split a paragraph-format line into four turn strings.

        The generator emits documents as four sentences joined with
        ``". "`` and terminated with a trailing period. Returns ``None``
        when the split does not yield exactly four non-empty turns.
        """
        line = line.strip()
        if not line:
            return None
        if line.endswith("."):
            line = line[:-1]
        parts = [p.strip() for p in line.split(". ")]
        if len(parts) != 4 or any(not p for p in parts):
            return None
        return parts

    @staticmethod
    def _build_chat_example(
        turns: list[str],
        tokenizer,
        max_len: int,
        system_prompt: str,
        pad_id: int,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
        """Tokenize a four-turn conversation with labels on assistant spans.

        Uses the prefix-length trick: tokenizes incremental prefixes of
        the conversation to find the exact token offsets of each
        assistant turn inside the full template. No reliance on the
        tokenizer's internal template structure — works with any model
        that implements ``apply_chat_template``.

        Returns ``None`` when the conversation exceeds ``max_len``.
        """
        sys_msg = {"role": "system", "content": system_prompt}
        u0 = {"role": "user", "content": turns[0]}
        a1 = {"role": "assistant", "content": turns[1]}
        u2 = {"role": "user", "content": turns[2]}
        a3 = {"role": "assistant", "content": turns[3]}

        prefix1 = tokenizer.apply_chat_template(
            [sys_msg, u0],
            tokenize=True, add_generation_prompt=True, return_dict=False,
        )
        prefix1_with_a1 = tokenizer.apply_chat_template(
            [sys_msg, u0, a1],
            tokenize=True, add_generation_prompt=False, return_dict=False,
        )
        prefix2 = tokenizer.apply_chat_template(
            [sys_msg, u0, a1, u2],
            tokenize=True, add_generation_prompt=True, return_dict=False,
        )
        full_ids = tokenizer.apply_chat_template(
            [sys_msg, u0, a1, u2, a3],
            tokenize=True, add_generation_prompt=False, return_dict=False,
        )

        n = len(full_ids)
        if n > max_len:
            return None

        ids_arr = np.full((max_len,), pad_id, dtype=np.int32)
        mask_arr = np.zeros((max_len,), dtype=np.int32)
        labels_arr = np.full((max_len,), -100, dtype=np.int32)

        ids_arr[:n] = full_ids
        mask_arr[:n] = 1

        a1_start, a1_end = len(prefix1), len(prefix1_with_a1)
        a3_start, a3_end = len(prefix2), n
        labels_arr[a1_start:a1_end] = full_ids[a1_start:a1_end]
        labels_arr[a3_start:a3_end] = full_ids[a3_start:a3_end]

        return ids_arr, mask_arr, labels_arr

    @staticmethod
    def _chat_tokenize_to_h5(
        lines: list[str],
        tokenizer,
        max_len: int,
        h5_path: Path,
        system_prompt: str,
        chunk_size: int,
    ) -> None:
        """Stream chat-formatted examples into an HDF5 file.

        Processes lines one at a time (per-document offset computation
        is not batchable) and writes to resizable datasets in chunks.
        Lines that fail to parse or overflow ``max_len`` are skipped.
        """
        n = len(lines)
        h5_path.parent.mkdir(parents=True, exist_ok=True)
        pad_id = tokenizer.pad_token_id
        if pad_id is None:
            pad_id = tokenizer.eos_token_id or 0

        chunk_rows = max(1, min(chunk_size, n))

        with h5py.File(h5_path, "w") as f:
            ids_ds = f.create_dataset(
                "input_ids", shape=(0, max_len),
                maxshape=(None, max_len), dtype=np.int32,
                chunks=(chunk_rows, max_len),
                compression="gzip", compression_opts=4,
            )
            mask_ds = f.create_dataset(
                "attention_mask", shape=(0, max_len),
                maxshape=(None, max_len), dtype=np.int32,
                chunks=(chunk_rows, max_len),
                compression="gzip", compression_opts=4,
            )
            labels_ds = f.create_dataset(
                "labels", shape=(0, max_len),
                maxshape=(None, max_len), dtype=np.int32,
                chunks=(chunk_rows, max_len),
                compression="gzip", compression_opts=4,
            )

            buf_ids: list[np.ndarray] = []
            buf_mask: list[np.ndarray] = []
            buf_labels: list[np.ndarray] = []
            written = 0
            skipped = 0

            def flush() -> None:
                nonlocal written
                if not buf_ids:
                    return
                k = len(buf_ids)
                new_size = written + k
                ids_ds.resize((new_size, max_len))
                mask_ds.resize((new_size, max_len))
                labels_ds.resize((new_size, max_len))
                ids_ds[written:new_size] = np.stack(buf_ids)
                mask_ds[written:new_size] = np.stack(buf_mask)
                labels_ds[written:new_size] = np.stack(buf_labels)
                written = new_size
                buf_ids.clear()
                buf_mask.clear()
                buf_labels.clear()

            for line_idx, line in enumerate(lines):
                turns = ChatTokenizedH5Dataset._parse_turns(line)
                if turns is None:
                    skipped += 1
                    continue
                example = ChatTokenizedH5Dataset._build_chat_example(
                    turns, tokenizer, max_len, system_prompt, pad_id,
                )
                if example is None:
                    skipped += 1
                    continue
                ids, mask, labels = example
                buf_ids.append(ids)
                buf_mask.append(mask)
                buf_labels.append(labels)

                if len(buf_ids) >= chunk_size:
                    flush()

                if (line_idx + 1) % 10000 == 0:
                    logger.info(
                        "  Processed %d/%d lines (%d kept, %d skipped)",
                        line_idx + 1, n,
                        written + len(buf_ids), skipped,
                    )

            flush()

        size_mb = h5_path.stat().st_size / (1024 * 1024)
        logger.info(
            "Saved %d chat examples to %s (%.1f MB, %d skipped)",
            written, h5_path, size_mb, skipped,
        )


class EnglishChatTokenizedH5Dataset(TokenizedH5Dataset):
    """HDF5-backed chat dataset for self-distilled English instruct data.

    Reads a JSONL produced by :mod:`scripts.self_distill_english` and
    builds single-turn ``[system, user, assistant]`` chat examples with
    assistant-only loss (same prefix-length trick used by
    :class:`ChatTokenizedH5Dataset`).  Interleaving this with the
    Neuroglot chat corpus during LoRA training defends the base model's
    English generation capacity — the assistant role gets gradient
    signal in both languages, selected by the system prompt at inference.

    Each JSONL record must have at least ``system``, ``user``, and
    ``assistant`` fields (additional fields like ``temperature`` are
    ignored).

    Storage layout is identical to :class:`TokenizedH5Dataset`.
    """

    @classmethod
    def from_corpus(
        cls,
        corpus_path: str | Path,
        tokenizer,
        max_len: int = 768,
        h5_path: str | Path | None = None,
        val_fraction: float = 0.05,
        chunk_size: int = 1000,
    ) -> tuple[EnglishChatTokenizedH5Dataset, EnglishChatTokenizedH5Dataset]:
        """Tokenize a self-distill JSONL into chat-format examples.

        Args:
            corpus_path: Path to the JSONL file, one record per line.
            tokenizer: HuggingFace tokenizer with ``apply_chat_template``.
            max_len: Maximum sequence length for the full conversation.
            h5_path: Output HDF5 path.  Defaults to ``corpus_path`` with
                a ``.chat.h5`` suffix so it never collides with the flat
                tokenizer cache.
            val_fraction: Fraction of records held out for validation.
            chunk_size: Examples buffered between HDF5 writes.
        """
        corpus_path = Path(corpus_path)
        if h5_path is None:
            h5_path = corpus_path.with_suffix(".chat.h5")
        h5_path = Path(h5_path)

        train_path = h5_path.with_stem(h5_path.stem + "_train")
        val_path = h5_path.with_stem(h5_path.stem + "_val")

        if train_path.exists() and val_path.exists():
            logger.info("Found existing English chat datasets, loading...")
            return cls(train_path), cls(val_path)

        logger.info("Reading English JSONL from %s", corpus_path)
        records: list[dict] = []
        with open(corpus_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    continue

        n_val = max(1, int(len(records) * val_fraction))
        val_records = records[:n_val]
        train_records = records[n_val:]

        logger.info(
            "Chat-tokenizing %d train + %d val English records (max_len=%d)...",
            len(train_records), len(val_records), max_len,
        )

        cls._chat_tokenize_to_h5(
            train_records, tokenizer, max_len, train_path, chunk_size,
        )
        cls._chat_tokenize_to_h5(
            val_records, tokenizer, max_len, val_path, chunk_size,
        )

        return cls(train_path), cls(val_path)

    @staticmethod
    def _build_chat_example(
        record: dict,
        tokenizer,
        max_len: int,
        pad_id: int,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
        """Tokenize one [sys, user, asst] conversation with asst-only loss."""
        system = record.get("system", "")
        user = record.get("user", "")
        assistant = record.get("assistant", "")
        if not user or not assistant:
            return None

        sys_msg = {"role": "system", "content": system}
        u_msg = {"role": "user", "content": user}
        a_msg = {"role": "assistant", "content": assistant}

        prefix = tokenizer.apply_chat_template(
            [sys_msg, u_msg],
            tokenize=True, add_generation_prompt=True, return_dict=False,
        )
        full_ids = tokenizer.apply_chat_template(
            [sys_msg, u_msg, a_msg],
            tokenize=True, add_generation_prompt=False, return_dict=False,
        )

        n = len(full_ids)
        if n > max_len:
            return None

        ids_arr = np.full((max_len,), pad_id, dtype=np.int32)
        mask_arr = np.zeros((max_len,), dtype=np.int32)
        labels_arr = np.full((max_len,), -100, dtype=np.int32)

        ids_arr[:n] = full_ids
        mask_arr[:n] = 1
        a_start, a_end = len(prefix), n
        labels_arr[a_start:a_end] = full_ids[a_start:a_end]

        return ids_arr, mask_arr, labels_arr

    @staticmethod
    def _chat_tokenize_to_h5(
        records: list[dict],
        tokenizer,
        max_len: int,
        h5_path: Path,
        chunk_size: int,
    ) -> None:
        n = len(records)
        h5_path.parent.mkdir(parents=True, exist_ok=True)
        pad_id = tokenizer.pad_token_id
        if pad_id is None:
            pad_id = tokenizer.eos_token_id or 0

        chunk_rows = max(1, min(chunk_size, n))

        with h5py.File(h5_path, "w") as f:
            ids_ds = f.create_dataset(
                "input_ids", shape=(0, max_len),
                maxshape=(None, max_len), dtype=np.int32,
                chunks=(chunk_rows, max_len),
                compression="gzip", compression_opts=4,
            )
            mask_ds = f.create_dataset(
                "attention_mask", shape=(0, max_len),
                maxshape=(None, max_len), dtype=np.int32,
                chunks=(chunk_rows, max_len),
                compression="gzip", compression_opts=4,
            )
            labels_ds = f.create_dataset(
                "labels", shape=(0, max_len),
                maxshape=(None, max_len), dtype=np.int32,
                chunks=(chunk_rows, max_len),
                compression="gzip", compression_opts=4,
            )

            buf_ids: list[np.ndarray] = []
            buf_mask: list[np.ndarray] = []
            buf_labels: list[np.ndarray] = []
            written = 0
            skipped = 0

            def flush() -> None:
                nonlocal written
                if not buf_ids:
                    return
                k = len(buf_ids)
                new_size = written + k
                ids_ds.resize((new_size, max_len))
                mask_ds.resize((new_size, max_len))
                labels_ds.resize((new_size, max_len))
                ids_ds[written:new_size] = np.stack(buf_ids)
                mask_ds[written:new_size] = np.stack(buf_mask)
                labels_ds[written:new_size] = np.stack(buf_labels)
                written = new_size
                buf_ids.clear()
                buf_mask.clear()
                buf_labels.clear()

            for rec_idx, rec in enumerate(records):
                example = EnglishChatTokenizedH5Dataset._build_chat_example(
                    rec, tokenizer, max_len, pad_id,
                )
                if example is None:
                    skipped += 1
                    continue
                ids, mask, labels = example
                buf_ids.append(ids)
                buf_mask.append(mask)
                buf_labels.append(labels)

                if len(buf_ids) >= chunk_size:
                    flush()

                if (rec_idx + 1) % 10000 == 0:
                    logger.info(
                        "  Processed %d/%d records (%d kept, %d skipped)",
                        rec_idx + 1, n,
                        written + len(buf_ids), skipped,
                    )

            flush()

        size_mb = h5_path.stat().st_size / (1024 * 1024)
        logger.info(
            "Saved %d English chat examples to %s (%.1f MB, %d skipped)",
            written, h5_path, size_mb, skipped,
        )

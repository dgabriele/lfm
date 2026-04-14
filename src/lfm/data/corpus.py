"""Corpus dataset implementations.

Provides dataset classes for loading text corpora used in structural prior
learning.  Supports both monolingual and multilingual corpus configurations.
"""

from __future__ import annotations

import torch
from torch import Tensor
from torch.utils.data import Dataset

from lfm.data.config import DataConfig


def pad_collate(
    batch: list[tuple[Tensor, int]],
) -> tuple[Tensor, Tensor]:
    """Pad a batch of variable-length ``(tokens, length)`` pairs.

    Pads each sequence on the right with 0 to the batch's max length and
    stacks into a ``(B, max_len)`` ``int64`` tensor (decoder indexing
    requires long).  Lengths are returned as a ``(B,)`` long tensor.
    """
    max_len = max(length for _, length in batch)
    padded = torch.zeros(len(batch), max_len, dtype=torch.long)
    lengths = torch.empty(len(batch), dtype=torch.long)
    for i, (tokens, length) in enumerate(batch):
        padded[i, :length] = tokens[:length].to(torch.long)
        lengths[i] = length
    return padded, lengths


def pad_collate_indexed(
    batch: list[tuple[Tensor, int, int]],
) -> tuple[Tensor, Tensor, Tensor]:
    """Same as :func:`pad_collate` but also returns the original index."""
    max_len = max(length for _, length, _ in batch)
    padded = torch.zeros(len(batch), max_len, dtype=torch.long)
    lengths = torch.empty(len(batch), dtype=torch.long)
    indices = torch.empty(len(batch), dtype=torch.long)
    for i, (tokens, length, orig_idx) in enumerate(batch):
        padded[i, :length] = tokens[:length].to(torch.long)
        lengths[i] = length
        indices[i] = orig_idx
    return padded, lengths, indices


class TextCorpusDataset(Dataset[tuple[Tensor, int, str, int]]):
    """Pre-tokenized text with metadata for collation.

    Can hold both raw text lines (for language detection, etc.) and
    pre-tokenized integer sequences.
    """

    def __init__(
        self, corpus: list[tuple[str, str]], config: DataConfig
    ) -> None:
        self.config = config
        self.data = corpus

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> tuple[str, str]:
        return self.data[idx]


# Backward-compat alias
CorpusDataset = TextCorpusDataset


class MultilingualCorpusDataset(Dataset[tuple[Tensor, int]]):
    """Pre-tokenized multilingual text sequences with EOS.

    Stores variable-length sequences (truncated to ``max_seq_len`` with
    EOS appended).  Padding is deferred to the collate function for
    dynamic per-batch padding, eliminating wasted compute on padding
    tokens.

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
        word_boundary_ids: set[int] | None = None,
    ) -> None:
        self.max_seq_len = max_seq_len
        # Store variable-length int32 tensors.  Padding to max-in-batch is
        # done by ``pad_collate`` on the DataLoader — pre-padding here
        # would allocate N × max_seq_len × 8 bytes (tens of GB for large
        # corpora).  int32 is sufficient for vocabularies ≤ 2**31.
        self.data: list[tuple[Tensor, int]] = []
        for ids in token_ids:
            if len(ids) >= max_seq_len:
                trunc = ids[: max_seq_len - 1]
                if word_boundary_ids is not None:
                    for j in range(len(trunc) - 1, -1, -1):
                        if trunc[j] in word_boundary_ids:
                            trunc = trunc[:j]
                            break
                ids = trunc
            ids_with_eos = ids + [eos_id]
            length = len(ids_with_eos)
            self.data.append(
                (torch.tensor(ids_with_eos, dtype=torch.int32), length)
            )

    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        return len(self.data)

    def __getitem__(self, idx: int) -> tuple[Tensor, int]:
        """Return a single sample: ``(token_ids, length)``.

        Returns:
            Tuple of ``(token_ids, length)`` where ``token_ids`` has
            shape ``(length,)`` — **not yet padded**.
        """
        return self.data[idx]


class IndexedDatasetWrapper(Dataset):
    """Wraps a dataset to additionally return the original sample index.

    Used by contrastive pretraining to look up pre-computed embeddings
    by index.  Works with ``random_split`` subsets: extracts the
    original dataset index from the Subset's index mapping.
    """

    def __init__(self, dataset: Dataset) -> None:
        self.dataset = dataset

    def __len__(self) -> int:
        return len(self.dataset)  # type: ignore[arg-type]

    def __getitem__(self, idx: int) -> tuple[Tensor, int, int]:
        tokens, length = self.dataset[idx]
        # If the dataset is a Subset, map to the original index
        if hasattr(self.dataset, "indices"):
            original_idx = self.dataset.indices[idx]
        else:
            original_idx = idx
        return tokens, length, original_idx


class ConstituentDataset(Dataset):
    """Paired (parent sentence, constituent) dataset for phase 2 training.

    Each sample returns encoder tokens (parent sentence) and decoder
    tokens (constituent).  The encoder sees the full sentence context;
    the decoder is supervised only on the constituent.

    Args:
        sentence_token_ids: Token ID lists for all full sentences.
        constituent_token_ids: Token ID lists for constituents.
        parent_indices: Maps each constituent to its parent sentence index.
        max_seq_len: Maximum sequence length (including EOS).
        eos_id: EOS token ID.
    """

    def __init__(
        self,
        sentence_token_ids: list[list[int]],
        constituent_token_ids: list[list[int]],
        parent_indices: list[int],
        max_seq_len: int,
        eos_id: int,
    ) -> None:
        self.max_seq_len = max_seq_len

        # Pre-process sentences (encoder inputs)
        self._sentences: list[tuple[Tensor, int]] = []
        for ids in sentence_token_ids:
            ids = ids[: max_seq_len - 1]
            ids_eos = ids + [eos_id]
            length = len(ids_eos)
            padded = ids_eos + [0] * (max_seq_len - length)
            self._sentences.append(
                (torch.tensor(padded, dtype=torch.long), length)
            )

        # Pre-process constituents (decoder targets) with parent links
        self.data: list[tuple[int, Tensor, int]] = []
        for ids, parent_idx in zip(constituent_token_ids, parent_indices):
            ids = ids[: max_seq_len - 1]
            ids_eos = ids + [eos_id]
            length = len(ids_eos)
            padded = ids_eos + [0] * (max_seq_len - length)
            self.data.append((
                parent_idx,
                torch.tensor(padded, dtype=torch.long),
                length,
            ))

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> tuple[Tensor, int, Tensor, int]:
        """Return (enc_tokens, enc_length, dec_tokens, dec_length)."""
        parent_idx, dec_tokens, dec_length = self.data[idx]
        enc_tokens, enc_length = self._sentences[parent_idx]
        return enc_tokens, enc_length, dec_tokens, dec_length


class InterleavedLoader:
    """Interleave batches from sentence and constituent DataLoaders.

    Ensures full coverage of both datasets per cycle. The epoch length
    is determined by the larger dataset. The smaller dataset repeats
    as needed to maintain the target mix ratio.

    Each batch is either a sentence batch (2-tuple) or a constituent
    batch (4-tuple), tagged with a boolean flag.

    Args:
        sentence_loader: DataLoader for full sentences.
        constituent_loader: DataLoader for (parent, constituent) pairs.
        mix_ratio: Fraction of batches that are full sentences (0-1).
            0.5 = equal mix. 0.7 = 70% sentences, 30% constituents.
        seed: Random seed for reproducible interleaving.
    """

    def __init__(
        self,
        sentence_loader: torch.utils.data.DataLoader,
        constituent_loader: torch.utils.data.DataLoader,
        mix_ratio: float = 0.5,
        seed: int = 42,
    ) -> None:
        self.sentence_loader = sentence_loader
        self.constituent_loader = constituent_loader
        self.mix_ratio = mix_ratio
        self.seed = seed

        # Total batches per epoch = enough to cover the larger dataset
        n_sent = len(sentence_loader)
        n_const = len(constituent_loader)
        # At mix_ratio, we need n_sent/ratio sentence batches and
        # n_const/(1-ratio) constituent batches. Take the max.
        self._total_batches = max(
            int(n_sent / max(mix_ratio, 0.01)),
            int(n_const / max(1 - mix_ratio, 0.01)),
        )
        self._epoch = 0

    def __len__(self) -> int:
        return self._total_batches

    def __iter__(self):
        """Yield (is_constituent, batch_data) tuples.

        is_constituent=False: batch_data is (tokens, lengths)
        is_constituent=True: batch_data is (enc_tokens, enc_lengths, dec_tokens, dec_lengths)
        """
        import random

        rng = random.Random(self.seed + self._epoch)
        self._epoch += 1

        sent_iter = iter(self.sentence_loader)
        const_iter = iter(self.constituent_loader)

        for i in range(self._total_batches):
            use_sentence = rng.random() < self.mix_ratio

            if use_sentence:
                try:
                    batch = next(sent_iter)
                except StopIteration:
                    sent_iter = iter(self.sentence_loader)
                    batch = next(sent_iter)
                yield False, batch
            else:
                try:
                    batch = next(const_iter)
                except StopIteration:
                    const_iter = iter(self.constituent_loader)
                    batch = next(const_iter)
                yield True, batch


def dynamic_pad_collate(
    batch: list[tuple[Tensor, int]],
) -> tuple[Tensor, Tensor]:
    """Collate variable-length sequences with per-batch dynamic padding.

    Pads all sequences in the batch to the length of the longest
    sequence in that batch, rather than a global maximum.  Combined
    with length-sorted batching, this virtually eliminates padding waste.

    Args:
        batch: List of ``(token_ids, length)`` tuples from the dataset.

    Returns:
        Tuple of ``(padded_ids, lengths)`` where ``padded_ids`` has shape
        ``(batch_size, max_len_in_batch)`` and ``lengths`` is ``(batch_size,)``.
    """
    tokens_list, lengths_list = zip(*batch)
    max_len = max(lengths_list)
    padded = torch.zeros(len(batch), max_len, dtype=torch.long)
    for i, (toks, length) in enumerate(batch):
        padded[i, :length] = toks
    return padded, torch.tensor(lengths_list, dtype=torch.long)


class LengthSortedSampler(torch.utils.data.Sampler):
    """Sampler that groups similar-length sequences into batches.

    Sorts by length, then chunks into buckets of ``bucket_size`` and
    shuffles within each bucket.  This keeps batches length-homogeneous
    (minimizing padding) while preserving stochasticity.

    Args:
        lengths: Per-sample sequence lengths.
        batch_size: Batch size for bucketing.
        bucket_multiplier: Number of batches per bucket for shuffling.
            Larger = more padding efficiency, less randomness.
        seed: Random seed for reproducible shuffling.
    """

    def __init__(
        self,
        lengths: list[int],
        batch_size: int,
        bucket_multiplier: int = 10,
        seed: int = 42,
    ) -> None:
        self.lengths = lengths
        self.batch_size = batch_size
        self.bucket_size = batch_size * bucket_multiplier
        self.seed = seed
        self._epoch = 0

    def __iter__(self):
        import random

        rng = random.Random(self.seed + self._epoch)
        self._epoch += 1

        # Sort indices by length
        indices = sorted(range(len(self.lengths)), key=lambda i: self.lengths[i])

        # Chunk into buckets, shuffle within each bucket
        for start in range(0, len(indices), self.bucket_size):
            bucket = indices[start : start + self.bucket_size]
            rng.shuffle(bucket)
            yield from bucket

    def __len__(self) -> int:
        return len(self.lengths)

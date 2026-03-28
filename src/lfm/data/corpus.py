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
        self.data: list[tuple[Tensor, int]] = []
        for ids in token_ids:
            if len(ids) >= max_seq_len:
                # Truncate at last word boundary before the limit.
                # Word boundaries are tokens whose sentencepiece piece
                # starts with ▁ (space prefix).
                trunc = ids[: max_seq_len - 1]
                if word_boundary_ids is not None:
                    # Find last word-boundary token position
                    for j in range(len(trunc) - 1, -1, -1):
                        if trunc[j] in word_boundary_ids:
                            trunc = trunc[:j]
                            break
                ids = trunc
            ids_with_eos = ids + [eos_id]
            length = len(ids_with_eos)
            padded = ids_with_eos + [0] * (max_seq_len - length)
            self.data.append(
                (torch.tensor(padded, dtype=torch.long), length)
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
        self._lengths = lengths
        self._batch_size = batch_size
        self._bucket_size = batch_size * bucket_multiplier
        self._seed = seed
        self._epoch = 0

    def set_epoch(self, epoch: int) -> None:
        """Set epoch for reproducible cross-epoch shuffling."""
        self._epoch = epoch

    def __iter__(self):
        import random

        rng = random.Random(self._seed + self._epoch)

        # Sort indices by length
        sorted_indices = sorted(
            range(len(self._lengths)), key=lambda i: self._lengths[i]
        )

        # Chunk into buckets and shuffle within each bucket
        batches = []
        for start in range(0, len(sorted_indices), self._bucket_size):
            bucket = sorted_indices[start : start + self._bucket_size]
            rng.shuffle(bucket)
            # Split bucket into batch-sized chunks
            for b_start in range(0, len(bucket), self._batch_size):
                batches.append(bucket[b_start : b_start + self._batch_size])

        # Shuffle batch order
        rng.shuffle(batches)

        for batch in batches:
            yield from batch

    def __len__(self) -> int:
        return len(self._lengths)


class PackedCorpusDataset(Dataset[tuple[Tensor, int]]):
    """Packed multi-sentence sequences with EOS delimiters.

    Concatenates multiple sentences into each training sequence, separated
    by EOS tokens, so the decoder learns to emit EOS as a natural sentence
    boundary — not just at the very end of a sequence.

    Each packed sequence looks like::

        [sent1_tok1, ..., sent1_tokN, EOS, sent2_tok1, ..., sent2_tokM, EOS, PAD, ...]

    This ensures every training sample contains at least one EOS at a
    non-final position, teaching the decoder that EOS is a regular part
    of the token distribution.

    Args:
        token_ids: List of integer token id lists, one per sentence.
        max_seq_len: Maximum packed sequence length.
        eos_id: EOS token index used as sentence delimiter.
        seed: Random seed for reproducible packing order.
    """

    def __init__(
        self,
        token_ids: list[list[int]],
        max_seq_len: int,
        eos_id: int,
        languages: list[str] | None = None,
        seed: int = 42,
    ) -> None:
        import random
        from collections import defaultdict

        self.eos_id = eos_id
        self.max_seq_len = max_seq_len
        self.data: list[tuple[Tensor, int]] = []
        rng = random.Random(seed)

        # Group sentences by language for same-language packing
        if languages is not None:
            by_lang: dict[str, list[list[int]]] = defaultdict(list)
            for ids, lang in zip(token_ids, languages):
                by_lang[lang].append(ids)
            groups = list(by_lang.values())
        else:
            groups = [token_ids]

        # Shuffle within each language, then greedily pack
        for sentences in groups:
            rng.shuffle(sentences)
            i = 0
            while i < len(sentences):
                packed: list[int] = []
                while i < len(sentences):
                    ids = sentences[i]
                    needed = len(ids) + 1  # sentence + EOS delimiter
                    if len(packed) + needed > max_seq_len:
                        break
                    packed.extend(ids)
                    packed.append(eos_id)
                    i += 1

                if not packed:
                    # Single sentence too long — truncate it
                    ids = sentences[i]
                    packed = ids[: max_seq_len - 1] + [eos_id]
                    i += 1

                length = len(packed)
                padded = packed + [0] * (max_seq_len - length)
                self.data.append(
                    (torch.tensor(padded, dtype=torch.long), length)
                )

        # Shuffle packed sequences across languages for training
        rng.shuffle(self.data)

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> tuple[Tensor, int]:
        return self.data[idx]

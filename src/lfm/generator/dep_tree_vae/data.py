"""Data loading for DepTreeVAE training.

Two-phase loading for memory efficiency:
  1. First run: read JSONL, tokenize with SPM, save as compact binary cache.
  2. Subsequent runs: load cache directly (fast, low memory).

The cache stores pre-tokenized sequences as contiguous int16 arrays
with an offset index, avoiding Python object overhead for millions
of samples.
"""

from __future__ import annotations

import json
import logging
import struct
from pathlib import Path

import numpy as np
import sentencepiece as spm
import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset, Sampler, random_split

from lfm.generator.dep_tree_vae.config import DEP_REL_TO_ID, DepTreeVAEConfig
from lfm.generator.dep_tree_vae.skeleton import SKEL_BOS, SKEL_EOS

logger = logging.getLogger(__name__)


class DepTreeDataset(Dataset):
    """Memory-efficient dataset for dependency-annotated IPA sentences.

    Stores all sequences in two flat int16 numpy arrays (interleaved
    tokens and skeleton roles) with offset/length indices.  Total RAM
    is ~2 bytes per token — 16M samples × 50 avg tokens ≈ 1.6GB.
    """

    def __init__(
        self,
        cache_dir: Path,
        max_samples: int | None = None,
    ) -> None:
        cache_dir = Path(cache_dir)

        # Load index: (offset_inter, len_inter, offset_skel, len_skel) per sample
        self._index = np.load(cache_dir / "index.npy")  # (N, 4) int64
        self._interleaved = np.load(cache_dir / "interleaved.npy")  # flat int16
        self._skeletons = np.load(cache_dir / "skeletons.npy")  # flat int16

        if max_samples and max_samples < len(self._index):
            self._index = self._index[:max_samples]

        logger.info(
            "Loaded cache: %d samples, interleaved=%dM tokens, skeletons=%dM tokens",
            len(self._index),
            len(self._interleaved) // 1_000_000,
            len(self._skeletons) // 1_000_000,
        )

    def __len__(self) -> int:
        return len(self._index)

    def __getitem__(self, idx: int) -> dict:
        off_i, len_i, off_s, len_s = self._index[idx]
        return {
            "interleaved": self._interleaved[off_i : off_i + len_i].astype(np.int64),
            "skeleton": self._skeletons[off_s : off_s + len_s].astype(np.int64),
        }

    @property
    def interleaved_vocab_size(self) -> int:
        """Infer from the max token id in the interleaved data."""
        return int(self._interleaved.max()) + 1


def build_cache(
    jsonl_path: str | Path,
    sp: spm.SentencePieceProcessor,
    max_seq_len: int,
    cache_dir: str | Path,
    max_samples: int | None = None,
) -> None:
    """Read JSONL, tokenize, and save as compact binary cache."""
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    spm_size = sp.get_piece_size()
    eos_id = spm_size + 1
    role_offset = spm_size + 2

    all_interleaved: list[np.ndarray] = []
    all_skeletons: list[np.ndarray] = []
    index: list[tuple[int, int, int, int]] = []  # (off_i, len_i, off_s, len_s)

    inter_offset = 0
    skel_offset = 0
    filtered = 0
    total = 0

    with open(jsonl_path) as f:
        for line in f:
            if max_samples and len(index) >= max_samples:
                break
            total += 1

            rec = json.loads(line)
            ipa_words = rec["ipa"].split()
            dep_labels = rec["dep_labels"]

            if len(ipa_words) != len(dep_labels):
                filtered += 1
                continue

            # Build interleaved sequence
            interleaved: list[int] = []
            role_sequence: list[int] = []

            for label, word in zip(dep_labels, ipa_words):
                role_id = DEP_REL_TO_ID.get(label, DEP_REL_TO_ID.get("dep", 0))
                interleaved.append(role_offset + role_id)
                for tid in sp.encode(word, out_type=int):
                    interleaved.append(tid)
                role_sequence.append(role_id)

            interleaved.append(eos_id)

            if len(interleaved) > max_seq_len:
                filtered += 1
                continue

            skeleton = [SKEL_BOS] + role_sequence + [SKEL_EOS]

            inter_arr = np.array(interleaved, dtype=np.int16)
            skel_arr = np.array(skeleton, dtype=np.int16)

            index.append((inter_offset, len(inter_arr), skel_offset, len(skel_arr)))
            all_interleaved.append(inter_arr)
            all_skeletons.append(skel_arr)

            inter_offset += len(inter_arr)
            skel_offset += len(skel_arr)

            if len(index) % 1_000_000 == 0:
                logger.info("  cached %dM samples...", len(index) // 1_000_000)

    # Concatenate and save
    inter_flat = np.concatenate(all_interleaved)
    skel_flat = np.concatenate(all_skeletons)
    index_arr = np.array(index, dtype=np.int64)

    np.save(cache_dir / "interleaved.npy", inter_flat)
    np.save(cache_dir / "skeletons.npy", skel_flat)
    np.save(cache_dir / "index.npy", index_arr)

    logger.info(
        "Cache built: %d samples (filtered %d/%d), "
        "interleaved=%dM tokens (%.1f MB), skeletons=%dM tokens (%.1f MB)",
        len(index), filtered, total,
        len(inter_flat) // 1_000_000, inter_flat.nbytes / 1e6,
        len(skel_flat) // 1_000_000, skel_flat.nbytes / 1e6,
    )


def collate_dep_tree(batch: list[dict]) -> dict[str, Tensor]:
    """Collate variable-length dep tree samples into padded tensors."""
    max_inter = max(len(s["interleaved"]) for s in batch)
    max_skel = max(len(s["skeleton"]) for s in batch)

    b = len(batch)
    tokens = torch.zeros(b, max_inter, dtype=torch.long)
    lengths = torch.zeros(b, dtype=torch.long)
    role_ids = torch.zeros(b, max_skel, dtype=torch.long)
    role_lengths = torch.zeros(b, dtype=torch.long)

    for i, s in enumerate(batch):
        inter = s["interleaved"]
        skel = s["skeleton"]
        tokens[i, : len(inter)] = torch.from_numpy(inter)
        lengths[i] = len(inter)
        role_ids[i, : len(skel)] = torch.from_numpy(skel)
        role_lengths[i] = len(skel)

    return {
        "tokens": tokens,
        "lengths": lengths,
        "role_ids": role_ids,
        "role_lengths": role_lengths,
    }


def build_dataloaders(
    cfg: DepTreeVAEConfig,
) -> tuple[DataLoader, DataLoader, spm.SentencePieceProcessor, int]:
    """Build train/val DataLoaders from config.

    On first run, builds a binary cache from the JSONL source.
    Subsequent runs load the cache directly.
    """
    sp = spm.SentencePieceProcessor()
    sp.load(cfg.spm_model_path)

    dataset_dir = Path(cfg.dataset_path)
    cache_dir = dataset_dir / "cache"
    index_path = cache_dir / "index.npy"

    # Build cache if missing
    if not index_path.exists():
        jsonl_path = dataset_dir / "sentences.jsonl"
        logger.info("Building binary cache from %s ...", jsonl_path)
        build_cache(
            jsonl_path, sp, cfg.max_seq_len, cache_dir,
            max_samples=cfg.max_samples,
        )

    dataset = DepTreeDataset(cache_dir, max_samples=cfg.max_samples)

    # Infer vocab size
    spm_size = sp.get_piece_size()
    vocab_size = spm_size + 2 + len(DEP_REL_TO_ID)  # SPM + BOS/EOS + roles

    val_size = max(1, int(len(dataset) * cfg.val_fraction))
    train_size = len(dataset) - val_size

    generator = torch.Generator().manual_seed(cfg.seed)
    train_ds, val_ds = random_split(
        dataset, [train_size, val_size], generator=generator,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        drop_last=True,
        collate_fn=collate_dep_tree,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        drop_last=False,
        collate_fn=collate_dep_tree,
    )

    logger.info(
        "Data: %d train, %d val, vocab=%d (spm=%d + 2 specials + %d roles)",
        train_size, val_size, vocab_size,
        spm_size, len(DEP_REL_TO_ID),
    )
    return train_loader, val_loader, sp, vocab_size

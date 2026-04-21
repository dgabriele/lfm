"""Extended data pipeline for tree-structured diffusion VAE.

Adds per-token dependency tree depth and per-role token counts to the
binary cache. Depth controls the noise schedule, token counts train
the length predictor.

Cache build is parallelized: chunks of JSONL are processed by worker
processes (JSON parse + SPM encode + depth compute), then assembled.
"""

from __future__ import annotations

import json
import logging
import multiprocessing as mp
from pathlib import Path

import numpy as np
import sentencepiece as spm
import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset

from lfm.generator.dep_tree_vae.config import DEP_REL_TO_ID
from lfm.generator.dep_tree_vae.skeleton import SKEL_BOS, SKEL_EOS

logger = logging.getLogger(__name__)


def _compute_depths(dep_heads: list[int]) -> list[int]:
    """Tree depth per word from dependency head indices (0 = root)."""
    n = len(dep_heads)
    depths = [-1] * n

    def _d(i: int) -> int:
        if depths[i] >= 0:
            return depths[i]
        h = dep_heads[i]
        if h == 0 or h - 1 == i or h - 1 < 0 or h - 1 >= n:
            depths[i] = 0
            return 0
        depths[i] = _d(h - 1) + 1
        return depths[i]

    for i in range(n):
        _d(i)
    return depths


def _process_chunk(args: tuple) -> list[dict] | None:
    """Worker: process a chunk of JSONL lines into tokenized records.

    Each record has: interleaved tokens, skeleton, depths, role_token_counts.
    """
    lines, spm_path, max_seq_len = args

    sp = spm.SentencePieceProcessor()
    sp.Load(spm_path)
    spm_size = sp.get_piece_size()
    eos_id = spm_size + 1
    role_offset = spm_size + 2

    results = []
    for line in lines:
        rec = json.loads(line)
        ipa_words = rec["ipa"].split()
        dep_labels = rec["dep_labels"]
        dep_heads = rec.get("dep_heads", [0] * len(dep_labels))

        if len(ipa_words) != len(dep_labels):
            continue

        word_depths = _compute_depths(dep_heads)

        interleaved = []
        token_depths = []
        role_sequence = []
        tokens_per_role = []

        for label, word, depth in zip(dep_labels, ipa_words, word_depths):
            role_id = DEP_REL_TO_ID.get(label, DEP_REL_TO_ID.get("dep", 0))
            interleaved.append(role_offset + role_id)
            token_depths.append(depth)
            word_tids = sp.encode(word, out_type=int)
            for tid in word_tids:
                interleaved.append(tid)
                token_depths.append(depth)
            role_sequence.append(role_id)
            tokens_per_role.append(len(word_tids))

        interleaved.append(eos_id)
        token_depths.append(0)

        if len(interleaved) > max_seq_len:
            continue

        skeleton = [SKEL_BOS] + role_sequence + [SKEL_EOS]
        role_counts_padded = [0] + tokens_per_role + [0]

        results.append({
            "interleaved": interleaved,
            "depths": token_depths,
            "skeleton": skeleton,
            "role_token_counts": role_counts_padded,
        })

    return results


def build_diffusion_cache(
    jsonl_path: str | Path,
    spm_path: str,
    max_seq_len: int,
    cache_dir: str | Path,
    max_samples: int | None = None,
    num_workers: int = 0,
) -> None:
    """Build cache with depths + role token counts. Parallelized."""
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    sp = spm.SentencePieceProcessor()
    sp.Load(spm_path)
    spm_size = sp.get_piece_size()
    eos_id = spm_size + 1
    role_offset = spm_size + 2

    # Two-pass: pass 1 counts sizes, pass 2 fills pre-allocated arrays.
    # Sequential but memory-efficient — no 10M-record list in RAM.

    # Pass 1: scan to count sizes and build index
    index: list[tuple[int, int, int, int]] = []
    inter_offset = 0
    skel_offset = 0
    filtered = 0

    with open(jsonl_path) as f:
        for line in f:
            if max_samples and len(index) >= max_samples:
                break
            rec = json.loads(line)
            ipa_words = rec["ipa"].split()
            dep_labels = rec["dep_labels"]
            if len(ipa_words) != len(dep_labels):
                filtered += 1
                continue
            inter_len = sum(1 + len(sp.encode(w, out_type=int)) for w in ipa_words) + 1
            if inter_len > max_seq_len:
                filtered += 1
                continue
            skel_len = len(dep_labels) + 2
            index.append((inter_offset, inter_len, skel_offset, skel_len))
            inter_offset += inter_len
            skel_offset += skel_len
            if len(index) % 2_000_000 == 0:
                logger.info("  pass 1: %dM samples...", len(index) // 1_000_000)

    logger.info("  pass 1: %d samples, %dM tokens", len(index), inter_offset // 1_000_000)

    # Pre-allocate flat arrays
    inter_flat = np.zeros(inter_offset, dtype=np.int16)
    skel_flat = np.zeros(skel_offset, dtype=np.int16)
    depth_flat = np.zeros(inter_offset, dtype=np.int8)
    role_counts_flat = np.zeros(skel_offset, dtype=np.int8)
    index_arr = np.array(index, dtype=np.int64)

    # Pass 2: fill arrays
    sample_idx = 0
    with open(jsonl_path) as f:
        for line in f:
            if sample_idx >= len(index):
                break
            rec = json.loads(line)
            ipa_words = rec["ipa"].split()
            dep_labels = rec["dep_labels"]
            dep_heads = rec.get("dep_heads", [0] * len(dep_labels))
            if len(ipa_words) != len(dep_labels):
                continue

            word_depths = _compute_depths(dep_heads)

            interleaved = []
            token_depths = []
            role_sequence = []
            tokens_per_role = []

            for label, word, depth in zip(dep_labels, ipa_words, word_depths):
                role_id = DEP_REL_TO_ID.get(label, DEP_REL_TO_ID.get("dep", 0))
                interleaved.append(role_offset + role_id)
                token_depths.append(depth)
                word_tids = sp.encode(word, out_type=int)
                for tid in word_tids:
                    interleaved.append(tid)
                    token_depths.append(depth)
                role_sequence.append(role_id)
                tokens_per_role.append(len(word_tids))

            interleaved.append(eos_id)
            token_depths.append(0)

            if len(interleaved) > max_seq_len:
                continue

            skeleton = [SKEL_BOS] + role_sequence + [SKEL_EOS]
            role_counts_padded = [0] + tokens_per_role + [0]

            off_i, len_i, off_s, len_s = index[sample_idx]
            inter_flat[off_i:off_i + len_i] = interleaved
            depth_flat[off_i:off_i + len_i] = np.clip(token_depths, 0, 127)
            skel_flat[off_s:off_s + len_s] = skeleton
            role_counts_flat[off_s:off_s + len_s] = np.clip(role_counts_padded, 0, 127)
            sample_idx += 1

            if sample_idx % 2_000_000 == 0:
                logger.info("  pass 2: %dM filled...", sample_idx // 1_000_000)

    np.save(cache_dir / "interleaved.npy", inter_flat)
    np.save(cache_dir / "skeletons.npy", skel_flat)
    np.save(cache_dir / "depths.npy", depth_flat)
    np.save(cache_dir / "role_token_counts.npy", role_counts_flat)
    np.save(cache_dir / "index.npy", index_arr)

    logger.info(
        "Diffusion cache: %d samples, %dM tokens, depths [0, %d]",
        len(index), inter_offset // 1_000_000, depth_flat.max(),
    )


class DiffusionDepTreeDataset(Dataset):
    """Dataset with per-token tree depths and per-role token counts."""

    def __init__(self, cache_dir: Path, max_samples: int | None = None) -> None:
        cache_dir = Path(cache_dir)
        self._index = np.load(cache_dir / "index.npy")
        self._interleaved = np.load(cache_dir / "interleaved.npy")
        self._skeletons = np.load(cache_dir / "skeletons.npy")
        self._depths = np.load(cache_dir / "depths.npy")
        self._role_token_counts = np.load(cache_dir / "role_token_counts.npy")

        if max_samples and max_samples < len(self._index):
            self._index = self._index[:max_samples]

        logger.info(
            "Loaded diffusion cache: %d samples, depths [0, %d]",
            len(self._index), self._depths.max(),
        )

    def __len__(self) -> int:
        return len(self._index)

    def __getitem__(self, idx: int) -> dict:
        off_i, len_i, off_s, len_s = self._index[idx]
        return {
            "interleaved": self._interleaved[off_i:off_i + len_i].astype(np.int64),
            "skeleton": self._skeletons[off_s:off_s + len_s].astype(np.int64),
            "depths": self._depths[off_i:off_i + len_i].astype(np.int64),
            "role_token_counts": self._role_token_counts[off_s:off_s + len_s].astype(np.int64),
        }


def collate_diffusion(batch: list[dict]) -> dict[str, Tensor]:
    """Collate with depths and per-role token counts."""
    max_inter = max(len(s["interleaved"]) for s in batch)
    max_skel = max(len(s["skeleton"]) for s in batch)
    b = len(batch)

    tokens = torch.zeros(b, max_inter, dtype=torch.long)
    depths = torch.zeros(b, max_inter, dtype=torch.long)
    lengths = torch.zeros(b, dtype=torch.long)
    role_ids = torch.zeros(b, max_skel, dtype=torch.long)
    role_lengths = torch.zeros(b, dtype=torch.long)
    role_token_counts = torch.zeros(b, max_skel, dtype=torch.long)

    for i, s in enumerate(batch):
        n_i = len(s["interleaved"])
        n_s = len(s["skeleton"])
        tokens[i, :n_i] = torch.tensor(s["interleaved"])
        depths[i, :n_i] = torch.tensor(s["depths"])
        lengths[i] = n_i
        role_ids[i, :n_s] = torch.tensor(s["skeleton"])
        role_lengths[i] = n_s
        role_token_counts[i, :n_s] = torch.tensor(s["role_token_counts"])

    return {
        "tokens": tokens,
        "depths": depths,
        "lengths": lengths,
        "role_ids": role_ids,
        "role_lengths": role_lengths,
        "role_token_counts": role_token_counts,
    }


def build_diffusion_dataloaders(cfg):
    """Build train/val loaders with depths. Builds cache if needed."""
    from pathlib import Path as _P

    dataset_path = _P(cfg.dataset_path)
    cache_dir = dataset_path / "diffusion_cache"

    if not (cache_dir / "depths.npy").exists():
        logger.info("Building diffusion cache...")
        build_diffusion_cache(
            dataset_path / "sentences.jsonl",
            cfg.spm_model_path,
            cfg.max_seq_len,
            cache_dir,
            cfg.max_samples,
        )

    ds = DiffusionDepTreeDataset(cache_dir, cfg.max_samples)
    n = len(ds)
    n_val = int(n * cfg.val_fraction)
    n_train = n - n_val

    train_ds, val_ds = torch.utils.data.random_split(
        ds, [n_train, n_val],
        generator=torch.Generator().manual_seed(cfg.seed),
    )

    sp = spm.SentencePieceProcessor()
    sp.Load(cfg.spm_model_path)
    vocab_size = sp.get_piece_size() + 2 + len(DEP_REL_TO_ID)

    train_loader = DataLoader(
        train_ds, batch_size=cfg.batch_size, shuffle=True,
        collate_fn=collate_diffusion, num_workers=0, pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=cfg.batch_size, shuffle=False,
        collate_fn=collate_diffusion, num_workers=0, pin_memory=True,
    )

    logger.info("Data: %d train, %d val, vocab=%d", n_train, n_val, vocab_size)
    return train_loader, val_loader, sp, vocab_size

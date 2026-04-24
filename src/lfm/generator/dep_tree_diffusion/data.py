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


_CHUNK_SIZE = 100_000


def build_diffusion_cache(
    jsonl_path: str | Path,
    spm_path: str,
    max_seq_len: int,
    cache_dir: str | Path,
    max_samples: int | None = None,
    num_workers: int = 8,
) -> None:
    """Build cache with depths + role token counts.

    Single pass: read JSONL in chunks, process chunks in parallel
    (JSON parse + SPM encode + depth compute), then assemble into
    pre-allocated flat arrays sequentially.
    """
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    # Read lines and dispatch to worker pool in chunks
    logger.info("Reading JSONL and dispatching to %d workers...", num_workers)
    chunks: list[list[str]] = []
    current_chunk: list[str] = []
    total_lines = 0

    with open(jsonl_path) as f:
        for line in f:
            current_chunk.append(line)
            total_lines += 1
            if len(current_chunk) >= _CHUNK_SIZE:
                chunks.append(current_chunk)
                current_chunk = []
            if max_samples and total_lines >= max_samples * 1.2:
                break
        if current_chunk:
            chunks.append(current_chunk)

    logger.info("  %d lines in %d chunks, processing...", total_lines, len(chunks))

    chunk_args = [(chunk, spm_path, max_seq_len) for chunk in chunks]

    if num_workers > 1:
        with mp.Pool(num_workers) as pool:
            chunk_results = []
            for i, result in enumerate(pool.imap(_process_chunk, chunk_args)):
                chunk_results.append(result)
                done = sum(len(r) for r in chunk_results)
                if (i + 1) % 10 == 0:
                    logger.info("  processed %d/%d chunks (%dM samples)...",
                                i + 1, len(chunks), done // 1_000_000)
    else:
        chunk_results = [_process_chunk(a) for a in chunk_args]

    # Flatten and optionally truncate to max_samples
    all_records: list[dict] = []
    for cr in chunk_results:
        if cr:
            all_records.extend(cr)
        if max_samples and len(all_records) >= max_samples:
            all_records = all_records[:max_samples]
            break

    n = len(all_records)
    logger.info("  %d samples after filtering, assembling arrays...", n)

    # Compute offsets (sequential prefix sum — fast, no SPM work)
    inter_offset = 0
    skel_offset = 0
    index: list[tuple[int, int, int, int]] = []
    for rec in all_records:
        i_len = len(rec["interleaved"])
        s_len = len(rec["skeleton"])
        index.append((inter_offset, i_len, skel_offset, s_len))
        inter_offset += i_len
        skel_offset += s_len

    # Pre-allocate and fill
    inter_flat = np.zeros(inter_offset, dtype=np.int16)
    skel_flat = np.zeros(skel_offset, dtype=np.int16)
    depth_flat = np.zeros(inter_offset, dtype=np.int8)
    role_counts_flat = np.zeros(skel_offset, dtype=np.int8)

    for i, rec in enumerate(all_records):
        off_i, len_i, off_s, len_s = index[i]
        inter_flat[off_i:off_i + len_i] = rec["interleaved"]
        depth_flat[off_i:off_i + len_i] = np.clip(rec["depths"], 0, 127)
        skel_flat[off_s:off_s + len_s] = rec["skeleton"]
        role_counts_flat[off_s:off_s + len_s] = np.clip(rec["role_token_counts"], 0, 127)
        if (i + 1) % 2_000_000 == 0:
            logger.info("  assembled %dM/%dM...", (i + 1) // 1_000_000, n // 1_000_000)

    index_arr = np.array(index, dtype=np.int64)
    np.save(cache_dir / "interleaved.npy", inter_flat)
    np.save(cache_dir / "skeletons.npy", skel_flat)
    np.save(cache_dir / "depths.npy", depth_flat)
    np.save(cache_dir / "role_token_counts.npy", role_counts_flat)
    np.save(cache_dir / "index.npy", index_arr)

    logger.info(
        "Diffusion cache: %d samples, %dM tokens, depths [0, %d]",
        n, inter_offset // 1_000_000, depth_flat.max(),
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

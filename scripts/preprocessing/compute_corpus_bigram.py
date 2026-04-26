"""One-time precompute of the training-corpus top-K bigram distribution.

For each training sample (delimited by ``index.npy``), drops role tokens
(IDs ≥ ``decoder_vocab``), then counts adjacent (a, b) token pairs.
Saves the top-K most frequent bigrams + their normalized probabilities,
plus the residual ``oov_prob`` (mass not in top-K).

This file is the reference distribution for the contrastive game's
``bigram_kl`` regularizer, which replaces the unigram-level
``corpus_kl``.  Cycles like "the the" / "of of" produce bigrams that
do not exist in this table; bigram-KL pushes the model's output
marginal away from such pairs at the layer where they appear.

Usage:
    python scripts/preprocessing/compute_corpus_bigram.py \\
        [--cache PATH] [--out PATH] [--top-k 50000]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import sentencepiece as spm
import yaml

sys.path.insert(0, "/workspace/lfm/src")
sys.path.insert(0, "/home/daniel/projects/lfm/src")
from lfm.generator.dep_tree_vae.config import DepTreeVAEConfig


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default=None)
    p.add_argument("--cache", default=None)
    p.add_argument("--out", default=None)
    p.add_argument("--top-k", type=int, default=50_000,
                   help="Number of most frequent bigrams to retain")
    p.add_argument("--smoothing", type=float, default=1.0,
                   help="add-k smoothing for kept bigrams")
    args = p.parse_args()

    on_vast = Path("/workspace/lfm").exists()
    cfg_default = (
        "/workspace/lfm/configs/dep_tree_vae_vast.yaml" if on_vast
        else "/home/daniel/projects/lfm/configs/dep_tree_vae_vast.yaml"
    )
    cache_default = (
        "/workspace/lfm/data/datasets/english-dep-trees-v16-depth4/cache" if on_vast
        else "/home/daniel/projects/lfm/data/datasets/english-dep-trees-v16/cache_depth4"
    )

    cfg_path = args.config or cfg_default
    cache_dir = Path(args.cache or cache_default)
    out_path = Path(args.out or (cache_dir / "bigram_topk.npz"))

    cfg_dict = yaml.safe_load(open(cfg_path))
    cfg_dict.pop("model_type", None)
    spm_path = cfg_dict.get("spm_model_path", "")
    if not Path(spm_path).exists():
        spm_path = (
            "/workspace/lfm/data/models/v15b_ipa/spm.model" if on_vast
            else "/home/daniel/projects/lfm/data/models/v15b_ipa/spm.model"
        )
        cfg_dict["spm_model_path"] = spm_path
    cfg_dict["dataset_path"] = str(cache_dir.parent)
    cfg_dict["output_dir"] = "/tmp/dummy"
    cfg = DepTreeVAEConfig(**cfg_dict)

    sp = spm.SentencePieceProcessor(model_file=cfg.spm_model_path)
    spm_size = sp.get_piece_size()
    decoder_vocab = cfg.spm_vocab_size + 2  # SPM + BOS + EOS

    print(f"Reading cache from {cache_dir}")
    interleaved = np.load(cache_dir / "interleaved.npy", mmap_mode="r")
    index = np.load(cache_dir / "index.npy", mmap_mode="r")
    print(f"  {len(interleaved):,} tokens, {len(index):,} samples")

    # Count adjacent SPM-vocab bigrams within each sample.
    # The flat interleaved array has role tokens (id >= decoder_vocab)
    # interspersed with content tokens; we drop those before pairing
    # so every bigram represents an actual decoder-output transition.
    print(f"Counting bigrams (decoder_vocab={decoder_vocab}, "
          f"V² = {decoder_vocab * decoder_vocab:,})...")
    counts: dict[int, int] = {}
    chunk = 1_000_000  # samples per progress tick

    for s_start in range(0, len(index), chunk):
        s_end = min(s_start + chunk, len(index))
        for sample_i in range(s_start, s_end):
            row = index[sample_i]
            start, length = int(row[0]), int(row[1])
            seg = np.asarray(interleaved[start:start + length])
            # Drop role tokens (encoder-only IDs ≥ decoder_vocab).
            content = seg[seg < decoder_vocab]
            if content.size < 2:
                continue
            # Pack adjacent pairs as 32-bit keys (a * V + b) for hashing.
            a = content[:-1].astype(np.int64)
            b = content[1:].astype(np.int64)
            keys = a * decoder_vocab + b
            uniq, cts = np.unique(keys, return_counts=True)
            for k, c in zip(uniq.tolist(), cts.tolist()):
                counts[k] = counts.get(k, 0) + c
        print(f"  ...{s_end:,}/{len(index):,} samples ({len(counts):,} unique bigrams)")

    print(f"\nTotal unique bigrams: {len(counts):,}")
    if not counts:
        raise RuntimeError("No bigrams found — cache may be malformed")

    # Top-K by frequency.
    keys = np.fromiter(counts.keys(), dtype=np.int64, count=len(counts))
    cnts = np.fromiter(counts.values(), dtype=np.int64, count=len(counts))
    total = int(cnts.sum())

    K = min(args.top_k, len(counts))
    top_idx = np.argpartition(cnts, -K)[-K:]
    top_idx = top_idx[np.argsort(-cnts[top_idx])]  # descending by count
    top_keys = keys[top_idx]
    top_cnts = cnts[top_idx].astype(np.float64)

    pairs = np.stack([top_keys // decoder_vocab, top_keys % decoder_vocab], axis=1).astype(np.int32)
    smoothed = top_cnts + args.smoothing
    probs = smoothed / total
    oov_prob = max(0.0, 1.0 - float(probs.sum()))

    print(f"Kept top-{K} bigrams covering {probs.sum() * 100:.2f}% of mass")
    print(f"OOV residual: {oov_prob:.4f}")
    print(f"\nTop 10:")
    for i in range(10):
        a, b = int(pairs[i, 0]), int(pairs[i, 1])
        pa = sp.IdToPiece(a) if a < spm_size else f"<{a}>"
        pb = sp.IdToPiece(b) if b < spm_size else f"<{b}>"
        print(f"  ({a:5d}, {b:5d})  p={probs[i]:.5f}  '{pa}' '{pb}'")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        out_path,
        pairs=pairs,                 # (K, 2) int32
        probs=probs.astype(np.float32),  # (K,)
        oov_prob=np.float32(oov_prob),
        decoder_vocab=np.int32(decoder_vocab),
    )
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()

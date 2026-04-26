"""One-time precompute of the training-corpus unigram distribution.

Reads the dep-tree cache, builds a per-token frequency table over the
decoder-vocab range (SPM tokens + BOS + EOS), normalizes to a probability
distribution, saves to a .npy file. The model loads this at training start
and uses it as the target of a KL regularizer on its batch-marginal output
distribution (well-formedness pressure).

Usage:
    python scripts/preprocessing/compute_corpus_unigram.py [--cache PATH] [--out PATH]
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
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
    p.add_argument("--smoothing", type=float, default=1.0,
                   help="add-k smoothing for unseen tokens")
    args = p.parse_args()

    # Sensible defaults
    if Path("/workspace/lfm").exists():
        cfg_default = "/workspace/lfm/configs/dep_tree_vae_vast.yaml"
        cache_default = "/workspace/lfm/data/datasets/english-dep-trees-v16/cache_depth4"
    else:
        cfg_default = "/home/daniel/projects/lfm/configs/dep_tree_vae_vast.yaml"
        cache_default = "/home/daniel/projects/lfm/data/datasets/english-dep-trees-v16/cache_depth4"

    cfg_path = args.config or cfg_default
    cache_dir = Path(args.cache or cache_default)
    out_path = Path(args.out or (cache_dir / "unigram.npy"))

    cfg_dict = yaml.safe_load(open(cfg_path))
    cfg_dict.pop("model_type", None)
    if not Path(cfg_dict.get("spm_model_path", "")).exists():
        # Fall back to local default
        cfg_dict["spm_model_path"] = (
            "/workspace/lfm/data/models/v15b_ipa/spm.model"
            if Path("/workspace/lfm").exists()
            else "/home/daniel/projects/lfm/data/models/v15b_ipa/spm.model"
        )
    cfg_dict["dataset_path"] = str(cache_dir.parent)
    cfg_dict["output_dir"] = "/tmp/dummy"
    cfg = DepTreeVAEConfig(**cfg_dict)

    sp = spm.SentencePieceProcessor(model_file=cfg.spm_model_path)
    spm_size = sp.get_piece_size()
    decoder_vocab = cfg.spm_vocab_size + 2  # SPM + BOS + EOS

    print(f"Reading cache from {cache_dir}")
    interleaved = np.load(cache_dir / "interleaved.npy", mmap_mode="r")
    print(f"  {len(interleaved):,} tokens total")

    print("Counting...")
    counts = np.zeros(decoder_vocab, dtype=np.int64)
    chunk = 10_000_000
    for start in range(0, len(interleaved), chunk):
        end = min(start + chunk, len(interleaved))
        block = np.asarray(interleaved[start:end])
        # Only count tokens in the decoder vocab range
        mask = (block >= 0) & (block < decoder_vocab)
        valid = block[mask]
        if len(valid):
            block_counts = np.bincount(valid, minlength=decoder_vocab)
            counts += block_counts
        print(f"  ...{end:,}/{len(interleaved):,}")

    # Add-k smoothing then normalize
    counts_smoothed = counts.astype(np.float64) + args.smoothing
    probs = counts_smoothed / counts_smoothed.sum()
    print(f"Top 10 most common token IDs:")
    top = np.argsort(probs)[::-1][:10]
    for tid in top:
        try:
            piece = sp.IdToPiece(int(tid)) if tid < spm_size else f"<special:{int(tid)}>"
        except Exception:
            piece = "<?>"
        print(f"  {int(tid):5d}  p={probs[int(tid)]:.5f}  '{piece}'")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(out_path, probs.astype(np.float32))
    print(f"\nSaved unigram (shape {probs.shape}) to {out_path}")


if __name__ == "__main__":
    main()

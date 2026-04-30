"""Precompute top-K alien n-gram statistics over the cipher-encoded corpus.

Saves a .npz with the structure expected by lfm.agents.games.base.NgramKLLoss:
    ngrams:   (K, n) int32   — top-K (a_0, a_1, ..., a_{n-1}) n-gram tuples
    probs:    (K,)   float32 — normalized probability mass
    oov_prob: scalar float32 — residual mass not in top-K

The contrastive game's ngram_kl losses penalise divergence between the
model's generated n-gram marginals and these corpus reference distributions,
keeping the alien LM's output in the natural-corpus regime at multiple
sequential orders.

Usage:
    poetry run python scripts/build_synth_ngram.py CONFIG_PATH \\
        --n 3 [--n-samples 200000] [--top-k 50000] [--out PATH]

n=2 reproduces the old build_synth_bigram.py output. n=3 (trigram) and
n=4 (4-gram) constrain deeper sequential structure than bigrams alone.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
import yaml
from transformers import PreTrainedTokenizerFast

from lfm.synth.cipher import WordCipher
from lfm.synth.config import SynthConfig
from lfm.synth.vocab import AlienVocab


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("config")
    p.add_argument("--n", type=int, required=True, help="n-gram order (2, 3, 4, ...)")
    p.add_argument("--n-samples", type=int, default=200_000)
    p.add_argument("--top-k", type=int, default=50_000)
    p.add_argument("--smoothing", type=float, default=1.0)
    p.add_argument("--max-len", type=int, default=80)
    p.add_argument("--out", default=None)
    args = p.parse_args()

    if args.n < 2:
        raise ValueError("n must be >= 2")

    cfg = SynthConfig(**yaml.safe_load(Path(args.config).read_text()))
    out_dir = Path(cfg.output_dir)
    vocab = AlienVocab.load(out_dir)
    cipher = WordCipher.from_dirs(vocab, out_dir)
    alien_tok = PreTrainedTokenizerFast.from_pretrained(str(out_dir / "alien_tokenizer"))
    V = len(alien_tok)
    out_path = Path(args.out) if args.out else (out_dir / f"{args.n}gram_topk.npz")

    dataset_path = Path(cfg.phase1_dataset_dir)
    if dataset_path.suffix == ".jsonl":
        all_lines = [json.loads(l)["text"]
                     for l in dataset_path.read_text().splitlines() if l.strip()]
    else:
        all_lines = [l.strip() for l in dataset_path.read_text().splitlines() if l.strip()]
    sample = all_lines[: args.n_samples]
    print(f"counting {args.n}-grams over {len(sample):,} sentences (V={V})")

    # We need n-tuple key encoding. Use Python tuples directly (V can be up
    # to ~32K; V**n exceeds int64 at n>=4 for V=32K, so we don't pack).
    counts: Counter = Counter()
    chunk = 1024
    n = args.n
    for i in range(0, len(sample), chunk):
        cipher_texts = cipher.encode_batch(sample[i : i + chunk])
        encs = alien_tok(
            cipher_texts, padding=False, truncation=True,
            max_length=args.max_len + 1, add_special_tokens=True,
        )["input_ids"]
        for ids in encs:
            arr = np.asarray(ids, dtype=np.int64)
            if arr.size < n:
                continue
            # Construct sliding n-tuples
            cols = [arr[j : arr.size - (n - 1) + j] for j in range(n)]
            stacked = np.stack(cols, axis=1)  # (T-n+1, n)
            for row in stacked:
                counts[tuple(row.tolist())] += 1
        if (i // chunk) % 50 == 0:
            print(f"  {min(i + chunk, len(sample)):,} / {len(sample):,}  unique={len(counts):,}")

    print(f"\ntotal unique {n}-grams: {len(counts):,}")
    keys = list(counts.keys())
    cnts = np.array([counts[k] for k in keys], dtype=np.int64)
    total = int(cnts.sum())

    K = min(args.top_k, len(counts))
    top_idx = np.argpartition(cnts, -K)[-K:]
    top_idx = top_idx[np.argsort(-cnts[top_idx])]
    top_keys = [keys[i] for i in top_idx]
    top_cnts = cnts[top_idx].astype(np.float64)

    grams = np.array(top_keys, dtype=np.int32)            # (K, n)
    smoothed = top_cnts + args.smoothing
    probs = smoothed / total
    oov_prob = max(0.0, 1.0 - float(probs.sum()))

    print(f"top-{K} covers {probs.sum() * 100:.2f}% of mass; OOV residual {oov_prob:.4f}")
    print(f"top 10 {n}-grams (token-id only, no surface):")
    for i in range(min(10, K)):
        ids = grams[i].tolist()
        print(f"  ({', '.join(str(x) for x in ids)})  p={probs[i]:.5f}")

    np.savez(
        out_path,
        ngrams=grams,
        probs=probs.astype(np.float32),
        oov_prob=np.float32(oov_prob),
    )
    print(f"\nsaved {out_path}")


if __name__ == "__main__":
    main()

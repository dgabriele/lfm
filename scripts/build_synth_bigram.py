"""Precompute top-K alien bigram statistics over the cipher-encoded corpus.

Saves a .npz with the structure expected by lfm.agents.games.base.BigramKLLoss:
    pairs:    (K, 2) int32 — top-K (a, b) bigram token-id pairs
    probs:    (K,)  float32 — normalized probability mass
    oov_prob: scalar float32 — residual mass not in top-K

The contrastive game's bigram_kl loss penalises divergence between the model's
generated bigram marginals and this corpus reference distribution. Used to
keep the alien LM's output distribution in the natural corpus regime.

Usage:
    poetry run python scripts/build_synth_bigram.py CONFIG_PATH \\
        [--n-samples 200000] [--top-k 50000] [--out PATH]
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
    p.add_argument("--n-samples", type=int, default=200_000)
    p.add_argument("--top-k", type=int, default=50_000)
    p.add_argument("--smoothing", type=float, default=1.0)
    p.add_argument("--max-len", type=int, default=80)
    p.add_argument("--out", default=None)
    args = p.parse_args()

    cfg = SynthConfig(**yaml.safe_load(Path(args.config).read_text()))
    out_dir = Path(cfg.output_dir)
    vocab = AlienVocab.load(out_dir)
    cipher = WordCipher.from_dirs(vocab, out_dir)
    alien_tok = PreTrainedTokenizerFast.from_pretrained(str(out_dir / "alien_tokenizer"))
    V = len(alien_tok)
    out_path = Path(args.out) if args.out else (out_dir / "bigram_topk.npz")

    dataset_path = Path(cfg.phase1_dataset_dir)
    if dataset_path.suffix == ".jsonl":
        all_lines = [json.loads(l)["text"]
                     for l in dataset_path.read_text().splitlines() if l.strip()]
    else:
        all_lines = [l.strip() for l in dataset_path.read_text().splitlines() if l.strip()]
    sample = all_lines[: args.n_samples]
    print(f"counting bigrams over {len(sample):,} sentences (V={V})")

    counts: Counter = Counter()
    chunk = 1024
    for i in range(0, len(sample), chunk):
        cipher_texts = cipher.encode_batch(sample[i : i + chunk])
        encs = alien_tok(cipher_texts, padding=False, truncation=True,
                        max_length=args.max_len + 1, add_special_tokens=True)["input_ids"]
        for ids in encs:
            arr = np.asarray(ids, dtype=np.int64)
            if arr.size < 2:
                continue
            keys = arr[:-1] * V + arr[1:]
            uniq, cts = np.unique(keys, return_counts=True)
            for k, c in zip(uniq.tolist(), cts.tolist()):
                counts[k] += c
        if (i // chunk) % 50 == 0:
            print(f"  {min(i + chunk, len(sample)):,} / {len(sample):,}  unique={len(counts):,}")

    print(f"\ntotal unique bigrams: {len(counts):,}")
    keys = np.fromiter(counts.keys(), dtype=np.int64, count=len(counts))
    cnts = np.fromiter(counts.values(), dtype=np.int64, count=len(counts))
    total = int(cnts.sum())

    K = min(args.top_k, len(counts))
    top_idx = np.argpartition(cnts, -K)[-K:]
    top_idx = top_idx[np.argsort(-cnts[top_idx])]
    top_keys = keys[top_idx]
    top_cnts = cnts[top_idx].astype(np.float64)

    pairs = np.stack([top_keys // V, top_keys % V], axis=1).astype(np.int32)
    smoothed = top_cnts + args.smoothing
    probs = smoothed / total
    oov_prob = max(0.0, 1.0 - float(probs.sum()))

    print(f"top-{K} covers {probs.sum() * 100:.2f}% of mass; OOV residual {oov_prob:.4f}")
    print("top 10 bigrams:")
    for i in range(10):
        a, b = int(pairs[i, 0]), int(pairs[i, 1])
        sa = alien_tok.convert_ids_to_tokens([a])[0]
        sb = alien_tok.convert_ids_to_tokens([b])[0]
        print(f"  ({a:5d}, {b:5d})  p={probs[i]:.5f}  {sa!r} {sb!r}")

    np.savez(
        out_path,
        pairs=pairs,
        probs=probs.astype(np.float32),
        oov_prob=np.float32(oov_prob),
    )
    print(f"\nsaved {out_path}")


if __name__ == "__main__":
    main()

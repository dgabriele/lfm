"""Convert filtered English sentences to IPA.

Strips punctuation, converts each word via CMU dict (falling back to
epitran), and writes the IPA corpus. Multiprocessed.

Usage:
    poetry run python scripts/convert_to_ipa.py \
        --input data/datasets/english-sentences-v15/sentences.txt \
        --output data/datasets/english-sentences-v15/ipa_sentences.txt \
        --workers 16
"""

from __future__ import annotations

import argparse
import multiprocessing as mp
import re
from pathlib import Path


_POST_REJECT = re.compile(
    r'http|www\.|\.com|\.org|\.net|\.edu|\.gov|\.io'
    r'|\.php|\.html|\.pdf|\.jpg|\.png'
    r'|\d'
    r'|["""]',
    re.IGNORECASE,
)


def _strip_punct(s: str) -> str:
    """Remove all punctuation, normalize whitespace."""
    # Keep only letters, spaces, and hyphens (for compound words)
    s = re.sub(r"[^a-zA-Z\s-]", " ", s)
    # Collapse whitespace
    s = re.sub(r"\s+", " ", s).strip()
    return s.lower()


def _convert_chunk(args: tuple[list[str], int]) -> tuple[list[tuple[str, str]], int]:
    """Worker: convert a chunk of sentences to IPA."""
    sentences, chunk_id = args
    from lfm.data.loaders.ipa import IPAConverter

    conv = IPAConverter(drop_unconvertible=False)
    results: list[tuple[str, str]] = []
    failed = 0

    for s in sentences:
        # Reject anything that leaked through earlier filters
        if _POST_REJECT.search(s):
            failed += 1
            continue
        cleaned = _strip_punct(s)
        if not cleaned or len(cleaned.split()) < 8:
            failed += 1
            continue
        try:
            ipa = conv.convert_line("eng", cleaned)
            if ipa and len(ipa.strip()) >= 10:
                results.append((cleaned, ipa.strip()))
            else:
                failed += 1
        except Exception:
            failed += 1

    return results, failed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--max-lines", type=int, default=0,
                    help="Process at most this many lines (0=all)")
    args = ap.parse_args()

    print(f"Reading {args.input}...")
    lines = [l.strip() for l in args.input.open() if l.strip()]
    if args.max_lines:
        lines = lines[:args.max_lines]
    print(f"  {len(lines):,} sentences")

    # Split into chunks for workers
    n = args.workers
    chunk_size = (len(lines) + n - 1) // n
    chunks = [(lines[i:i + chunk_size], idx) for idx, i in enumerate(range(0, len(lines), chunk_size))]

    print(f"Converting to IPA with {n} workers...")
    with mp.Pool(n) as pool:
        results = pool.map(_convert_chunk, chunks)

    all_pairs: list[tuple[str, str]] = []
    total_failed = 0
    for chunk_pairs, chunk_failed in results:
        all_pairs.extend(chunk_pairs)
        total_failed += chunk_failed

    print(f"  Converted: {len(all_pairs):,}")
    print(f"  Failed: {total_failed:,}")

    args.output.parent.mkdir(parents=True, exist_ok=True)

    # Write IPA-only corpus (for VAE training)
    with open(args.output, "w") as f:
        for _, ipa in all_pairs:
            f.write(ipa + "\n")

    # Write paired corpus (English + IPA, tab-separated, for reference)
    paired_path = args.output.with_suffix(".paired.txt")
    with open(paired_path, "w") as f:
        for eng, ipa in all_pairs:
            f.write(f"{eng}\t{ipa}\n")

    print(f"  IPA corpus: {args.output}")
    print(f"  Paired corpus: {paired_path}")

    # Stats
    import numpy as np
    ipa_lengths = [len(ipa.split()) for _, ipa in all_pairs]
    la = np.array(ipa_lengths)
    print(f"\n  IPA length (words): min={la.min()} max={la.max()} "
          f"mean={la.mean():.1f} p50={int(np.median(la))} p90={int(np.percentile(la, 90))}")

    # Sample
    import random
    random.seed(42)
    sample = random.sample(all_pairs, min(20, len(all_pairs)))
    print(f"\n{'='*70}")
    print(f"SAMPLE")
    print(f"{'='*70}\n")
    for eng, ipa in sample:
        print(f"  ENG: {eng}")
        print(f"  IPA: {ipa}")
        print()


if __name__ == "__main__":
    main()

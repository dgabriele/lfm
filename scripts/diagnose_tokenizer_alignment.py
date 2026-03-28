#!/usr/bin/env python3
"""Diagnose BPE tokenizer alignment with syllable boundaries.

Measures how well the trained sentencepiece BPE model's token boundaries
align with phonologically motivated syllable boundaries (Sonority
Sequencing Principle).  High alignment means the tokenizer preserves
phonotactic structure; low alignment means tokens cross syllable
boundaries, potentially degrading the decoder's ability to learn
phonotactic constraints.

Usage::

    python scripts/diagnose_tokenizer_alignment.py \
        --spm-path data/models/v4-phase1/spm.model \
        --dataset-path data/datasets/leipzig \
        --max-samples 5000
"""

from __future__ import annotations

import argparse
import logging
import sys
from collections import Counter

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(message)s",
    stream=sys.stderr,
)
logger = logging.getLogger(__name__)


def main(
    spm_path: str,
    dataset_path: str,
    max_samples: int = 5000,
    tolerance: int = 1,
) -> None:
    import sentencepiece as spm

    from lfm.data.dataset.reader import DatasetReader
    from lfm.data.syllabify import (
        sonority_alignment_score,
        syllable_boundaries,
        syllabify_ipa,
    )

    # Load SPM model
    sp = spm.SentencePieceProcessor()
    sp.load(spm_path)
    logger.info("Loaded SPM: %s (vocab=%d)", spm_path, sp.get_piece_size())

    # Load IPA samples
    reader = DatasetReader(dataset_path)
    tuples = reader.load_ipa_tuples(max_samples_per_language=max_samples // 16)
    logger.info("Loaded %d IPA samples", len(tuples))

    # Analyze alignment
    alignment_scores: list[float] = []
    tokens_per_syllable: list[float] = []
    cross_boundary_tokens = 0
    total_tokens = 0
    syllable_length_dist: Counter[int] = Counter()
    token_length_dist: Counter[int] = Counter()

    for _, ipa in tuples[:max_samples]:
        # Syllabify
        syllables = syllabify_ipa(ipa)
        syl_bounds = syllable_boundaries(ipa)
        n_syllables = len([s for s in syllables if s.strip()])

        # Tokenize
        pieces = sp.encode(ipa, out_type=str)
        # Compute token boundaries in character space
        tok_bounds: list[int] = [0]
        pos = 0
        for piece in pieces:
            # Sentencepiece uses ▁ for word boundaries
            raw = piece.replace("▁", " ").lstrip()
            pos += len(raw) if raw else len(piece)
            tok_bounds.append(pos)

        # Score alignment
        # Skip first (0) and last boundary — they always align
        inner_tok_bounds = tok_bounds[1:-1]
        inner_syl_bounds = syl_bounds[1:-1]

        if inner_tok_bounds:
            score = sonority_alignment_score(
                inner_tok_bounds, syl_bounds, tolerance=tolerance,
            )
            alignment_scores.append(score)

        total_tokens += len(pieces)
        if n_syllables > 0:
            tokens_per_syllable.append(len(pieces) / n_syllables)

        for s in syllables:
            if s.strip():
                syllable_length_dist[len(s)] += 1
        for p in pieces:
            token_length_dist[len(p)] += 1

    # Report
    import numpy as np

    scores = np.array(alignment_scores)
    tps = np.array(tokens_per_syllable)

    print("\n" + "=" * 60)
    print("TOKENIZER-SYLLABLE ALIGNMENT DIAGNOSTIC")
    print("=" * 60)
    print(f"SPM model:      {spm_path}")
    print(f"Dataset:        {dataset_path}")
    print(f"Samples:        {len(tuples[:max_samples])}")
    print(f"Tolerance:      ±{tolerance} chars")
    print()
    print(f"Alignment score (mean): {scores.mean():.3f}")
    print(f"Alignment score (med):  {np.median(scores):.3f}")
    print(f"Alignment score (std):  {scores.std():.3f}")
    print(f"Alignment score (min):  {scores.min():.3f}")
    print()
    print(f"Tokens per syllable (mean): {tps.mean():.2f}")
    print(f"Tokens per syllable (med):  {np.median(tps):.2f}")
    print()

    # Interpretation
    mean_score = scores.mean()
    if mean_score >= 0.8:
        verdict = "GOOD — BPE tokens mostly respect syllable boundaries"
    elif mean_score >= 0.6:
        verdict = "MODERATE — some syllable-crossing, consider constrained BPE"
    else:
        verdict = "POOR — significant syllable-crossing, recommend constrained BPE"
    print(f"Verdict: {verdict}")
    print()

    # Syllable length distribution
    print("Syllable length distribution (chars):")
    for length in sorted(syllable_length_dist.keys())[:10]:
        count = syllable_length_dist[length]
        bar = "█" * min(count // 100, 40)
        print(f"  {length:2d}: {count:6d} {bar}")

    print()
    print("Token length distribution (chars):")
    for length in sorted(token_length_dist.keys())[:10]:
        count = token_length_dist[length]
        bar = "█" * min(count // 100, 40)
        print(f"  {length:2d}: {count:6d} {bar}")

    # Show some examples
    print()
    print("Sample syllabifications:")
    for _, ipa in tuples[:8]:
        syllables = syllabify_ipa(ipa)
        display = "·".join(s for s in syllables if s.strip())
        if len(display) > 80:
            display = display[:80] + "..."
        print(f"  {display}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Diagnose BPE tokenizer alignment with syllable boundaries",
    )
    parser.add_argument("--spm-path", required=True, help="Path to spm.model")
    parser.add_argument("--dataset-path", required=True, help="Path to dataset dir")
    parser.add_argument("--max-samples", type=int, default=5000)
    parser.add_argument("--tolerance", type=int, default=1)
    args = parser.parse_args()
    main(
        spm_path=args.spm_path,
        dataset_path=args.dataset_path,
        max_samples=args.max_samples,
        tolerance=args.tolerance,
    )

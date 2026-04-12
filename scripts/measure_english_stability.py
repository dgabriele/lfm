#!/usr/bin/env python
"""Measure BPE concat-stability for common English single-token words.

We want a principled lower bound for our Neuroglot alphabet: phonemes
shouldn't need to be MORE stable under concatenation than English words
themselves are.  If "the" tokenizes consistently only 90% of the time
when we jam it against random other tokens, we shouldn't demand 0.95
from our phonemes.

Method:
  1. Take ~100 common English single-token words (curated list).
  2. For each, test left/right stability against a diverse partner set:
     - English partners ("the", "of", ...)
     - A small sample of other-language partners (same multilingual mix
       used in Neuroglot alphabet design)
  3. Report distribution — use median / 25th percentile as the natural
     lower bound for our alphabet's stability threshold.
"""

from __future__ import annotations

from collections import Counter

from transformers import AutoTokenizer

MODEL_NAME = "Qwen/Qwen2.5-0.5B"


# Curated list of ~100 high-frequency English words that Qwen tokenizes
# to exactly one (space-prefixed) token.  Mix of function words and
# content words across different POS.
ENGLISH_WORDS = [
    "the", "and", "for", "but", "not", "was", "are", "with", "this", "that",
    "have", "from", "they", "will", "been", "said", "what", "when", "there",
    "their", "would", "make", "like", "time", "just", "know", "take", "into",
    "year", "your", "some", "them", "other", "than", "then", "look", "only",
    "come", "work", "life", "over", "think", "also", "back", "after", "first",
    "well", "want", "because", "these", "give", "most", "very", "world", "day",
    "still", "between", "never", "last", "good", "much", "before", "being",
    "while", "place", "right", "down", "where", "mean", "away", "through",
    "even", "both", "those", "such", "each", "great", "little", "long",
    "high", "different", "same", "important", "many", "new", "big", "old",
    "small", "part", "water", "people", "school", "house", "country", "word",
    "another", "group", "where", "again", "around",
]


def measure_stability(tok, word: str, partners: list[str]) -> tuple[float, float]:
    ids_self = tok.encode(" " + word, add_special_tokens=False)
    if len(ids_self) != 1:
        return -1.0, -1.0  # word itself not a single token — skip
    self_id = ids_self[0]
    partner_ids = [tok.encode(" " + b, add_special_tokens=False) for b in partners]
    partners_clean = [
        (b, ids[0]) for b, ids in zip(partners, partner_ids) if len(ids) == 1
    ]
    n = max(len(partners_clean), 1)
    ls = rs = 0
    for b, b_id in partners_clean:
        ids = tok.encode(" " + word + b, add_special_tokens=False)
        if len(ids) >= 2 and ids[0] == self_id:
            ls += 1
        ids2 = tok.encode(" " + b + word, add_special_tokens=False)
        if len(ids2) == 2 and ids2[0] == b_id:
            if tok.convert_ids_to_tokens(ids2)[1] == word:
                rs += 1
    return ls / n, rs / n


def main() -> None:
    tok = AutoTokenizer.from_pretrained(MODEL_NAME)

    # Partner set: English + small multilingual sample (mirrors our
    # Neuroglot alphabet stability test partner distribution).
    partners = (
        ENGLISH_WORDS[:60]  # English partners
        + ["ith", "opp", "jak", "uz", "bir", "ini", "ogr", "ell", "ky", "wy"]
    )

    measurements = []
    for w in ENGLISH_WORDS:
        ls, rs = measure_stability(tok, w, partners)
        if ls < 0:
            continue
        measurements.append((w, ls, rs, ls * rs))

    measurements.sort(key=lambda x: -x[3])
    print(f"Stability of {len(measurements)} common English single-token words:")
    print(f"{'word':>12}  {'left':>5}  {'right':>5}  {'prod':>5}")
    for w, ls, rs, p in measurements:
        print(f"{w!r:>12}  {ls:.2f}  {rs:.2f}   {p:.2f}")

    import statistics
    left_scores = [m[1] for m in measurements]
    right_scores = [m[2] for m in measurements]
    prod_scores = [m[3] for m in measurements]
    print("\nDistribution:")
    print(f"  left  stability: median={statistics.median(left_scores):.3f}  "
          f"mean={statistics.mean(left_scores):.3f}  "
          f"min={min(left_scores):.3f}  max={max(left_scores):.3f}")
    print(f"  right stability: median={statistics.median(right_scores):.3f}  "
          f"mean={statistics.mean(right_scores):.3f}  "
          f"min={min(right_scores):.3f}  max={max(right_scores):.3f}")
    print(f"  product:         median={statistics.median(prod_scores):.3f}  "
          f"mean={statistics.mean(prod_scores):.3f}  "
          f"min={min(prod_scores):.3f}  max={max(prod_scores):.3f}")

    # Percentiles
    sorted_prod = sorted(prod_scores)
    n = len(sorted_prod)
    p25 = sorted_prod[n // 4]
    p10 = sorted_prod[n // 10]
    p05 = sorted_prod[n // 20]
    print(f"\nProduct percentiles:")
    print(f"  5th  percentile: {p05:.3f}")
    print(f"  10th percentile: {p10:.3f}")
    print(f"  25th percentile: {p25:.3f}")
    print(f"\nRecommendation: set Neuroglot stability threshold to ≈ "
          f"{p10:.2f} (10th percentile of English) — this demands no more "
          "stability than real English has.")


if __name__ == "__main__":
    main()

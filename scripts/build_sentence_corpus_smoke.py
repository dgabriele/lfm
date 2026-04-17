"""Smoke-test sentence corpus builder.

Filters raw English sentences for the "generic English" VAE corpus:
  - No quoted speech or dialogue
  - No proper nouns / named entities (heuristic + optional SpaCy)
  - Length within reasonable range
  - No lists, fragments, or overly technical notation

Outputs a small sample for manual inspection and iteration.

Usage:
    poetry run python scripts/build_sentence_corpus_smoke.py \
        --input data/datasets/english-constituents-v13/sentences.txt \
        --output /tmp/corpus_smoke.txt \
        --n 500
"""

from __future__ import annotations

import argparse
import random
import re
import sys
from pathlib import Path


def has_quotes(s: str) -> bool:
    return bool(re.search(r'["""\'\']', s) and re.search(r'(said|asked|told|replied|exclaimed|shouted|whispered|added|noted|claimed|argued|insisted|remarked|declared|announced|explained|stated|wrote|reported)', s.lower()))


def has_direct_quotes(s: str) -> bool:
    return bool(re.search(r'["""\'\']', s) or "``" in s or "''" in s)


_NLP = None
_REJECT_NER_LABELS = {"PERSON", "GPE", "ORG", "NORP", "FAC", "LOC"}


def _get_nlp():
    global _NLP
    if _NLP is None:
        import spacy
        _NLP = spacy.load("en_core_web_sm")
    return _NLP


def has_named_entities(s: str) -> bool:
    """SpaCy-based: reject if sentence has named people, places, orgs."""
    doc = _get_nlp()(s)
    for ent in doc.ents:
        if ent.label_ in _REJECT_NER_LABELS:
            return True
    for tok in doc:
        if tok.pos_ == "PROPN":
            return True
    return False


def has_digits(s: str) -> bool:
    """Reject sentences with digit characters (notation, not language)."""
    return bool(re.search(r'\d', s))


def has_special_chars(s: str) -> bool:
    """Filter sentences with URLs, code, math notation, etc."""
    return bool(re.search(r'http|www\.|\.com|\.org|\{|\}|\\[a-z]|<[a-z]|@|\||\[|\]', s.lower()))


def is_list_or_fragment(s: str) -> bool:
    """Filter fragments, headings, list items, truncated sentences."""
    s = s.strip()
    if not s:
        return True
    # Must start with a letter (catches leading ), *, -, etc.)
    if not s[0].isalpha():
        return True
    # Handle both normal and lowercased/spaced-punct corpora
    ends_ok = s.rstrip().endswith((".", "!", "?", '."', '!"', '?"', " .", " !", " ?"))
    if not ends_ok:
        return True
    # Reject sentences ending in abbreviation-like patterns (i.e., e.g., etc.)
    if re.search(r'\b(i\.e|e\.g|etc|et al|viz|cf|vs)\s*\.\s*$', s):
        return True
    # Reject sentences ending in ". )" or similar broken closings
    if re.search(r'[)\]]\s*\.?\s*$', s):
        return True
    # Reject keyword lists / heading-style (no verb — proxy: too many commas
    # relative to length, or starts with gerund + no subject)
    words = s.split()
    comma_ratio = s.count(",") / max(len(words), 1)
    if comma_ratio > 0.25 and len(words) < 15:
        return True
    # Reject lines that look like section headings or labels
    if s.rstrip(" .").count(" ") < 3 and len(s) < 30:
        return True
    return False


def length_ok(s: str, min_words: int = 8, max_words: int = 35) -> bool:
    return min_words <= len(s.split()) <= max_words


def passes_all(s: str) -> bool:
    s = s.strip()
    if not s:
        return False
    if is_list_or_fragment(s):
        return False
    if not length_ok(s):
        return False
    if has_direct_quotes(s):
        return False
    if has_proper_nouns(s):
        return False
    if has_numbers_heavy(s):
        return False
    if has_special_chars(s):
        return False
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, type=Path)
    ap.add_argument("--output", default="/tmp/corpus_smoke.txt", type=Path)
    ap.add_argument("--n", type=int, default=500, help="Sample size for inspection")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--show-rejects", action="store_true",
                    help="Also write rejected sentences with rejection reason")
    args = ap.parse_args()

    print(f"Reading {args.input}...")
    lines = args.input.read_text().splitlines()
    print(f"  {len(lines):,} raw sentences")

    # Pass 1: filter
    passed = []
    reject_reasons: dict[str, int] = {
        "fragment/no_period": 0,
        "too_short": 0,
        "too_long": 0,
        "has_quotes": 0,
        "proper_nouns": 0,
        "has_digits": 0,
        "special_chars": 0,
        "named_entities": 0,
    }

    # Pre-filter fast heuristics on all lines
    pre_passed: list[str] = []
    for s in lines:
        s = s.strip()
        if not s:
            continue
        if is_list_or_fragment(s):
            reject_reasons["fragment/no_period"] += 1
            continue
        if not length_ok(s):
            if len(s.split()) < 8:
                reject_reasons["too_short"] += 1
            else:
                reject_reasons["too_long"] += 1
            continue
        if has_direct_quotes(s):
            reject_reasons["has_quotes"] += 1
            continue
        if has_digits(s):
            reject_reasons["has_digits"] += 1
            continue
        if has_special_chars(s):
            reject_reasons["special_chars"] += 1
            continue
        pre_passed.append(s)

    print(f"  {len(pre_passed):,} passed heuristic filters")

    # Subsample for SpaCy NER (expensive)
    ner_candidates = pre_passed
    if len(pre_passed) > 50000:
        rng_sub = random.Random(args.seed)
        ner_candidates = rng_sub.sample(pre_passed, 50000)
        print(f"  subsampled {len(ner_candidates):,} for NER filtering")

    print("  running SpaCy NER filter...")
    for s in ner_candidates:
        if has_named_entities(s):
            reject_reasons["named_entities"] += 1
            continue
        passed.append(s)

    print(f"\n  {len(passed):,} passed ({100*len(passed)/len(lines):.1f}%)")
    print(f"\n  rejection breakdown:")
    for reason, count in sorted(reject_reasons.items(), key=lambda kv: -kv[1]):
        print(f"    {reason:<25} {count:>8,} ({100*count/len(lines):.1f}%)")

    # Length stats on passed
    lengths = [len(s.split()) for s in passed]
    import numpy as np
    la = np.array(lengths)
    print(f"\n  passed length stats (words):")
    print(f"    min={la.min()} max={la.max()} mean={la.mean():.1f} "
          f"std={la.std():.1f} p50={int(np.median(la))} p90={int(np.percentile(la, 90))}")

    # Sample
    rng = random.Random(args.seed)
    sample = rng.sample(passed, min(args.n, len(passed)))

    args.output.write_text("\n".join(sample) + "\n")
    print(f"\n  wrote {len(sample)} samples to {args.output}")

    # Print sample to stdout for immediate inspection
    print(f"\n{'='*70}")
    print(f"SAMPLE ({len(sample)} sentences)")
    print(f"{'='*70}\n")
    for i, s in enumerate(sample[:100]):
        print(f"  {i+1:>3}. {s}")

    if args.show_rejects:
        print(f"\n{'='*70}")
        print("REJECTED EXAMPLES (first 50 per reason)")
        print(f"{'='*70}\n")
        # Re-scan for examples
        for reason_name, check_fn, label in [
            ("proper_nouns", has_proper_nouns, "PROPER NOUN"),
            ("has_quotes", has_direct_quotes, "QUOTES"),
            ("numbers_heavy", has_numbers_heavy, "NUMBERS"),
        ]:
            count = 0
            print(f"\n  --- {label} ---")
            for s in lines:
                s = s.strip()
                if not s or is_list_or_fragment(s) or not length_ok(s):
                    continue
                if check_fn(s):
                    print(f"    {s[:120]}")
                    count += 1
                    if count >= 10:
                        break


if __name__ == "__main__":
    main()

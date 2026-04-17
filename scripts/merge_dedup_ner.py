"""Merge, dedup, and NER-filter the sentence corpus.

Pipeline: merge all sources → exact dedup → SpaCy NER → output.

Usage:
    poetry run python scripts/merge_dedup_ner.py \
        --inputs data/datasets/c4_english_sentences_clean.txt \
                 data/datasets/english-constituents-v13/sentences.txt \
        --leipzig data/leipzig/eng_news_2020_1M \
                  data/leipzig/eng_news_2023_1M \
                  data/leipzig/eng_wikipedia_2016_1M \
                  data/leipzig/eng-simple_wikipedia_2021_100K \
        --output data/datasets/english-sentences-v15/sentences.txt \
        --target 10000000
"""

from __future__ import annotations

import argparse
import hashlib
import re
import sys
from pathlib import Path


def has_digits(s: str) -> bool:
    return bool(re.search(r'\d', s))


def has_direct_quotes(s: str) -> bool:
    return bool(re.search(r'["""\'\']', s) or "``" in s or "''" in s)


def has_special_chars(s: str) -> bool:
    return bool(re.search(
        r'http|www\.|\.com|\.org|\.net|\.edu|\.gov|\.io'
        r'|\{|\}|\\[a-z]|<[a-z]|@|\||\[|\]'
        r'|\.php|\.html|\.pdf|\.jpg|\.png',
        s.lower(),
    ))


def is_fragment(s: str) -> bool:
    s = s.strip()
    if not s or not s[0].isalpha():
        return True
    if not s.rstrip().endswith((".", "!", "?")):
        return True
    if re.search(r'\b(i\.e|e\.g|etc|et al|viz|cf|vs)\s*\.\s*$', s):
        return True
    if re.search(r'[)\]]\s*\.?\s*$', s):
        return True
    words = s.split()
    if len(words) < 3:
        return True
    if s.rstrip(" .").count(" ") < 3 and len(s) < 30:
        return True
    return False


def length_ok(s: str, min_w: int = 8, max_w: int = 35) -> bool:
    return min_w <= len(s.split()) <= max_w


_SPAM_RE = re.compile(
    r'click here|sign up|subscribe|buy now|order now|free shipping'
    r'|add to cart|checkout|coupon|promo code|discount code'
    r'|call us|contact us today|get in touch|request a quote'
    r'|we (offer|provide|deliver|specialize|guarantee)'
    r'|our (team|staff|experts|professionals|company) (will|can|is)'
    r'|your (order|subscription|account|purchase|payment)'
    r'|terms and conditions|privacy policy|cookie policy'
    r'|all rights reserved|copyright|permission',
    re.IGNORECASE,
)
_MARKET_RE = re.compile(
    r"don't miss|limited time|act now|hurry|exclusive offer"
    r'|best (deal|price|value|offer)|lowest price'
    r'|satisfaction guaranteed|money back|risk.free'
    r'|award.winning|world.class|top.rated|best.in.class'
    r'|trusted by|as seen on|featured in',
    re.IGNORECASE,
)


def is_spam(s: str) -> bool:
    if _SPAM_RE.search(s):
        return True
    if _MARKET_RE.search(s):
        return True
    if re.search(r'\byou(r)?\b', s, re.IGNORECASE):
        hits = re.findall(
            r'\b(buy|purchase|order|subscribe|visit|download|install'
            r'|website|site|shop|store|product|service|price|cost'
            r'|shipping|delivery|customer|client)\b',
            s, re.IGNORECASE,
        )
        if len(hits) >= 2:
            return True
    return False


def passes_heuristic(s: str) -> bool:
    s = s.strip()
    if not s:
        return False
    if is_fragment(s):
        return False
    if not length_ok(s):
        return False
    if has_digits(s):
        return False
    if has_direct_quotes(s):
        return False
    if has_special_chars(s):
        return False
    if is_spam(s):
        return False
    return True


_NER_REJECT = {"PERSON", "GPE", "ORG", "NORP", "FAC", "LOC"}


def _ner_filter_chunk(sentences: list[str]) -> tuple[list[str], int]:
    """Worker function: load SpaCy per-process, filter a chunk."""
    import spacy
    nlp = spacy.load("en_core_web_sm")
    nlp.select_pipes(enable=["ner", "tagger"])

    passed: list[str] = []
    rejected = 0

    for doc in nlp.pipe(sentences, batch_size=1000):
        has_named = False
        for ent in doc.ents:
            if ent.label_ in _NER_REJECT:
                has_named = True
                break
        if not has_named:
            for tok in doc:
                if tok.pos_ == "PROPN":
                    has_named = True
                    break
        if has_named:
            rejected += 1
        else:
            passed.append(doc.text)

    return passed, rejected


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", nargs="+", type=Path, default=[],
                    help="Plain text files (one sentence per line)")
    ap.add_argument("--leipzig", nargs="+", type=Path, default=[],
                    help="Leipzig corpus directories (contain *-sentences.txt)")
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--target", type=int, default=10_000_000)
    ap.add_argument("--ner-batch", type=int, default=1000,
                    help="SpaCy pipe batch size")
    ap.add_argument("--workers", type=int, default=16,
                    help="Number of parallel workers for NER filtering")
    args = ap.parse_args()

    args.output.parent.mkdir(parents=True, exist_ok=True)

    # --- Stage 1: Load + heuristic filter ---
    print("=== Stage 1: Load + heuristic filter ===")
    raw_count = 0
    heuristic_passed: list[str] = []

    for path in args.inputs:
        print(f"  loading {path}...")
        with open(path) as f:
            for line in f:
                raw_count += 1
                s = line.strip()
                if passes_heuristic(s):
                    heuristic_passed.append(s)
        print(f"    {raw_count:,} raw → {len(heuristic_passed):,} passed so far")

    for ldir in args.leipzig:
        sent_files = list(ldir.glob("*-sentences.txt"))
        for sf in sent_files:
            print(f"  loading {sf}...")
            with open(sf) as f:
                for line in f:
                    raw_count += 1
                    parts = line.strip().split("\t", 1)
                    s = parts[1] if len(parts) > 1 else parts[0]
                    if passes_heuristic(s):
                        heuristic_passed.append(s)
            print(f"    {raw_count:,} raw → {len(heuristic_passed):,} passed so far")

    print(f"\n  Total: {raw_count:,} raw → {len(heuristic_passed):,} passed heuristic "
          f"({100*len(heuristic_passed)/raw_count:.1f}%)")

    # --- Stage 2: Exact dedup ---
    print("\n=== Stage 2: Exact dedup ===")
    seen: set[str] = set()
    deduped: list[str] = []
    for s in heuristic_passed:
        h = hashlib.md5(s.lower().encode()).hexdigest()
        if h not in seen:
            seen.add(h)
            deduped.append(s)
    dupes = len(heuristic_passed) - len(deduped)
    print(f"  Removed {dupes:,} duplicates ({100*dupes/len(heuristic_passed):.1f}%)")
    print(f"  Remaining: {len(deduped):,}")
    del seen, heuristic_passed

    # --- Stage 3: SpaCy NER filter (multiprocessed) ---
    print(f"\n=== Stage 3: SpaCy NER filter (on {len(deduped):,} sentences) ===")
    import multiprocessing as mp

    n_workers = min(args.workers, mp.cpu_count())
    print(f"  using {n_workers} workers")

    # Split deduped into chunks for workers
    chunk_size = (len(deduped) + n_workers - 1) // n_workers
    chunks = [deduped[i:i + chunk_size] for i in range(0, len(deduped), chunk_size)]

    with mp.Pool(n_workers) as pool:
        results = pool.map(_ner_filter_chunk, chunks)

    passed: list[str] = []
    ner_rejected = 0
    for chunk_passed, chunk_rejected in results:
        passed.extend(chunk_passed)
        ner_rejected += chunk_rejected

    print(f"  NER rejected: {ner_rejected:,}")
    print(f"  Passed NER: {len(passed):,}")

    if len(passed) > args.target:
        passed = passed[:args.target]
        print(f"  Trimmed to target: {args.target:,}")

    print(f"  Final: {len(passed):,} sentences")

    # --- Stage 4: Write ---
    print(f"\n=== Writing to {args.output} ===")
    with open(args.output, "w") as f:
        for s in passed:
            f.write(s + "\n")

    # Length stats
    import numpy as np
    lengths = np.array([len(s.split()) for s in passed])
    print(f"  Length stats (words): min={lengths.min()} max={lengths.max()} "
          f"mean={lengths.mean():.1f} std={lengths.std():.1f} "
          f"p50={int(np.median(lengths))} p90={int(np.percentile(lengths, 90))}")

    # Show sample
    import random
    random.seed(42)
    sample = random.sample(passed, min(30, len(passed)))
    print(f"\n{'='*70}")
    print(f"SAMPLE ({len(sample)} sentences)")
    print(f"{'='*70}\n")
    for i, s in enumerate(sample):
        print(f"  {i+1:>2}. {s}")

    print(f"\nDone: {len(passed):,} sentences written to {args.output}")


if __name__ == "__main__":
    main()

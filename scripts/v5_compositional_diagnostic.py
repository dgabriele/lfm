"""Per-document structural diagnostic for K-sentence compositional corpora.

Key UNMT-relevant signals when the corpus has K sentences per document:
  * within-doc Jaccard token overlap   — same-doc sentences share content
  * across-doc Jaccard token overlap    — random-pair baseline
  * coherence ratio = within / across   — >1 means real topical clustering
  * sentence-position effects           — does sentence k differ from k+1
                                          in length / opening tokens?
  * within-doc TTR vs concat-of-random  — same-doc sentences should reuse
                                          tokens more than random pairs

Output is purely numerical — never echoes any alien token surface.

Usage:
  poetry run python scripts/v5_compositional_diagnostic.py PATH.jsonl
"""

from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path

import numpy as np


def _jaccard(a: set, b: set) -> float:
    if not a and not b:
        return 1.0
    return len(a & b) / max(len(a | b), 1)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("path")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--n-cross-pairs", type=int, default=2000,
                   help="random sentence pairs from different docs to compute "
                        "the across-doc baseline")
    args = p.parse_args()

    random.seed(args.seed)
    rng = np.random.default_rng(args.seed)

    docs: list[list[set]] = []   # for each doc, list of token sets per sentence
    sent_lengths: list[list[int]] = []
    first_token_per_pos: list[list[str]] = []
    n_skipped = 0
    with Path(args.path).open() as fh:
        for line in fh:
            rec = json.loads(line)
            sents = rec.get("sents", [])
            if len(sents) < 2:
                n_skipped += 1
                continue
            tok_sets = []
            lens = []
            firsts = []
            for s in sents:
                toks = s.split()
                tok_sets.append(set(toks))
                lens.append(len(toks))
                firsts.append(toks[0] if toks else "")
            docs.append(tok_sets)
            sent_lengths.append(lens)
            first_token_per_pos.append(firsts)

    n_docs = len(docs)
    K_per_doc = [len(d) for d in docs]
    K_mode = max(set(K_per_doc), key=K_per_doc.count)
    print(f"loaded {n_docs} docs (skipped {n_skipped} with <2 sents)")
    print(f"K distribution: min={min(K_per_doc)}  median={int(np.median(K_per_doc))}  "
          f"max={max(K_per_doc)}  mode={K_mode}")
    print()

    # ── (1) Sentence length per position ───────────────────────────────
    print("[1] sentence length by position (K=mode)")
    by_pos: list[list[int]] = [[] for _ in range(K_mode)]
    for d in sent_lengths:
        for k in range(min(K_mode, len(d))):
            by_pos[k].append(d[k])
    for k, lens in enumerate(by_pos):
        lens_a = np.array(lens)
        print(f"  pos {k}: mean={lens_a.mean():6.1f}  median={int(np.median(lens_a)):3d}  "
              f"std={lens_a.std():5.1f}  n={len(lens_a)}")

    # ── (2) Within-doc Jaccard ─────────────────────────────────────────
    print("\n[2] within-doc pairwise Jaccard (token overlap between sentences of same doc)")
    within = []
    for tok_sets in docs:
        K = len(tok_sets)
        for i in range(K):
            for j in range(i + 1, K):
                within.append(_jaccard(tok_sets[i], tok_sets[j]))
    within = np.array(within)
    print(f"  pairs={len(within)}  mean={within.mean():.4f}  median={np.median(within):.4f}  "
          f"p90={np.quantile(within, 0.90):.4f}  p99={np.quantile(within, 0.99):.4f}")

    # ── (3) Across-doc Jaccard baseline ────────────────────────────────
    print(f"\n[3] across-doc Jaccard ({args.n_cross_pairs} random pairs from DIFFERENT docs)")
    cross = []
    for _ in range(args.n_cross_pairs):
        a = int(rng.integers(0, n_docs))
        b = int(rng.integers(0, n_docs))
        while b == a:
            b = int(rng.integers(0, n_docs))
        ka = int(rng.integers(0, len(docs[a])))
        kb = int(rng.integers(0, len(docs[b])))
        cross.append(_jaccard(docs[a][ka], docs[b][kb]))
    cross = np.array(cross)
    print(f"  pairs={len(cross)}  mean={cross.mean():.4f}  median={np.median(cross):.4f}  "
          f"p90={np.quantile(cross, 0.90):.4f}  p99={np.quantile(cross, 0.99):.4f}")

    # ── (4) Coherence ratio ────────────────────────────────────────────
    ratio = within.mean() / max(cross.mean(), 1e-12)
    print(f"\n[4] coherence ratio  within / across = {ratio:.3f}")
    print(f"    >1 → same-doc sentences share more tokens than random pairs (topical clustering)")
    print(f"    =1 → no within-doc structure — K sentences are functionally independent")
    print(f"    >>1 → high redundancy (near-duplicate sentences); not necessarily good")

    # ── (5) Within-doc TTR vs concat-random TTR ────────────────────────
    print("\n[5] within-doc TTR vs equal-length concat-of-random")
    # Within-doc TTR: type/token over the doc's K sentences
    same_doc_ttrs = []
    for tok_sets in docs:
        all_tokens = []
        for s in tok_sets:
            all_tokens.extend(list(s))   # sets lose freq; for TTR we need the multiset, recompute
        # Actually re-read original sentences for this — we lost freq when we made sets.
    # Recompute properly using the lengths already counted
    # Simpler: rebuild from the input file once more
    same_doc_ttrs = []
    rand_doc_ttrs = []
    all_sents_by_doc: list[list[list[str]]] = []
    with Path(args.path).open() as fh:
        for line in fh:
            rec = json.loads(line)
            all_sents_by_doc.append([s.split() for s in rec.get("sents", [])])
    for sents in all_sents_by_doc:
        if len(sents) < 2:
            continue
        flat = []
        for s in sents:
            flat.extend(s)
        if not flat:
            continue
        same_doc_ttrs.append(len(set(flat)) / len(flat))
    # Random concat: pick one sentence each from K different random docs, concat, compute TTR
    n_rand = len(same_doc_ttrs)
    for _ in range(n_rand):
        K = K_mode
        flat = []
        for k in range(K):
            d = int(rng.integers(0, len(all_sents_by_doc)))
            sents = all_sents_by_doc[d]
            if not sents:
                continue
            ki = int(rng.integers(0, len(sents)))
            flat.extend(sents[ki])
        if flat:
            rand_doc_ttrs.append(len(set(flat)) / len(flat))
    ttr_same = np.array(same_doc_ttrs)
    ttr_rand = np.array(rand_doc_ttrs)
    print(f"  same-doc concat TTR: mean={ttr_same.mean():.4f}  median={np.median(ttr_same):.4f}")
    print(f"  random-doc concat TTR: mean={ttr_rand.mean():.4f}  median={np.median(ttr_rand):.4f}")
    print(f"  TTR ratio same/random = {ttr_same.mean()/max(ttr_rand.mean(), 1e-12):.4f}")
    print(f"    <1 → same-doc sentences reuse more tokens than random sentences (topical recurrence)")
    print(f"    =1 → no within-doc reuse advantage")

    # ── (6) First-token diversity by position ──────────────────────────
    print("\n[6] first-token diversity per sentence position (K=mode)")
    for k in range(K_mode):
        firsts = [d[k] for d in first_token_per_pos if k < len(d) and d[k]]
        unique = len(set(firsts))
        print(f"  pos {k}: {unique:5d} unique starts / {len(firsts):5d} sents = "
              f"{unique/max(len(firsts), 1)*100:5.1f}% unique")

    # ── (7) Cross-position similarity decay ────────────────────────────
    print("\n[7] within-doc Jaccard by sentence-position distance")
    print(f"    (e.g. dist=1 = adjacent sentences; dist=K-1 = first vs last)")
    by_dist: dict[int, list[float]] = {d: [] for d in range(1, K_mode)}
    for tok_sets in docs:
        K = len(tok_sets)
        for i in range(K):
            for j in range(i + 1, K):
                d = j - i
                by_dist.setdefault(d, []).append(_jaccard(tok_sets[i], tok_sets[j]))
    for d in sorted(by_dist):
        arr = np.array(by_dist[d])
        if len(arr) > 0:
            print(f"  dist={d}: mean Jaccard={arr.mean():.4f}  n_pairs={len(arr)}")


if __name__ == "__main__":
    main()

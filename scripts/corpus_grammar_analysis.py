"""Empirical computational-linguistic analysis of the alien corpus.

Outputs metrics only — never echoes alien tokens, so it is safe to log/transmit.

Pipelines:
  1. Distributional laws (Zipf, Heaps)
  2. Sequential structure (n-gram conditional entropy, redundancy ratio)
  3. Long-range mutual information (power-law vs exponential decay)
  4. Surface structure (bracket pairing, sentence-end punctuation, doc lengths)
  5. Burstiness / topical coherence (within-doc reuse vs unigram baseline)
  6. Compression-based structure (gzip vs shuffled baseline)
  7. Distributional class induction (PPMI + SVD + k-means → POS-like classes)
  8. Class-bigram grammar analysis (PCFG-light: rule-count, entropy, structure)

Reference values for natural English (where known) are quoted in comments.
"""

from __future__ import annotations

import argparse
import gzip
import math
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from scipy.sparse import csr_matrix
from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import TruncatedSVD


# ── helpers ─────────────────────────────────────────────────────────────────


def load_corpus(path: Path, max_docs: int | None = None) -> list[list[str]]:
    docs = []
    with path.open() as f:
        for i, line in enumerate(f):
            if max_docs and i >= max_docs:
                break
            tok = line.split()
            if tok:
                docs.append(tok)
    return docs


def loglog_slope(xs: list[float], ys: list[float]) -> tuple[float, float]:
    """OLS slope+intercept of log(ys) vs log(xs). Returns (slope, R^2)."""
    lx = np.log(xs)
    ly = np.log(ys)
    n = len(lx)
    mx, my = lx.mean(), ly.mean()
    sxy = ((lx - mx) * (ly - my)).sum()
    sxx = ((lx - mx) ** 2).sum()
    syy = ((ly - my) ** 2).sum()
    slope = sxy / sxx
    r2 = (sxy ** 2) / (sxx * syy) if syy > 0 else 0.0
    return float(slope), float(r2)


# ── 1. distributional laws ─────────────────────────────────────────────────


def analyze_zipf(counts: Counter) -> dict:
    """Zipf: freq ~ rank^slope; natural English ~ -1.0."""
    sorted_freqs = sorted(counts.values(), reverse=True)
    ranks = list(range(1, len(sorted_freqs) + 1))
    slope_full, r2_full = loglog_slope(ranks, sorted_freqs)
    top_k = min(1000, len(sorted_freqs))
    slope_top, r2_top = loglog_slope(ranks[:top_k], sorted_freqs[:top_k])
    return {
        "zipf_slope_full": slope_full,
        "zipf_r2_full": r2_full,
        "zipf_slope_top1000": slope_top,
        "zipf_r2_top1000": r2_top,
    }


def analyze_heaps(docs: list[list[str]]) -> dict:
    """Heaps: V(N) ~ N^β; natural language β ∈ [0.4, 0.6]."""
    cumulative = 0
    seen = set()
    points_n = []
    points_v = []
    sample_at = [10**i for i in range(2, 8)]
    for d in docs:
        for tok in d:
            cumulative += 1
            if tok not in seen:
                seen.add(tok)
            if sample_at and cumulative >= sample_at[0]:
                points_n.append(cumulative)
                points_v.append(len(seen))
                sample_at.pop(0)
            if not sample_at:
                break
        if not sample_at:
            break
    if len(points_n) >= 3:
        slope, r2 = loglog_slope(points_n, points_v)
    else:
        slope, r2 = float("nan"), float("nan")
    return {"heaps_beta": slope, "heaps_r2": r2,
            "heaps_max_N": max(points_n) if points_n else 0,
            "heaps_max_V": max(points_v) if points_v else 0}


# ── 2. sequential / n-gram structure ───────────────────────────────────────


def conditional_entropies(docs: list[list[str]], orders: list[int]) -> dict:
    """H(w_i | w_{i-k}..w_{i-1}) at orders 0,1,2,3,...

    H_n is the entropy of the next token given the previous n. For natural
    English, H decreases noticeably from H0 (unigram, ~10 bits) to H3 (~5-6
    bits / token); flat or near-flat curves indicate weak grammatical structure.
    """
    out = {}
    # unigram (order 0): H0 = unigram entropy
    counts = Counter()
    for d in docs:
        counts.update(d)
    n = sum(counts.values())
    H0 = -sum((c / n) * math.log2(c / n) for c in counts.values())
    out["H_order_0"] = H0
    # higher orders via conditional context counts
    for order in orders:
        if order == 0:
            continue
        ctx_counts = Counter()
        joint_counts = Counter()
        for d in docs:
            for i in range(order, len(d)):
                ctx = tuple(d[i - order:i])
                joint = ctx + (d[i],)
                ctx_counts[ctx] += 1
                joint_counts[joint] += 1
        # H(W | C) = sum_c p(c) sum_w p(w|c) log p(w|c)
        n_total = sum(ctx_counts.values())
        H = 0.0
        # group joints by context
        ctx_group: dict[tuple, list[int]] = defaultdict(list)
        for joint, c in joint_counts.items():
            ctx_group[joint[:-1]].append(c)
        for ctx, freqs in ctx_group.items():
            p_ctx = ctx_counts[ctx] / n_total
            tot = sum(freqs)
            h_local = -sum((f / tot) * math.log2(f / tot) for f in freqs)
            H += p_ctx * h_local
        out[f"H_order_{order}"] = H
    return out


# ── 3. long-range mutual information ──────────────────────────────────────


def long_range_mi(docs: list[list[str]], distances: list[int],
                  vocab_cap: int = 2000) -> dict:
    """Mutual information I(W_i; W_{i+d}) at multiple distances.

    Natural language: power-law decay (I ~ d^-α with α ∈ [0.4, 1.0]).
    Markov / random text: exponential decay or rapid drop to 0.
    """
    counts = Counter()
    for d in docs:
        counts.update(d)
    top_tokens = {t: i for i, (t, _) in enumerate(counts.most_common(vocab_cap))}
    V = len(top_tokens)
    # unigram p
    flat = [t for d in docs for t in d if t in top_tokens]
    n = len(flat)
    unigram_count = np.zeros(V)
    for t in flat:
        unigram_count[top_tokens[t]] += 1
    p = unigram_count / unigram_count.sum()
    H_uni = -(p[p > 0] * np.log2(p[p > 0])).sum()

    out = {"unigram_entropy_topV": float(H_uni)}
    for d_dist in distances:
        joint = np.zeros((V, V), dtype=np.int64)
        for doc in docs:
            kept = [top_tokens[t] for t in doc if t in top_tokens]
            for i in range(len(kept) - d_dist):
                joint[kept[i], kept[i + d_dist]] += 1
        if joint.sum() == 0:
            out[f"MI_d{d_dist}"] = float("nan")
            continue
        pj = joint / joint.sum()
        pi = pj.sum(axis=1, keepdims=True)
        pk = pj.sum(axis=0, keepdims=True)
        # nonzero mask
        mask = pj > 0
        outer = pi * pk
        outer = np.where(outer > 0, outer, 1.0)
        mi = float((pj[mask] * np.log2(pj[mask] / outer[mask])).sum())
        out[f"MI_d{d_dist}"] = mi
    return out


# ── 4. surface structure ──────────────────────────────────────────────────


def surface_structure(docs: list[list[str]]) -> dict:
    """Bracket pairing, sentence-end punctuation distribution.

    Natural language: matched-bracket nesting depth distribution roughly
    geometric; quote/paren imbalance very low.
    """
    pairs = [('"', '"'), ("'", "'"), ('(', ')'), ('[', ']'), ('{', '}'),
             ('“', '”'), ('‘', '’')]
    paired_balance = 0
    total_brackets = 0
    for d in docs:
        joined = "".join(d)
        for o, c in pairs:
            if o == c:
                # symmetric quotes — count must be even
                cnt = joined.count(o)
                total_brackets += cnt
                paired_balance += (cnt % 2 == 0)
            else:
                co = joined.count(o)
                cc = joined.count(c)
                total_brackets += co + cc
                paired_balance += (co == cc)
    bracket_pairing_consistency = paired_balance / (len(docs) * len(pairs))

    enders = (".", "?", "!")
    ender_counts = Counter()
    for d in docs:
        if d:
            last = d[-1]
            if last in enders:
                ender_counts[last] += 1
            else:
                ender_counts["[other]"] += 1
    n = sum(ender_counts.values())
    end_dist = {k: v / n for k, v in ender_counts.items()}
    end_entropy = -sum(p * math.log2(p) for p in end_dist.values() if p > 0)

    return {
        "bracket_pairing_consistency": bracket_pairing_consistency,
        "doc_end_period_frac": end_dist.get(".", 0.0),
        "doc_end_question_frac": end_dist.get("?", 0.0),
        "doc_end_exclam_frac": end_dist.get("!", 0.0),
        "doc_end_other_frac": end_dist.get("[other]", 0.0),
        "doc_end_entropy": end_entropy,
        "total_brackets_observed": total_brackets,
    }


# ── 5. burstiness / topical coherence ────────────────────────────────────


def burstiness(docs: list[list[str]], top_k: int = 100) -> dict:
    """For top-K tokens, compare within-doc reuse rate to unigram baseline.

    Natural language: top tokens *are* repeated within docs (burstiness > 1
    means within-doc repetition exceeds the unigram model's prediction).
    A purely Markov text would produce burstiness ≈ 1 for most tokens.
    """
    counts = Counter()
    for d in docs:
        counts.update(d)
    n_tokens = sum(counts.values())
    n_docs = len(docs)
    top_tokens = [t for t, _ in counts.most_common(top_k)]
    p = {t: counts[t] / n_tokens for t in top_tokens}
    # within-doc presence rate (binary): observed vs expected under independence
    burst_ratios = []
    for t in top_tokens:
        present = 0
        in_doc_count = 0
        for d in docs:
            if t in d:
                present += 1
                in_doc_count += d.count(t)
        observed_pres = present / n_docs
        # expected presence under independence: 1 - (1-p)^L_avg
        avg_len = n_tokens / n_docs
        expected_pres = 1.0 - (1.0 - p[t]) ** avg_len
        if expected_pres > 0:
            burst_ratios.append(observed_pres / expected_pres)
        # also: avg occurrences when present vs. p[t] * avg_len
        if present > 0:
            avg_when_present = in_doc_count / present
            expected_when_present = p[t] * avg_len
    mean_burst = float(np.mean(burst_ratios))
    median_burst = float(np.median(burst_ratios))
    return {"burstiness_mean_top100": mean_burst,
            "burstiness_median_top100": median_burst}


# ── 6. compression-based structure ────────────────────────────────────────


def compression_ratio(docs: list[list[str]], n_sample: int = 5000) -> dict:
    """gzip(corpus) / gzip(shuffled corpus). Ratio < 1 indicates ordered
    structure beyond unigram (i.e., grammatical regularities). Natural
    languages show ratios around 0.55-0.70. Random shuffle ≈ 1.0.
    """
    sample = docs[:n_sample]
    text = "\n".join(" ".join(d) for d in sample)
    flat = []
    for d in sample:
        flat.extend(d)
    rng = random.Random(0)
    rng.shuffle(flat)
    avg_len = sum(len(d) for d in sample) / len(sample)
    shuf_text = []
    cursor = 0
    for d in sample:
        l = len(d)
        shuf_text.append(" ".join(flat[cursor:cursor + l]))
        cursor += l
    shuf_text_str = "\n".join(shuf_text)

    gzipped = len(gzip.compress(text.encode("utf-8")))
    gzipped_shuf = len(gzip.compress(shuf_text_str.encode("utf-8")))
    return {
        "gzip_corpus_bytes": gzipped,
        "gzip_shuffled_bytes": gzipped_shuf,
        "gzip_ratio_corpus_vs_shuffled": gzipped / gzipped_shuf,
    }


# ── 7-8. distributional class induction & class-bigram grammar ─────────────


def induce_classes(docs: list[list[str]], vocab_cap: int = 2000,
                   n_classes: int = 50, window: int = 3,
                   svd_dims: int = 100) -> dict:
    """PPMI + SVD + KMeans → distributional POS-like classes.

    Then run a class-bigram model on the original corpus and report:
      - class transition entropy H(C_{i+1} | C_i)
      - perplexity reduction from class-bigram over class-unigram
      - max-class concentration (largest class's share)
    """
    counts = Counter()
    for d in docs:
        counts.update(d)
    top_words = [w for w, _ in counts.most_common(vocab_cap)]
    word_idx = {w: i for i, w in enumerate(top_words)}
    V = len(top_words)

    # build word x context co-occurrence (sparse) over a window
    rows, cols, data = [], [], []
    for d in docs:
        L = len(d)
        for i in range(L):
            wi = word_idx.get(d[i])
            if wi is None:
                continue
            for j in range(max(0, i - window), min(L, i + window + 1)):
                if j == i:
                    continue
                wj = word_idx.get(d[j])
                if wj is None:
                    continue
                rows.append(wi)
                cols.append(wj)
                data.append(1)
    cooc = csr_matrix((data, (rows, cols)), shape=(V, V), dtype=np.float64)
    cooc.sum_duplicates()

    # PPMI
    total = cooc.sum()
    if total == 0:
        return {}
    row_sum = np.asarray(cooc.sum(axis=1)).ravel()
    col_sum = np.asarray(cooc.sum(axis=0)).ravel()
    # convert to PPMI = max(0, log(p(w,c)*N / (p(w)*p(c))) ) sparsely
    cooc = cooc.tocoo()
    pmi_vals = []
    for i, j, v in zip(cooc.row, cooc.col, cooc.data):
        if row_sum[i] > 0 and col_sum[j] > 0:
            pmi = math.log((v * total) / (row_sum[i] * col_sum[j]))
            if pmi > 0:
                pmi_vals.append((i, j, pmi))
    if not pmi_vals:
        return {"class_induction_failed": 1.0}
    rs, cs, vs = zip(*pmi_vals)
    ppmi = csr_matrix((vs, (rs, cs)), shape=(V, V))

    # SVD
    svd = TruncatedSVD(n_components=min(svd_dims, V - 1), random_state=0)
    embs = svd.fit_transform(ppmi)

    # k-means
    km = MiniBatchKMeans(n_clusters=n_classes, random_state=0, n_init=10,
                         batch_size=512)
    labels = km.fit_predict(embs)
    word_class = {w: int(labels[word_idx[w]]) for w in top_words}

    # class size distribution
    class_sizes = Counter(labels.tolist())
    sizes = sorted(class_sizes.values(), reverse=True)
    max_class_share = sizes[0] / V if V else 0.0
    class_size_entropy = -sum((s / V) * math.log2(s / V) for s in sizes if s > 0)
    class_size_entropy_uniform = math.log2(n_classes)

    # class-bigram statistics over the original corpus (only known tokens)
    class_unigram = Counter()
    class_bigram = Counter()
    for d in docs:
        prev = None
        for t in d:
            c = word_class.get(t)
            if c is None:
                prev = None
                continue
            class_unigram[c] += 1
            if prev is not None:
                class_bigram[(prev, c)] += 1
            prev = c
    n_uni = sum(class_unigram.values())
    n_bi = sum(class_bigram.values())
    H_class_unigram = -sum((c / n_uni) * math.log2(c / n_uni)
                           for c in class_unigram.values()) if n_uni else 0.0

    # H(C_{i+1} | C_i)
    cond_groups: dict[int, list[int]] = defaultdict(list)
    for (a, b), c in class_bigram.items():
        cond_groups[a].append(c)
    H_class_bigram = 0.0
    for a, freqs in cond_groups.items():
        p_a = class_unigram[a] / n_uni
        tot = sum(freqs)
        h_local = -sum((f / tot) * math.log2(f / tot) for f in freqs)
        H_class_bigram += p_a * h_local

    # rule-count: how many distinct (class_a, class_b) bigrams are observed
    n_class_rules = len(class_bigram)
    n_class_rules_possible = n_classes * n_classes
    grammar_density = n_class_rules / n_class_rules_possible

    return {
        "n_classes": n_classes,
        "vocab_cap": vocab_cap,
        "class_max_share": max_class_share,
        "class_size_entropy": class_size_entropy,
        "class_size_entropy_uniform_max": class_size_entropy_uniform,
        "H_class_unigram": H_class_unigram,
        "H_class_bigram": H_class_bigram,
        "class_bigram_entropy_reduction": H_class_unigram - H_class_bigram,
        "n_class_bigram_rules_observed": n_class_rules,
        "class_grammar_density": grammar_density,
    }


# ── main ──────────────────────────────────────────────────────────────────


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("path")
    p.add_argument("--max-docs", type=int, default=200_000)
    args = p.parse_args()

    print(f"loading {args.path} (cap={args.max_docs})")
    docs = load_corpus(Path(args.path), max_docs=args.max_docs)
    print(f"loaded {len(docs):,} docs, {sum(len(d) for d in docs):,} tokens", flush=True)

    counts = Counter()
    for d in docs:
        counts.update(d)
    print(f"vocab size: {len(counts):,}\n", flush=True)

    print("[1] distributional laws")
    z = analyze_zipf(counts)
    h = analyze_heaps(docs)
    for k, v in {**z, **h}.items():
        print(f"    {k:35s}  {v:.4f}" if isinstance(v, float) else f"    {k:35s}  {v}")
    # natural language references:
    print("    -- ref: natural English Zipf slope ≈ -1.0; Heaps β ≈ 0.4-0.6 --")

    print("\n[2] sequential conditional entropy (bits/token)")
    ce = conditional_entropies(docs, orders=[1, 2, 3, 4, 5])
    for k, v in ce.items():
        print(f"    {k:35s}  {v:.4f}")
    print("    -- ref: English H0≈10, H1≈8, H2≈6, H3≈5; flat curve = weak grammar --")

    print("\n[3] long-range mutual information (top-2K vocab, bits)")
    mi = long_range_mi(docs, distances=[1, 2, 4, 8, 16, 32, 64], vocab_cap=2000)
    for k, v in mi.items():
        print(f"    {k:35s}  {v:.4f}")
    # power-law fit on MI(d)
    d_keys = sorted(int(k.split("d")[1]) for k in mi if k.startswith("MI_d"))
    mi_vals = [mi[f"MI_d{d}"] for d in d_keys]
    if all(v > 0 for v in mi_vals):
        slope, r2 = loglog_slope(d_keys, mi_vals)
        print(f"    {'MI_decay_slope_loglog':35s}  {slope:.4f}  (r2={r2:.3f})")
        print("    -- ref: natural language MI decays as power law, slope α ∈ [-1.0, -0.4] --")

    print("\n[4] surface structure")
    ss = surface_structure(docs)
    for k, v in ss.items():
        print(f"    {k:35s}  {v:.4f}" if isinstance(v, float) else f"    {k:35s}  {v}")
    print("    -- ref: natural-text bracket consistency > 0.95 --")

    print("\n[5] burstiness (top-100 token within-doc reuse)")
    b = burstiness(docs, top_k=100)
    for k, v in b.items():
        print(f"    {k:35s}  {v:.4f}")
    print("    -- ref: natural language burstiness ≈ 1.5-3.0; pure n-gram ≈ 1.0 --")

    print("\n[6] compression structure")
    c = compression_ratio(docs)
    for k, v in c.items():
        print(f"    {k:35s}  {v:.4f}" if isinstance(v, float) else f"    {k:35s}  {v:,}")
    print("    -- ref: natural language gzip(real)/gzip(shuf) ≈ 0.55-0.75 --")

    print("\n[7] distributional class induction (PPMI+SVD+KMeans → POS-like classes)")
    cls = induce_classes(docs, vocab_cap=2000, n_classes=50, window=3, svd_dims=100)
    for k, v in cls.items():
        print(f"    {k:35s}  {v:.4f}" if isinstance(v, float) else f"    {k:35s}  {v}")
    print("    -- ref: natural language H(C|prev_C)/H(C) ≈ 0.6-0.8; very-low ratio = rigid --")


if __name__ == "__main__":
    main()

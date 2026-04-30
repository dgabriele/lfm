"""Local-only corpus quality analyzer.

Computes summary statistics on a generated alien corpus WITHOUT surfacing
the alien text itself. Output is metrics only — counts, ratios, percentiles,
distribution shape — so it's safe to log/transmit.

Usage:
    python scripts/analyze_corpus_quality.py PATH [PATH...]
"""

from __future__ import annotations

import argparse
import math
from collections import Counter
from pathlib import Path


def analyze(path: Path) -> dict:
    docs = [l.rstrip() for l in path.read_text().splitlines() if l.strip()]
    n_docs = len(docs)
    word_counts = [len(d.split()) for d in docs]

    all_tokens: list[str] = []
    for d in docs:
        all_tokens.extend(d.split())
    n_tokens = len(all_tokens)
    counts = Counter(all_tokens)
    n_types = len(counts)

    # Distribution shape — fractions only, no token strings
    total = float(n_tokens)
    sorted_freqs = sorted(counts.values(), reverse=True)
    top1_frac = sorted_freqs[0] / total
    top5_frac = sum(sorted_freqs[:5]) / total
    top20_frac = sum(sorted_freqs[:20]) / total
    top100_frac = sum(sorted_freqs[:100]) / total

    # Entropy (in bits)
    H = -sum((c / total) * math.log2(c / total) for c in sorted_freqs)

    # Length distribution
    word_counts_sorted = sorted(word_counts)
    def pct(p: float) -> int:
        i = int(p * (n_docs - 1))
        return word_counts_sorted[i]
    p50 = pct(0.5)
    p90 = pct(0.9)
    p95 = pct(0.95)
    p99 = pct(0.99)

    # Doc-end heuristic — what fraction terminate with punctuation that
    # signals natural EOS (period, question, exclamation, ellipsis-period)
    enders = ('.', '?', '!')
    natural_end = sum(1 for d in docs if d.rstrip().endswith(enders))

    # Cross-doc diversity — how unique are the first tokens?
    first_tokens = set(d.split()[0] for d in docs if d.split())
    unique_starts_frac = len(first_tokens) / n_docs

    # Hapax legomena — fraction of types appearing exactly once
    hapax = sum(1 for c in counts.values() if c == 1)
    hapax_frac = hapax / n_types

    # Zipf check — slope of log(rank) vs log(freq) on top 1000 types
    # Natural language tends toward slope ~ -1
    top_n = min(1000, len(sorted_freqs))
    if top_n >= 20:
        log_ranks = [math.log(r + 1) for r in range(top_n)]
        log_freqs = [math.log(f) for f in sorted_freqs[:top_n]]
        n_pts = float(top_n)
        mean_x = sum(log_ranks) / n_pts
        mean_y = sum(log_freqs) / n_pts
        sxy = sum((log_ranks[i] - mean_x) * (log_freqs[i] - mean_y) for i in range(top_n))
        sxx = sum((log_ranks[i] - mean_x) ** 2 for i in range(top_n))
        zipf_slope = sxy / sxx if sxx > 0 else 0.0
    else:
        zipf_slope = float("nan")

    return {
        "n_docs": n_docs,
        "n_tokens": n_tokens,
        "n_types": n_types,
        "ttr": n_types / total,
        "avg_doc_len": sum(word_counts) / n_docs,
        "p50_len": p50,
        "p90_len": p90,
        "p95_len": p95,
        "p99_len": p99,
        "max_len": max(word_counts),
        "natural_end_frac": natural_end / n_docs,
        "unique_starts_frac": unique_starts_frac,
        "top1_frac": top1_frac,
        "top5_frac": top5_frac,
        "top20_frac": top20_frac,
        "top100_frac": top100_frac,
        "entropy_bits": H,
        "hapax_frac": hapax_frac,
        "zipf_slope_top1000": zipf_slope,
    }


def fmt_row(label: str, m: dict) -> str:
    return (
        f"{label:<32}  "
        f"docs={m['n_docs']:>6}  "
        f"toks={m['n_tokens']:>9,}  "
        f"types={m['n_types']:>6,}  "
        f"TTR={m['ttr']:.3f}  "
        f"len(p50/p90/p99/max)={m['p50_len']}/{m['p90_len']}/{m['p99_len']}/{m['max_len']}"
    )


def fmt_distribution(label: str, m: dict) -> str:
    return (
        f"{label:<32}  "
        f"top1={m['top1_frac']*100:5.2f}%  "
        f"top5={m['top5_frac']*100:5.2f}%  "
        f"top20={m['top20_frac']*100:5.2f}%  "
        f"top100={m['top100_frac']*100:5.2f}%  "
        f"H={m['entropy_bits']:.2f}  "
        f"hapax={m['hapax_frac']*100:.1f}%  "
        f"zipf={m['zipf_slope_top1000']:.3f}"
    )


def fmt_structure(label: str, m: dict) -> str:
    return (
        f"{label:<32}  "
        f"natural_end={m['natural_end_frac']*100:5.1f}%  "
        f"unique_starts={m['unique_starts_frac']*100:5.1f}%  "
        f"avg_len={m['avg_doc_len']:.1f}"
    )


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("paths", nargs="+")
    args = p.parse_args()

    results = [(Path(p).name, analyze(Path(p))) for p in args.paths]

    print("=== sizes & lexical ===")
    for label, m in results:
        print(fmt_row(label, m))
    print("\n=== distribution shape ===")
    for label, m in results:
        print(fmt_distribution(label, m))
    print("\n=== structural ===")
    for label, m in results:
        print(fmt_structure(label, m))


if __name__ == "__main__":
    main()

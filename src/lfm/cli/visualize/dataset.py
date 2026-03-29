"""Dataset diagnostic visualization: ``lfm visualize dataset``.

Generates plots inspecting dataset composition, IPA quality, and
linguistic properties — useful for validating a dataset before
committing to a multi-day training run.

Plots:
1. Per-language sample count (bar chart)
2. IPA sequence length distribution (histogram, per-language)
3. Token frequency / Zipf law (rank-frequency log-log)
4. Character-level unigram distribution (IPA phoneme frequencies)
5. Per-language mean sequence length (grouped bar)
6. Vocabulary overlap heatmap (Jaccard similarity between languages)
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from lfm.cli.base import CLICommand

logger = logging.getLogger(__name__)


class DatasetVizCommand(CLICommand):
    """Visualize dataset composition and linguistic properties."""

    @property
    def name(self) -> str:
        return "dataset"

    @property
    def help(self) -> str:
        return "Dataset diagnostic plots (composition, lengths, Zipf, phoneme frequencies)"

    @property
    def description(self) -> str:
        return (
            "Generate diagnostic visualizations for a pre-generated HDF5 "
            "dataset: per-language counts, length distributions, Zipf law, "
            "phoneme frequencies, and cross-language vocabulary overlap."
        )

    def add_arguments(self, parser: argparse.ArgumentParser) -> None:
        parser.add_argument(
            "--dataset-path", required=True,
            help="Path to dataset directory (containing samples.h5)",
        )
        parser.add_argument(
            "--output-dir", default="output/viz/dataset",
            help="Output directory for plots",
        )
        parser.add_argument(
            "--max-samples", type=int, default=50000,
            help="Max samples to load for analysis (default: 50000)",
        )

    def execute(self, args: argparse.Namespace) -> int:
        import collections
        from collections import Counter, defaultdict

        import h5py
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np

        dataset_path = Path(args.dataset_path)
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        h5_path = dataset_path / "samples.h5"
        if not h5_path.exists():
            print(f"samples.h5 not found in {dataset_path}")
            return 1

        # Load data
        with h5py.File(h5_path, "r") as f:
            grp = f["samples"]
            n = min(len(grp["seq"]), args.max_samples)
            langs = [x.decode() if isinstance(x, bytes) else x for x in grp["language"][:n]]
            ipas = [x.decode() if isinstance(x, bytes) else x for x in grp["ipa"][:n]]
            ipa_lengths = grp["ipa_length"][:n].tolist()

        print(f"Loaded {n:,} samples from {dataset_path.name}")

        # Group by language
        by_lang: dict[str, list[str]] = defaultdict(list)
        len_by_lang: dict[str, list[int]] = defaultdict(list)
        for lang, ipa, length in zip(langs, ipas, ipa_lengths):
            by_lang[lang].append(ipa)
            len_by_lang[lang].append(length)

        lang_order = sorted(by_lang.keys(), key=lambda l: -len(by_lang[l]))

        # ── 1. Per-language sample count ──────────────────────────
        fig, ax = plt.subplots(figsize=(10, 5))
        counts = [len(by_lang[l]) for l in lang_order]
        bars = ax.bar(lang_order, counts, color="steelblue")
        ax.set_xlabel("Language")
        ax.set_ylabel("Samples")
        ax.set_title(f"Samples per Language (n={n:,})")
        for bar, count in zip(bars, counts):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                    f"{count:,}", ha="center", va="bottom", fontsize=7)
        plt.tight_layout()
        plt.savefig(output_dir / "lang_counts.png", dpi=150)
        plt.close()
        print("  Saved lang_counts.png")

        # ── 2. IPA length distribution ────────────────────────────
        fig, ax = plt.subplots(figsize=(10, 5))
        all_lengths = ipa_lengths
        ax.hist(all_lengths, bins=80, color="steelblue", alpha=0.7, edgecolor="white")
        ax.axvline(np.mean(all_lengths), color="red", linestyle="--",
                   label=f"Mean={np.mean(all_lengths):.0f}")
        ax.axvline(np.median(all_lengths), color="orange", linestyle="--",
                   label=f"Median={np.median(all_lengths):.0f}")
        ax.set_xlabel("IPA Length (characters)")
        ax.set_ylabel("Count")
        ax.set_title("IPA Sequence Length Distribution")
        ax.legend()
        plt.tight_layout()
        plt.savefig(output_dir / "length_distribution.png", dpi=150)
        plt.close()
        print("  Saved length_distribution.png")

        # ── 3. Per-language mean length ───────────────────────────
        fig, ax = plt.subplots(figsize=(10, 5))
        means = [np.mean(len_by_lang[l]) for l in lang_order]
        stds = [np.std(len_by_lang[l]) for l in lang_order]
        ax.bar(lang_order, means, yerr=stds, color="coral", capsize=3)
        ax.set_xlabel("Language")
        ax.set_ylabel("Mean IPA Length (chars)")
        ax.set_title("Mean IPA Sequence Length per Language")
        plt.tight_layout()
        plt.savefig(output_dir / "lang_mean_length.png", dpi=150)
        plt.close()
        print("  Saved lang_mean_length.png")

        # ── 4. IPA character frequency ────────────────────────────
        char_counter: Counter = Counter()
        for ipa in ipas:
            for c in ipa:
                if not c.isspace():
                    char_counter[c] += 1

        top_chars = char_counter.most_common(50)
        fig, ax = plt.subplots(figsize=(14, 5))
        chars, freqs = zip(*top_chars)
        ax.bar(range(len(chars)), freqs, color="mediumpurple")
        ax.set_xticks(range(len(chars)))
        ax.set_xticklabels(chars, fontsize=9)
        ax.set_ylabel("Frequency")
        ax.set_title("Top 50 IPA Characters")
        plt.tight_layout()
        plt.savefig(output_dir / "ipa_char_freq.png", dpi=150)
        plt.close()
        print("  Saved ipa_char_freq.png")

        # ── 5. Zipf / rank-frequency ─────────────────────────────
        word_counter: Counter = Counter()
        for ipa in ipas:
            for word in ipa.split():
                word_counter[word] += 1

        ranks = list(range(1, len(word_counter) + 1))
        freqs_sorted = sorted(word_counter.values(), reverse=True)

        fig, ax = plt.subplots(figsize=(8, 6))
        ax.loglog(ranks, freqs_sorted, ".", markersize=2, color="steelblue")
        # Zipf reference line
        max_freq = freqs_sorted[0]
        zipf_ref = [max_freq / r for r in ranks]
        ax.loglog(ranks, zipf_ref, "--", color="red", alpha=0.5, label="Zipf (1/r)")
        ax.set_xlabel("Rank")
        ax.set_ylabel("Frequency")
        ax.set_title(f"Word Rank-Frequency (Zipf) — {len(word_counter):,} unique words")
        ax.legend()
        plt.tight_layout()
        plt.savefig(output_dir / "zipf.png", dpi=150)
        plt.close()
        print("  Saved zipf.png")

        # ── 6. Cross-language vocabulary overlap (Jaccard) ────────
        lang_vocabs: dict[str, set[str]] = {}
        for lang in lang_order:
            vocab: set[str] = set()
            for ipa in by_lang[lang]:
                vocab.update(ipa.split())
            lang_vocabs[lang] = vocab

        n_langs = len(lang_order)
        jaccard = np.zeros((n_langs, n_langs))
        for i in range(n_langs):
            for j in range(n_langs):
                vi = lang_vocabs[lang_order[i]]
                vj = lang_vocabs[lang_order[j]]
                intersection = len(vi & vj)
                union = len(vi | vj)
                jaccard[i, j] = intersection / max(union, 1)

        fig, ax = plt.subplots(figsize=(9, 8))
        im = ax.imshow(jaccard, cmap="YlOrRd", vmin=0, vmax=0.3)
        ax.set_xticks(range(n_langs))
        ax.set_yticks(range(n_langs))
        ax.set_xticklabels(lang_order, rotation=45, ha="right")
        ax.set_yticklabels(lang_order)
        ax.set_title("Cross-Language Vocabulary Overlap (Jaccard)")
        plt.colorbar(im, ax=ax, shrink=0.8)
        plt.tight_layout()
        plt.savefig(output_dir / "vocab_overlap.png", dpi=150)
        plt.close()
        print("  Saved vocab_overlap.png")

        # ── Summary stats ─────────────────────────────────────────
        print(f"\n  Total samples:     {n:,}")
        print(f"  Languages:         {len(by_lang)}")
        print(f"  Unique words:      {len(word_counter):,}")
        print(f"  Unique IPA chars:  {len(char_counter):,}")
        print(f"  Mean IPA length:   {np.mean(all_lengths):.0f} chars")
        print(f"  Median IPA length: {np.median(all_lengths):.0f} chars")
        print(f"  Output: {output_dir}/")

        return 0

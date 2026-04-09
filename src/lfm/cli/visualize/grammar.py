"""CLI subcommand for grammar analysis visualizations.

Usage::

    lfm visualize grammar --corpus data/translator/corpus.txt
    lfm visualize grammar --corpus data/corpus.txt --num-categories 32 --min-freq 3
"""

from __future__ import annotations

import argparse
import logging

from lfm.cli.base import CLICommand

logger = logging.getLogger(__name__)


class GrammarCommand(CLICommand):
    """Distributional grammar analysis and visualization for emergent language."""

    @property
    def name(self) -> str:
        return "grammar"

    @property
    def help(self) -> str:
        return "Grammar analysis plots (categories, transitions, Zipf, compositionality)"

    @property
    def description(self) -> str:
        return (
            "Distributional grammar analysis of a Neuroglot corpus. "
            "Discovers latent word categories via PPMI + SVD + KMeans, "
            "then generates transition heatmaps, positional distributions, "
            "productivity scatter plots, Zipf analysis, and compositionality metrics."
        )

    def add_arguments(self, parser: argparse.ArgumentParser) -> None:
        parser.add_argument(
            "--corpus",
            required=True,
            help="Path to corpus text file (one paragraph per line)",
        )
        parser.add_argument(
            "--output-dir",
            default="output/viz-grammar/",
            help="Output directory for figures (default: output/viz-grammar/)",
        )
        parser.add_argument(
            "--num-samples",
            type=int,
            default=10_000,
            help="Max paragraphs to analyze (default: 10000)",
        )
        parser.add_argument(
            "--num-categories",
            type=int,
            default=24,
            help="Number of word clusters / categories (default: 24)",
        )
        parser.add_argument(
            "--min-freq",
            type=int,
            default=5,
            help="Minimum word frequency for inclusion (default: 5)",
        )

    def execute(self, args: argparse.Namespace) -> int:
        from pathlib import Path

        from lfm.visualize.grammar import GrammarAnalyzer, plot_all

        corpus_path = self.validate_file_exists(args.corpus, "Corpus file")
        if corpus_path is None:
            return 1

        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        print(f"Analyzing corpus: {corpus_path}")
        print(f"  Paragraphs:  up to {args.num_samples:,}")
        print(f"  Categories:  {args.num_categories}")
        print(f"  Min freq:    {args.min_freq}")
        print(f"  Output:      {output_dir}/")
        print()

        analyzer = GrammarAnalyzer(
            corpus_path=str(corpus_path),
            num_samples=args.num_samples,
            num_categories=args.num_categories,
            min_freq=args.min_freq,
        )
        analysis = analyzer.analyze()

        print(f"\nCorpus statistics:")
        print(f"  Sentences:   {len(analysis.sentences):,}")
        print(f"  Vocabulary:  {len(analysis.vocab):,} words (min_freq >= {args.min_freq})")
        print(f"  Categories:  {analysis.num_categories}")
        print(f"  Position MI: {analysis.position_category_mi:.4f} bits")
        print()

        paths = plot_all(analysis, output_dir)

        print(f"\nGenerated {len(paths)} plots:")
        for p in paths:
            print(f"  {p}")

        return 0

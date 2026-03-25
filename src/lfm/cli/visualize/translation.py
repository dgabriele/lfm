"""CLI subcommand for translation evaluation visualizations."""

from __future__ import annotations

import argparse

from lfm.cli.base import CLICommand


class TranslationCommand(CLICommand):
    """Visualize translation evaluation results."""

    @property
    def name(self) -> str:
        return "translation"

    @property
    def help(self) -> str:
        return "Translation evaluation visualizations (BLEU, similarity, examples)"

    def add_arguments(self, parser: argparse.ArgumentParser) -> None:
        parser.add_argument(
            "--results-dir",
            default="data/models/v1/translator",
            help="Directory with results.json and translations.jsonl",
        )
        parser.add_argument(
            "--output-dir",
            default=None,
            help="Output directory for figures (default: results-dir)",
        )
        parser.add_argument(
            "--format",
            choices=["png", "svg", "pdf"],
            default="png",
            help="Output format (default: png)",
        )
        parser.add_argument(
            "--dpi",
            type=int,
            default=150,
            help="Output resolution (default: 150)",
        )

    def execute(self, args: argparse.Namespace) -> int:
        from lfm.visualize.translation import TranslationVisualization

        results_dir = self.validate_file_exists(args.results_dir, "Results directory")
        if results_dir is None:
            return 1

        viz = TranslationVisualization(
            results_dir=str(results_dir),
            output_dir=args.output_dir,
            fmt=args.format,
            dpi=args.dpi,
        )
        paths = viz.generate_all()

        if not paths:
            print("No visualizations generated. Check that results.json exists.")
            return 1

        for p in paths:
            print(f"  Saved: {p}")
        return 0

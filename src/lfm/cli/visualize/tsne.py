"""CLI subcommand: ``lfm visualize tsne``."""

from __future__ import annotations

import argparse

from lfm.cli.base import CLICommand


class TSNECommand(CLICommand):
    @property
    def name(self) -> str:
        return "tsne"

    @property
    def help(self) -> str:
        return "t-SNE / UMAP 2D latent space projection"

    def add_arguments(self, parser: argparse.ArgumentParser) -> None:
        parser.add_argument(
            "--color-by",
            choices=["language", "family", "type"],
            default="language",
            help="Color scatter points by (default: language)",
        )
        parser.add_argument(
            "--perplexity",
            type=int,
            default=30,
            help="t-SNE perplexity (default: 30)",
        )
        parser.add_argument(
            "--method",
            choices=["tsne", "umap"],
            default="tsne",
            help="Dimensionality reduction method (default: tsne)",
        )

    def execute(self, args: argparse.Namespace) -> int:
        from lfm.visualize.config import VisualizeConfig
        from lfm.visualize.loader import encode_labeled_corpus, load_checkpoint
        from lfm.visualize.tsne import TSNEVisualization

        config = VisualizeConfig.from_args(args)
        print(f"Loading checkpoint: {config.checkpoint}")
        model_data = load_checkpoint(config)

        print("Encoding labeled corpus...")
        corpus_data = encode_labeled_corpus(model_data, config)

        viz = TSNEVisualization(config)
        figures = viz.generate(corpus_data)
        paths = viz.save(figures, ["by_language", "by_family", "by_type"])
        for p in paths:
            print(f"  Saved: {p}")
        return 0

"""CLI subcommand: ``lfm visualize attention``."""

from __future__ import annotations

import argparse

from lfm.cli.base import CLICommand


class AttentionCommand(CLICommand):
    @property
    def name(self) -> str:
        return "attention"

    @property
    def help(self) -> str:
        return "Multi-scale attention pattern heatmaps"

    def add_arguments(self, parser: argparse.ArgumentParser) -> None:
        parser.add_argument(
            "--n-sentences",
            type=int,
            default=20,
            help="Number of sentences to decode for attention (default: 20)",
        )
        parser.add_argument(
            "--heads",
            default="",
            help="Comma-separated head indices to plot (default: all)",
        )

    def execute(self, args: argparse.Namespace) -> int:
        from lfm.visualize.attention import AttentionVisualization
        from lfm.visualize.config import VisualizeConfig
        from lfm.visualize.loader import encode_corpus, load_checkpoint

        config = VisualizeConfig.from_args(args)
        print(f"Loading checkpoint: {config.checkpoint}")
        model_data = load_checkpoint(config)

        print("Encoding corpus...")
        corpus_data = encode_corpus(model_data, config)

        viz = AttentionVisualization(config)
        figures = viz.generate({**corpus_data, **model_data})
        paths = viz.save(figures, ["per_head", "average", "entropy"])
        for p in paths:
            print(f"  Saved: {p}")
        return 0

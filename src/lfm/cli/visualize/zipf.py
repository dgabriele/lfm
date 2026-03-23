"""CLI subcommand: ``lfm visualize zipf``."""

from __future__ import annotations

import argparse

from lfm.cli.base import CLICommand


class ZipfCommand(CLICommand):
    @property
    def name(self) -> str:
        return "zipf"

    @property
    def help(self) -> str:
        return "Token frequency / Zipf law analysis"

    def add_arguments(self, parser: argparse.ArgumentParser) -> None:
        parser.add_argument(
            "--n-samples",
            type=int,
            default=10000,
            help="Number of z samples to decode (default: 10000)",
        )

    def execute(self, args: argparse.Namespace) -> int:
        from lfm.visualize.config import VisualizeConfig
        from lfm.visualize.loader import encode_corpus, load_checkpoint
        from lfm.visualize.zipf import ZipfVisualization

        config = VisualizeConfig.from_args(args)
        print(f"Loading checkpoint: {config.checkpoint}")
        model_data = load_checkpoint(config)

        print("Encoding corpus...")
        corpus_data = encode_corpus(model_data, config)

        viz = ZipfVisualization(config)
        figures = viz.generate({**corpus_data, **model_data})
        paths = viz.save(figures, ["rank_frequency", "exponent_comparison"])
        for p in paths:
            print(f"  Saved: {p}")
        return 0

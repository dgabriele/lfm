"""CLI subcommand: ``lfm visualize random-z-quality``."""

from __future__ import annotations

import argparse

from lfm.cli.base import CLICommand


class RandomZQualityCommand(CLICommand):
    @property
    def name(self) -> str:
        return "random-z-quality"

    @property
    def help(self) -> str:
        return "Decode N random z vectors from the prior; report tag validity + uniqueness"

    def add_arguments(self, parser: argparse.ArgumentParser) -> None:
        parser.add_argument(
            "--num-samples", type=int, default=1000,
            help="How many z vectors to sample (default: 1000).",
        )
        # Note: --seed is already provided by _add_shared_arguments.

    def execute(self, args: argparse.Namespace) -> int:
        from lfm.visualize.config import VisualizeConfig
        from lfm.visualize.loader import load_checkpoint
        from lfm.visualize.random_z_quality import RandomZQualityVisualization

        config = VisualizeConfig.from_args(args)
        print(f"Loading checkpoint: {config.checkpoint}")
        model_data = load_checkpoint(config)

        viz = RandomZQualityVisualization(config)
        figures = viz.generate({
            "model_data": model_data,
            "num_samples": args.num_samples,
            "seed": args.seed,
        })
        paths = viz.save(figures, ["tag_validity", "length_hist", "tag_types"])
        for p in paths:
            print(f"  Saved: {p}")
        return 0

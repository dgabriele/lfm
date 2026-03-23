"""CLI subcommand: ``lfm visualize interpolation``."""

from __future__ import annotations

import argparse

from lfm.cli.base import CLICommand


class InterpolationCommand(CLICommand):
    @property
    def name(self) -> str:
        return "interpolation"

    @property
    def help(self) -> str:
        return "Latent interpolation trajectories between languages"

    def add_arguments(self, parser: argparse.ArgumentParser) -> None:
        parser.add_argument(
            "--pairs",
            default="pol-vie,ara-fin",
            help="Language pairs for interpolation, e.g. pol-vie,ara-fin",
        )
        parser.add_argument(
            "--steps",
            type=int,
            default=20,
            help="Number of interpolation steps (default: 20)",
        )

    def execute(self, args: argparse.Namespace) -> int:
        from lfm.visualize.config import VisualizeConfig
        from lfm.visualize.interpolation import InterpolationVisualization
        from lfm.visualize.loader import encode_labeled_corpus, load_checkpoint

        config = VisualizeConfig.from_args(args)
        print(f"Loading checkpoint: {config.checkpoint}")
        model_data = load_checkpoint(config)

        print("Encoding labeled corpus...")
        corpus_data = encode_labeled_corpus(model_data, config)

        viz = InterpolationVisualization(config)
        figures = viz.generate({**corpus_data, **model_data})
        paths = viz.save(figures, ["trajectories", "decoded_text"])
        for p in paths:
            print(f"  Saved: {p}")
        return 0

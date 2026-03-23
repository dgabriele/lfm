"""CLI subcommand: ``lfm visualize length-dist``."""

from __future__ import annotations

import argparse

from lfm.cli.base import CLICommand


class LengthDistCommand(CLICommand):
    @property
    def name(self) -> str:
        return "length-dist"

    @property
    def help(self) -> str:
        return "Output length distribution histograms"

    def add_arguments(self, parser: argparse.ArgumentParser) -> None:
        pass  # No extra args

    def execute(self, args: argparse.Namespace) -> int:
        from lfm.visualize.config import VisualizeConfig
        from lfm.visualize.length_dist import LengthDistVisualization
        from lfm.visualize.loader import encode_corpus, load_checkpoint

        config = VisualizeConfig.from_args(args)
        print(f"Loading checkpoint: {config.checkpoint}")
        model_data = load_checkpoint(config)

        print("Encoding corpus...")
        corpus_data = encode_corpus(model_data, config)

        viz = LengthDistVisualization(config)
        figures = viz.generate({**corpus_data, **model_data})
        paths = viz.save(figures, ["histogram", "by_language", "vs_znorm"])
        for p in paths:
            print(f"  Saved: {p}")
        return 0

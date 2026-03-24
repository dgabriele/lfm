"""CLI subcommand: ``lfm visualize smoothness``."""

from __future__ import annotations

import argparse

from lfm.cli.base import CLICommand


class SmoothnessCommand(CLICommand):
    @property
    def name(self) -> str:
        return "smoothness"

    @property
    def help(self) -> str:
        return "Lipschitz continuity and interpolation smoothness analysis"

    def add_arguments(self, parser: argparse.ArgumentParser) -> None:
        pass

    def execute(self, args: argparse.Namespace) -> int:
        from lfm.visualize.config import VisualizeConfig
        from lfm.visualize.loader import encode_corpus, load_checkpoint
        from lfm.visualize.smoothness import SmoothnessVisualization

        config = VisualizeConfig.from_args(args)
        print(f"Loading checkpoint: {config.checkpoint}")
        model_data = load_checkpoint(config)

        print("Encoding corpus...")
        corpus_data = encode_corpus(model_data, config)

        viz = SmoothnessVisualization(config)
        figures = viz.generate({**corpus_data, **model_data})
        paths = viz.save(figures, ["lipschitz", "jaccard", "interpolation_continuity"])
        for p in paths:
            print(f"  Saved: {p}")
        return 0

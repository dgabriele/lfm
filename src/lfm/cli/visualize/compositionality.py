"""CLI subcommand: ``lfm visualize compositionality``."""

from __future__ import annotations

import argparse

from lfm.cli.base import CLICommand


class CompositionalityCommand(CLICommand):
    @property
    def name(self) -> str:
        return "compositionality"

    @property
    def help(self) -> str:
        return "Positional disentanglement and feature-position mutual information"

    def add_arguments(self, parser: argparse.ArgumentParser) -> None:
        pass

    def execute(self, args: argparse.Namespace) -> int:
        from lfm.visualize.compositionality import CompositionalityVisualization
        from lfm.visualize.config import VisualizeConfig
        from lfm.visualize.loader import encode_corpus, load_checkpoint

        config = VisualizeConfig.from_args(args)
        print(f"Loading checkpoint: {config.checkpoint}")
        model_data = load_checkpoint(config)

        print("Encoding corpus...")
        corpus_data = encode_corpus(model_data, config)

        viz = CompositionalityVisualization(config)
        figures = viz.generate({**corpus_data, **model_data})
        paths = viz.save(figures, ["heatmap", "scores", "mutual_info"])
        for p in paths:
            print(f"  Saved: {p}")
        return 0

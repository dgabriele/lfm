"""CLI subcommand: ``lfm visualize adaptiveness``."""

from __future__ import annotations

import argparse

from lfm.cli.base import CLICommand


class AdaptivenessCommand(CLICommand):
    @property
    def name(self) -> str:
        return "adaptiveness"

    @property
    def help(self) -> str:
        return "Input complexity vs output length and diversity analysis"

    def add_arguments(self, parser: argparse.ArgumentParser) -> None:
        pass

    def execute(self, args: argparse.Namespace) -> int:
        from lfm.visualize.adaptiveness import AdaptivenessVisualization
        from lfm.visualize.config import VisualizeConfig
        from lfm.visualize.loader import encode_corpus, load_checkpoint

        config = VisualizeConfig.from_args(args)
        print(f"Loading checkpoint: {config.checkpoint}")
        model_data = load_checkpoint(config)

        print("Encoding corpus...")
        corpus_data = encode_corpus(model_data, config)

        viz = AdaptivenessVisualization(config)
        figures = viz.generate({**corpus_data, **model_data})
        paths = viz.save(figures, ["length_adaptation", "diversity", "complexity_profile"])
        for p in paths:
            print(f"  Saved: {p}")
        return 0

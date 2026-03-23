"""CLI subcommand: ``lfm visualize all`` — run all visualizations."""

from __future__ import annotations

import argparse

from lfm.cli.base import CLICommand


class AllCommand(CLICommand):
    @property
    def name(self) -> str:
        return "all"

    @property
    def help(self) -> str:
        return "Run all visualizations"

    def add_arguments(self, parser: argparse.ArgumentParser) -> None:
        pass  # Uses only shared args

    def execute(self, args: argparse.Namespace) -> int:
        from lfm.visualize.config import VisualizeConfig
        from lfm.visualize.loader import (
            encode_corpus,
            encode_labeled_corpus,
            load_checkpoint,
        )
        from lfm.visualize.suite import VisualizationSuite

        config = VisualizeConfig.from_args(args)
        print(f"Loading checkpoint: {config.checkpoint}")
        model_data = load_checkpoint(config)

        print("Encoding labeled corpus...")
        labeled_data = encode_labeled_corpus(model_data, config)

        print("Encoding full corpus...")
        corpus_data = encode_corpus(model_data, config)

        suite = VisualizationSuite(config)
        paths = suite.run_all(
            model_data=model_data,
            labeled_data=labeled_data,
            corpus_data=corpus_data,
        )
        print(f"\nGenerated {len(paths)} figures in {config.output_dir}/")
        return 0

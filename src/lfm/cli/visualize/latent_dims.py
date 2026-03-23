"""CLI subcommand: ``lfm visualize latent-dims``."""

from __future__ import annotations

import argparse

from lfm.cli.base import CLICommand


class LatentDimsCommand(CLICommand):
    @property
    def name(self) -> str:
        return "latent-dims"

    @property
    def help(self) -> str:
        return "Per-dimension variance, PCA, language discrimination"

    def add_arguments(self, parser: argparse.ArgumentParser) -> None:
        pass  # No extra args

    def execute(self, args: argparse.Namespace) -> int:
        from lfm.visualize.config import VisualizeConfig
        from lfm.visualize.latent_dims import LatentDimsVisualization
        from lfm.visualize.loader import encode_labeled_corpus, load_checkpoint

        config = VisualizeConfig.from_args(args)
        print(f"Loading checkpoint: {config.checkpoint}")
        model_data = load_checkpoint(config)

        print("Encoding labeled corpus...")
        corpus_data = encode_labeled_corpus(model_data, config)

        viz = LatentDimsVisualization(config)
        figures = viz.generate(corpus_data)
        paths = viz.save(figures, ["variance", "lang_heatmap", "pca", "f_statistic"])
        for p in paths:
            print(f"  Saved: {p}")
        return 0

"""CLI subcommand: ``lfm visualize clustering``."""

from __future__ import annotations

import argparse

from lfm.cli.base import CLICommand


class ClusteringCommand(CLICommand):
    @property
    def name(self) -> str:
        return "clustering"

    @property
    def help(self) -> str:
        return "Hierarchical clustering dendrogram + distance heatmap"

    def add_arguments(self, parser: argparse.ArgumentParser) -> None:
        parser.add_argument(
            "--metric",
            choices=["cosine", "l2"],
            default="cosine",
            help="Distance metric (default: cosine)",
        )
        parser.add_argument(
            "--linkage",
            choices=["ward", "average", "complete"],
            default="average",
            help="Linkage method (default: average)",
        )

    def execute(self, args: argparse.Namespace) -> int:
        from lfm.visualize.clustering import ClusteringVisualization
        from lfm.visualize.config import VisualizeConfig
        from lfm.visualize.loader import encode_labeled_corpus, load_checkpoint

        config = VisualizeConfig.from_args(args)
        print(f"Loading checkpoint: {config.checkpoint}")
        model_data = load_checkpoint(config)

        print("Encoding labeled corpus...")
        corpus_data = encode_labeled_corpus(model_data, config)

        viz = ClusteringVisualization(config)
        figures = viz.generate(corpus_data)
        paths = viz.save(figures, ["dendrogram", "heatmap"])
        for p in paths:
            print(f"  Saved: {p}")
        return 0

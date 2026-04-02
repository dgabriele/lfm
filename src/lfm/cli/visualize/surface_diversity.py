"""CLI subcommand: ``lfm visualize surface-diversity``."""

from __future__ import annotations

import argparse

from lfm.cli.base import CLICommand


class SurfaceDiversityCommand(CLICommand):
    @property
    def name(self) -> str:
        return "surface-diversity"

    @property
    def help(self) -> str:
        return "Surface diversity: unique decoded forms per z vector"

    def add_arguments(self, parser: argparse.ArgumentParser) -> None:
        pass

    def execute(self, args: argparse.Namespace) -> int:
        from lfm.visualize.config import VisualizeConfig
        from lfm.visualize.loader import encode_corpus, load_checkpoint
        from lfm.visualize.surface_diversity import SurfaceDiversityVisualization

        config = VisualizeConfig.from_args(args)
        print(f"Loading checkpoint: {config.checkpoint}")
        model_data = load_checkpoint(config)

        print("Encoding corpus...")
        corpus_data = encode_corpus(model_data, config)

        viz = SurfaceDiversityVisualization(config)
        figures = viz.generate({**corpus_data, **model_data})
        paths = viz.save(figures, ["summary", "edit_vs_z", "position_entropy"])
        for p in paths:
            print(f"  Saved: {p}")
        return 0

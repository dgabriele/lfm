"""CLI subcommand: ``lfm visualize pc-traversal``."""

from __future__ import annotations

import argparse

from lfm.cli.base import CLICommand


class PCTraversalCommand(CLICommand):
    @property
    def name(self) -> str:
        return "pc-traversal"

    @property
    def help(self) -> str:
        return "Sweep top PCs of the posterior and decode each step"

    def add_arguments(self, parser: argparse.ArgumentParser) -> None:
        parser.add_argument(
            "--top-k", type=int, default=8,
            help="How many principal components to traverse (default: 8).",
        )
        parser.add_argument(
            "--n-steps", type=int, default=7,
            help="Samples along each PC (default: 7, covers ±3σ in 1σ steps).",
        )
        parser.add_argument(
            "--span-sigma", type=float, default=3.0,
            help="Half-span of the sweep in σ units (default: 3.0).",
        )

    def execute(self, args: argparse.Namespace) -> int:
        from lfm.visualize.config import VisualizeConfig
        from lfm.visualize.loader import encode_labeled_corpus, load_checkpoint
        from lfm.visualize.pc_traversal import PCTraversalVisualization

        config = VisualizeConfig.from_args(args)
        print(f"Loading checkpoint: {config.checkpoint}")
        model_data = load_checkpoint(config)
        print("Encoding labeled corpus...")
        corpus_data = encode_labeled_corpus(model_data, config)

        viz = PCTraversalVisualization(config)
        figures = viz.generate({
            "model_data": model_data,
            "z": corpus_data["z"],
            "top_k": args.top_k,
            "n_steps": args.n_steps,
            "span_sigma": args.span_sigma,
        })
        paths = viz.save(figures, ["explained_variance"])
        for p in paths:
            print(f"  Saved: {p}")
        return 0

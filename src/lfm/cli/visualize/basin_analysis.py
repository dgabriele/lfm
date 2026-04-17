"""CLI subcommand: ``lfm visualize basin-analysis``."""

from __future__ import annotations

import argparse

from lfm.cli.base import CLICommand


class BasinAnalysisCommand(CLICommand):
    @property
    def name(self) -> str:
        return "basin-analysis"

    @property
    def help(self) -> str:
        return "Quantify latent attractor basins: radii, transitions, attractor count"

    @property
    def description(self) -> str:
        return (
            "Measure attractor-basin geometry of the VAE latent space: "
            "per-anchor basin/tag/degeneration radii, A→B interpolation "
            "transition widths, and prior-sample attractor count.  "
            "All raw data logged to stdout; figures mirror what's logged."
        )

    def add_arguments(self, parser: argparse.ArgumentParser) -> None:
        parser.add_argument(
            "--config-yaml",
            default="",
            help=(
                "Path to the training config.yaml used to rebuild modules. "
                "Empty = auto-detect as <checkpoint_dir>/config.yaml."
            ),
        )
        parser.add_argument(
            "--anchors",
            default="",
            help=(
                "Semicolon-separated list of anchor prompts (include tags). "
                "Empty = use defaults shared with scripts/decode_checkpoint.py."
            ),
        )
        parser.add_argument(
            "--sigmas",
            default="0.0,0.05,0.10,0.15,0.20,0.30,0.40,0.50,0.70,1.00,1.50",
            help="Comma-separated σ grid for the perturbation sweep.",
        )
        parser.add_argument(
            "--n-directions",
            type=int,
            default=32,
            help="Random directions per σ for basin density (default: 32).",
        )
        parser.add_argument(
            "--alpha-resolution",
            type=int,
            default=21,
            help="Number of α values between A and B for interpolation (default: 21).",
        )
        parser.add_argument(
            "--n-prior",
            type=int,
            default=512,
            help="Prior samples for attractor-count estimate (default: 512).",
        )

    def execute(self, args: argparse.Namespace) -> int:
        from pathlib import Path

        import torch

        from lfm.visualize.basin_analysis import (
            BasinAnalysisVisualization,
            DEFAULT_ANCHORS,
            load_via_build_model,
        )
        from lfm.visualize.config import VisualizeConfig

        config = VisualizeConfig.from_args(args)
        config_yaml = args.config_yaml.strip() or str(
            Path(config.checkpoint).parent / "config.yaml"
        )
        print(f"Loading checkpoint: {config.checkpoint}")
        print(f"Loading config:     {config_yaml}")
        device = torch.device(config.device)
        model_data = load_via_build_model(
            config.checkpoint, config_yaml, device,
            spm_model_path=config.spm_model,
        )
        step = model_data.get("global_step", "?")
        epoch = model_data.get("epoch", "?")
        print(f"Checkpoint metadata: epoch={epoch} step={step}")

        anchors: list[str]
        if args.anchors.strip():
            anchors = [a.strip() for a in args.anchors.split(";") if a.strip()]
        else:
            anchors = list(DEFAULT_ANCHORS)

        sigmas = [float(x) for x in args.sigmas.split(",") if x.strip()]

        viz = BasinAnalysisVisualization(config)
        figures = viz.generate({
            "model_data": model_data,
            "anchors": anchors,
            "sigmas": sigmas,
            "n_directions": args.n_directions,
            "alpha_resolution": args.alpha_resolution,
            "n_prior": args.n_prior,
            "seed": args.seed,
        })
        paths = viz.save(figures)
        for p in paths:
            print(f"  Saved: {p}")
        return 0

"""Visualize subcommand group for ``lfm visualize <type>``."""

from __future__ import annotations

import argparse


def _add_shared_arguments(parser: argparse.ArgumentParser) -> None:
    """Add arguments shared by all visualization subcommands."""
    parser.add_argument(
        "--checkpoint",
        required=True,
        help="Path to vae_resume.pt checkpoint (needs encoder + decoder)",
    )
    parser.add_argument(
        "--spm-model",
        default="data/spm.model",
        help="Path to sentencepiece model (default: data/spm.model)",
    )
    parser.add_argument(
        "--corpus-cache",
        default="data/preprocessed_cache.pt",
        help="Path to tokenized corpus cache (default: data/preprocessed_cache.pt)",
    )
    parser.add_argument(
        "--output-dir",
        default="output/viz",
        help="Output directory for figures (default: output/viz)",
    )
    parser.add_argument(
        "--format",
        choices=["png", "svg", "pdf"],
        default="png",
        help="Output format (default: png)",
    )
    parser.add_argument(
        "--device",
        default="cuda",
        help="Compute device (default: cuda)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=256,
        help="Encoding batch size (default: 256)",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=50000,
        help="Max corpus samples for expensive ops (default: 50000)",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=150,
        help="Output resolution (default: 150)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)",
    )


def register_visualize_group(
    parent_subparsers: argparse._SubParsersAction,
) -> None:
    """Register the ``visualize`` subcommand group with sub-subparsers."""
    viz_parser = parent_subparsers.add_parser(
        "visualize",
        help="Generate publication-quality visualizations",
        description="Visualization tools for the pretrained multilingual VAE decoder.",
    )
    viz_subparsers = viz_parser.add_subparsers(
        title="visualizations",
        description="Available visualization types",
        dest="viz_type",
    )

    # Import and register each visualization subcommand
    from lfm.cli.visualize.tsne import TSNECommand
    from lfm.cli.visualize.clustering import ClusteringCommand
    from lfm.cli.visualize.attention import AttentionCommand
    from lfm.cli.visualize.latent_dims import LatentDimsCommand
    from lfm.cli.visualize.length_dist import LengthDistCommand
    from lfm.cli.visualize.interpolation import InterpolationCommand
    from lfm.cli.visualize.zipf import ZipfCommand
    from lfm.cli.visualize.compositionality import CompositionalityCommand
    from lfm.cli.visualize.smoothness import SmoothnessCommand
    from lfm.cli.visualize.adaptiveness import AdaptivenessCommand
    from lfm.cli.visualize.all import AllCommand

    commands = [
        TSNECommand(),
        ClusteringCommand(),
        AttentionCommand(),
        LatentDimsCommand(),
        LengthDistCommand(),
        InterpolationCommand(),
        ZipfCommand(),
        CompositionalityCommand(),
        SmoothnessCommand(),
        AdaptivenessCommand(),
        AllCommand(),
    ]

    for cmd in commands:
        sub = viz_subparsers.add_parser(
            cmd.name,
            help=cmd.help,
            description=cmd.description,
        )
        _add_shared_arguments(sub)
        cmd.add_arguments(sub)
        sub.set_defaults(command_handler=cmd)

    # Default: print help if no viz_type given
    viz_parser.set_defaults(
        command_handler=type(
            "_VizHelp",
            (),
            {"execute": staticmethod(lambda _args: viz_parser.print_help() or 0)},
        )()
    )

"""LFM command-line interface.

Provides the ``lfm`` entry point with subcommand dispatch.
"""

from __future__ import annotations

import argparse
import logging
import sys


def create_parser() -> argparse.ArgumentParser:
    """Build the top-level argument parser with all subcommands."""
    parser = argparse.ArgumentParser(
        prog="lfm",
        description="LFM — Language Faculty Model CLI",
    )
    subparsers = parser.add_subparsers(
        title="commands",
        description="Available commands",
        dest="command",
    )

    # --- visualize subcommand group ---
    from lfm.cli.visualize import register_visualize_group

    register_visualize_group(subparsers)

    # --- translate subcommand group ---
    from lfm.cli.translate import register_translate_group

    register_translate_group(subparsers)

    # --- dataset subcommand group ---
    from lfm.cli.dataset import register_dataset_group

    register_dataset_group(subparsers)

    # --- explore subcommand group ---
    from lfm.cli.explore import register_explore_group

    register_explore_group(subparsers)

    # --- publish subcommand group ---
    from lfm.cli.publish import register_publish_group

    register_publish_group(subparsers)

    # --- agent subcommand group ---
    from lfm.cli.agent import register_agent_group

    register_agent_group(subparsers)

    # --- train subcommand group ---
    from lfm.cli.train import register_train_group

    register_train_group(subparsers)

    # --- cloud subcommand group ---
    from lfm.cli.cloud import register_cloud_group

    register_cloud_group(subparsers)

    # --- unmt subcommand group ---
    from lfm.cli.unmt import register_unmt_group

    register_unmt_group(subparsers)

    # --- qwen-targets subcommand group ---
    from lfm.cli.qwen_targets import register_qwen_targets_group

    register_qwen_targets_group(subparsers)

    # --- pretrain command ---
    from lfm.cli.pretrain import PretrainCommand

    pretrain_cmd = PretrainCommand()
    pretrain_parser = subparsers.add_parser(
        pretrain_cmd.name,
        help=pretrain_cmd.help,
        description=pretrain_cmd.description,
    )
    pretrain_cmd.add_arguments(pretrain_parser)
    pretrain_parser.set_defaults(command_handler=pretrain_cmd)

    # --- setup command ---
    from lfm.cli.setup import SetupCommand

    setup_cmd = SetupCommand()
    setup_parser = subparsers.add_parser(
        setup_cmd.name,
        help=setup_cmd.help,
        description=setup_cmd.description,
    )
    setup_cmd.add_arguments(setup_parser)
    setup_parser.set_defaults(command_handler=setup_cmd)

    return parser


def main() -> int:
    """CLI entry point."""
    # Force line-buffered stderr so log output appears immediately in pipes
    sys.stderr = open(sys.stderr.fileno(), "w", buffering=1, closefd=False)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(message)s",
        stream=sys.stderr,
    )

    parser = create_parser()
    args = parser.parse_args()

    if not hasattr(args, "command_handler"):
        parser.print_help()
        return 0

    try:
        return args.command_handler.execute(args)
    except KeyboardInterrupt:
        print("\nInterrupted by user", file=sys.stderr)
        return 130
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1

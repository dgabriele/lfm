"""Publish subcommand group for ``lfm publish {model,dataset}``."""

from __future__ import annotations

import argparse

from lfm.cli.base import CLICommand


class PublishModelCommand(CLICommand):
    """Publish a pretrained LFM decoder to HuggingFace Hub."""

    @property
    def name(self) -> str:
        return "model"

    @property
    def help(self) -> str:
        return "Upload pretrained decoder to HuggingFace"

    def add_arguments(self, parser: argparse.ArgumentParser) -> None:
        parser.add_argument(
            "--repo-id",
            required=True,
            help="HuggingFace repo ID (e.g. username/lfm-decoder-v1)",
        )
        parser.add_argument(
            "--model-dir",
            default="data/models/v1",
            help="Directory containing decoder checkpoint (default: data/models/v1)",
        )
        parser.add_argument(
            "--description",
            default="",
            help="Additional description for the model card",
        )
        parser.add_argument(
            "--private",
            action="store_true",
            help="Make the repo private",
        )
        parser.add_argument(
            "--token",
            default=None,
            help="HuggingFace API token (default: use cached login)",
        )

    def execute(self, args: argparse.Namespace) -> int:
        from lfm.publish.model import ModelRelease

        model_dir = self.validate_file_exists(args.model_dir, "Model directory")
        if model_dir is None:
            return 1

        publisher = ModelRelease(
            repo_id=args.repo_id,
            private=args.private,
            token=args.token,
        )
        manifest = publisher.publish(
            model_dir=str(model_dir),
            description=args.description,
        )

        print(f"Published model to: {manifest.result.get('url', '?')}")
        print(f"Manifest saved to: {manifest.save()}")
        return 0


class PublishDatasetCommand(CLICommand):
    """Publish an LFM IPA corpus to HuggingFace Hub."""

    @property
    def name(self) -> str:
        return "dataset"

    @property
    def help(self) -> str:
        return "Upload IPA corpus to HuggingFace"

    def add_arguments(self, parser: argparse.ArgumentParser) -> None:
        parser.add_argument(
            "--repo-id",
            required=True,
            help="HuggingFace repo ID (e.g. username/lfm-ipa-16lang)",
        )
        parser.add_argument(
            "--model-dir",
            default="data/models/v1",
            help="Directory containing preprocessed cache (default: data/models/v1)",
        )
        parser.add_argument(
            "--description",
            default="",
            help="Additional description for the dataset card",
        )
        parser.add_argument(
            "--private",
            action="store_true",
            help="Make the repo private",
        )
        parser.add_argument(
            "--token",
            default=None,
            help="HuggingFace API token (default: use cached login)",
        )

    def execute(self, args: argparse.Namespace) -> int:
        from lfm.publish.dataset import DatasetRelease

        model_dir = self.validate_file_exists(args.model_dir, "Model directory")
        if model_dir is None:
            return 1

        publisher = DatasetRelease(
            repo_id=args.repo_id,
            private=args.private,
            token=args.token,
        )
        manifest = publisher.publish(
            model_dir=str(model_dir),
            description=args.description,
            repo_id=args.repo_id,
        )

        print(f"Published dataset to: {manifest.result.get('url', '?')}")
        print(f"Manifest saved to: {manifest.save()}")
        return 0


def register_publish_group(
    parent_subparsers: argparse._SubParsersAction,
) -> None:
    """Register the ``publish`` subcommand group."""
    publish_parser = parent_subparsers.add_parser(
        "publish",
        help="Publish models and datasets to HuggingFace Hub",
        description="Upload LFM artifacts to HuggingFace Hub with release manifests.",
    )
    publish_subparsers = publish_parser.add_subparsers(
        title="publish commands",
        description="Available publish targets",
        dest="publish_cmd",
    )

    commands = [
        PublishModelCommand(),
        PublishDatasetCommand(),
    ]

    for cmd in commands:
        sub = publish_subparsers.add_parser(
            cmd.name,
            help=cmd.help,
            description=cmd.description,
        )
        cmd.add_arguments(sub)
        sub.set_defaults(command_handler=cmd)

    publish_parser.set_defaults(
        command_handler=type(
            "_PublishHelp",
            (),
            {"execute": staticmethod(lambda _args: publish_parser.print_help() or 0)},
        )()
    )

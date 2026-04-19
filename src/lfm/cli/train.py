"""Train subcommand group for ``lfm train {reconstruction,...}``."""

from __future__ import annotations

import argparse

from lfm.cli.base import CLICommand


class DepTreeVAECommand(CLICommand):
    """Train DepTreeVAE — dependency-disentangled latent space."""

    @property
    def name(self) -> str:
        return "dep-tree-vae"

    @property
    def help(self) -> str:
        return "Pretrain DepTreeVAE (syntax/content disentangled latent space)"

    def add_arguments(self, parser: argparse.ArgumentParser) -> None:
        parser.add_argument("config", help="YAML config file")

    def execute(self, args: argparse.Namespace) -> int:
        import yaml

        from lfm.generator.dep_tree_vae.config import DepTreeVAEConfig
        from lfm.generator.dep_tree_vae.trainer import train_dep_tree_vae

        with open(args.config) as f:
            raw = yaml.safe_load(f)

        if "attention_head_windows" in raw:
            raw["attention_head_windows"] = tuple(raw["attention_head_windows"])

        config = DepTreeVAEConfig(**raw)
        train_dep_tree_vae(config)
        return 0


class ReconstructionCommand(CLICommand):
    """Train reconstruction model — embedding recovery through the bottleneck."""

    @property
    def name(self) -> str:
        return "reconstruction"

    @property
    def help(self) -> str:
        return "Train reconstruction: embedding → IPA → inverse → reconstruct"

    def add_arguments(self, parser: argparse.ArgumentParser) -> None:
        parser.add_argument("config", nargs="?", default=None,
                            help="YAML config file")
        parser.add_argument("--resume", default=None)
        parser.add_argument("--steps", type=int, default=None)
        parser.add_argument("--batch-size", type=int, default=None)
        parser.add_argument("--device", default=None)

    def execute(self, args: argparse.Namespace) -> int:
        import yaml

        from lfm.reconstruction.config import ReconstructionConfig
        from lfm.reconstruction.trainer import ReconstructionTrainer

        cfg_dict: dict = {}
        if args.config is not None:
            with open(args.config) as f:
                cfg_dict = yaml.safe_load(f) or {}

        for key in ("steps", "batch_size", "device"):
            val = getattr(args, key, None)
            if val is not None:
                cfg_dict[key] = val

        config = ReconstructionConfig(**cfg_dict)
        trainer = ReconstructionTrainer(config)
        results = trainer.train(resume=args.resume)
        print(f"Final: cos_sim={results['cosine_sim']:.4f}")
        return 0


def register_train_group(
    parent_subparsers: argparse._SubParsersAction,
) -> None:
    """Register the ``train`` subcommand group."""
    train_parser = parent_subparsers.add_parser(
        "train",
        help="Train subsystems",
        description="Train various LFM subsystems.",
    )
    train_subparsers = train_parser.add_subparsers(
        title="training targets",
        description="Available training targets",
        dest="train_cmd",
    )

    commands = [
        DepTreeVAECommand(),
        ReconstructionCommand(),
    ]

    for cmd in commands:
        sub = train_subparsers.add_parser(
            cmd.name,
            help=cmd.help,
            description=cmd.description,
        )
        cmd.add_arguments(sub)
        sub.set_defaults(command_handler=cmd)

    train_parser.set_defaults(
        command_handler=type(
            "_TrainHelp",
            (),
            {"execute": staticmethod(lambda _args: train_parser.print_help() or 0)},
        )()
    )

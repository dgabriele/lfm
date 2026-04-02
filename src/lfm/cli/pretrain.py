"""Pretrain subcommand for ``lfm pretrain``."""

from __future__ import annotations

import argparse

from lfm.cli.base import CLICommand


class PretrainCommand(CLICommand):
    """Pretrain the VAE decoder from a YAML config file."""

    @property
    def name(self) -> str:
        return "pretrain"

    @property
    def help(self) -> str:
        return "Pretrain VAE decoder from YAML config"

    def add_arguments(self, parser: argparse.ArgumentParser) -> None:
        parser.add_argument(
            "config", help="Path to YAML config file",
        )

    def execute(self, args: argparse.Namespace) -> int:
        import yaml

        from lfm.generator.pretrain import VAEPretrainConfig, pretrain_vae_decoder

        with open(args.config) as f:
            raw = yaml.safe_load(f)

        if "attention_head_windows" in raw:
            raw["attention_head_windows"] = tuple(raw["attention_head_windows"])

        config = VAEPretrainConfig(**raw)
        pretrain_vae_decoder(config)
        return 0

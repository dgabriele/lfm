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

        with open(args.config) as f:
            raw = yaml.safe_load(f)

        if "attention_head_windows" in raw:
            raw["attention_head_windows"] = tuple(raw["attention_head_windows"])

        model_type = raw.pop("model_type", "phrase_vae")

        if model_type == "phrase_vae":
            from lfm.generator.pretrain import VAEPretrainConfig, pretrain_vae_decoder
            config = VAEPretrainConfig(**raw)
            pretrain_vae_decoder(config)
        elif model_type == "dep_tree_vae":
            from lfm.generator.dep_tree_vae.config import DepTreeVAEConfig
            from lfm.generator.dep_tree_vae.trainer import train_dep_tree_vae
            config = DepTreeVAEConfig(**raw)
            train_dep_tree_vae(config)
        elif model_type == "dep_tree_diffusion":
            from lfm.generator.dep_tree_diffusion.config import DepTreeDiffusionConfig
            from lfm.generator.dep_tree_diffusion.trainer import train_dep_tree_diffusion
            config = DepTreeDiffusionConfig(**raw)
            train_dep_tree_diffusion(config)
        else:
            raise ValueError(
                f"Unknown model_type: {raw.get('model_type')!r}. "
                f"Expected: phrase_vae, dep_tree_vae, dep_tree_diffusion"
            )

        return 0

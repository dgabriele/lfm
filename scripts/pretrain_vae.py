"""Pretrain the VAE decoder from a YAML config file.

Usage::

    python scripts/pretrain_vae.py configs/pretrain_vae.yaml
"""

from __future__ import annotations

import argparse
import logging

import yaml

from lfm.generator.pretrain import VAEPretrainConfig, pretrain_vae_decoder

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(message)s",
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Pretrain VAE decoder")
    parser.add_argument("config", help="Path to YAML config file")
    args = parser.parse_args()

    with open(args.config) as f:
        raw = yaml.safe_load(f)

    # Convert tuple fields that YAML loads as lists
    if "attention_head_windows" in raw:
        raw["attention_head_windows"] = tuple(raw["attention_head_windows"])

    config = VAEPretrainConfig(**raw)
    pretrain_vae_decoder(config)


if __name__ == "__main__":
    main()

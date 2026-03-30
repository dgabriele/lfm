#!/usr/bin/env python3
"""Thin wrapper for the referential backprop game.

Prefer ``poetry run lfm agent referential`` for the full CLI experience.
This script provides a quick entry point with the same functionality.
"""

from __future__ import annotations

import argparse
import logging
import sys

# Force line-buffered output
sys.stderr = open(sys.stderr.fileno(), "w", buffering=1, closefd=False)
sys.stdout = open(sys.stdout.fileno(), "w", buffering=1, closefd=False)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(message)s",
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Referential game (backprop)")
    parser.add_argument("--resume", default=None)
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--decoder-path", default="data/vae_decoder.pt")
    parser.add_argument("--spm-path", default="data/spm.model")
    parser.add_argument("--vq-codebook", default=None)
    parser.add_argument("--vq-alpha", type=float, default=1.0)
    parser.add_argument("--num-memory-tokens", type=int, default=8)
    parser.add_argument("--encoder-layers", type=int, default=2)
    parser.add_argument("--encoder-heads", type=int, default=8)
    parser.add_argument("--sender-lr", type=float, default=3e-5)
    parser.add_argument("--receiver-lr", type=float, default=3e-4)
    parser.add_argument("--output-dir", default="data/referential_game")
    args = parser.parse_args()

    import torch

    from lfm.agents.config import CurriculumConfig, MessageEncoderConfig
    from lfm.agents.games.referential import ReferentialGame, ReferentialGameConfig
    from lfm.agents.trainer import AgentTrainer
    from lfm.faculty.model import LanguageFaculty

    config = ReferentialGameConfig(
        decoder_path=args.decoder_path,
        spm_path=args.spm_path,
        vq_codebook_path=args.vq_codebook,
        vq_residual_alpha=args.vq_alpha,
        num_memory_tokens=args.num_memory_tokens,
        encoder=MessageEncoderConfig(
            num_layers=args.encoder_layers,
            num_heads=args.encoder_heads,
        ),
        batch_size=args.batch_size,
        steps=args.steps,
        sender_lr=args.sender_lr,
        receiver_lr=args.receiver_lr,
        output_dir=args.output_dir,
        device=args.device,
    )

    device = torch.device(config.device)
    faculty = LanguageFaculty(config.build_faculty_config()).to(device)
    game = ReferentialGame(config, faculty).to(device)
    trainer = AgentTrainer(game, config)
    trainer.train(resume=args.resume)


if __name__ == "__main__":
    main()

"""Agent subcommand group for ``lfm agent {referential,...}``."""

from __future__ import annotations

import argparse

from lfm.cli.base import CLICommand


class ReferentialCommand(CLICommand):
    """Train the referential backprop game through the linguistic bottleneck."""

    @property
    def name(self) -> str:
        return "referential"

    @property
    def help(self) -> str:
        return "Train referential game with direct backprop through frozen decoder"

    def add_arguments(self, parser: argparse.ArgumentParser) -> None:
        # Paths
        parser.add_argument(
            "--decoder-path", default="data/vae_decoder.pt",
            help="Path to pretrained VAE decoder (default: data/vae_decoder.pt)",
        )
        parser.add_argument(
            "--spm-path", default="data/spm.model",
            help="Path to sentencepiece model (default: data/spm.model)",
        )
        parser.add_argument(
            "--embedding-store", default="data/embeddings",
            help="Path to embedding store directory (default: data/embeddings)",
        )
        parser.add_argument(
            "--vq-codebook", default=None,
            help="Path to VQ codebook (optional)",
        )
        parser.add_argument(
            "--output-dir", default="data/referential_game",
            help="Output directory for checkpoints (default: data/referential_game)",
        )
        parser.add_argument(
            "--resume", default=None,
            help="Resume from checkpoint path",
        )

        # Architecture
        parser.add_argument(
            "--num-memory-tokens", type=int, default=8,
            help="Memory tokens (must match decoder checkpoint, default: 8)",
        )
        parser.add_argument(
            "--encoder-layers", type=int, default=2,
            help="Message encoder self-attention layers (default: 2)",
        )
        parser.add_argument(
            "--encoder-heads", type=int, default=8,
            help="Message encoder attention heads (default: 8)",
        )

        # Training
        parser.add_argument("--steps", type=int, default=2000)
        parser.add_argument("--batch-size", type=int, default=256)
        parser.add_argument("--sender-lr", type=float, default=3e-5)
        parser.add_argument("--receiver-lr", type=float, default=3e-4)
        parser.add_argument("--num-distractors", type=int, default=15)
        parser.add_argument("--max-output-len", type=int, default=96)
        parser.add_argument("--curriculum-warmup", type=int, default=500)
        parser.add_argument("--no-curriculum", action="store_true")
        parser.add_argument("--device", default="cuda")
        parser.add_argument("--seed", type=int, default=42)
        parser.add_argument("--log-every", type=int, default=50)
        parser.add_argument("--checkpoint-every", type=int, default=100)

    def execute(self, args: argparse.Namespace) -> int:
        from lfm.agents.config import CurriculumConfig, MessageEncoderConfig
        from lfm.agents.games.referential import (
            ReferentialGame,
            ReferentialGameConfig,
        )
        from lfm.agents.trainer import AgentTrainer
        from lfm.faculty.model import LanguageFaculty

        config = ReferentialGameConfig(
            decoder_path=args.decoder_path,
            spm_path=args.spm_path,
            embedding_store_dir=args.embedding_store,
            vq_codebook_path=args.vq_codebook,
            output_dir=args.output_dir,
            num_memory_tokens=args.num_memory_tokens,
            max_output_len=args.max_output_len,
            encoder=MessageEncoderConfig(
                num_layers=args.encoder_layers,
                num_heads=args.encoder_heads,
            ),
            batch_size=args.batch_size,
            steps=args.steps,
            sender_lr=args.sender_lr,
            receiver_lr=args.receiver_lr,
            num_distractors=args.num_distractors,
            curriculum=CurriculumConfig(
                enabled=not args.no_curriculum,
                warmup_steps=args.curriculum_warmup,
            ),
            device=args.device,
            seed=args.seed,
            log_every=args.log_every,
            checkpoint_every=args.checkpoint_every,
        )

        import torch
        device = torch.device(config.device)

        faculty_config = config.build_faculty_config()
        faculty = LanguageFaculty(faculty_config).to(device)

        game = ReferentialGame(config, faculty).to(device)
        trainer = AgentTrainer(game, config)
        trainer.train(resume=args.resume)

        return 0


class ExpressionCommand(CLICommand):
    """Train the expression game with GRU z-sequence generation."""

    @property
    def name(self) -> str:
        return "expression"

    @property
    def help(self) -> str:
        return "Train expression game with GRU z-sequence through frozen decoder"

    def add_arguments(self, parser: argparse.ArgumentParser) -> None:
        parser.add_argument("--decoder-path", default="data/vae_decoder.pt")
        parser.add_argument("--spm-path", default="data/spm.model")
        parser.add_argument("--embedding-store", default="data/embeddings")
        parser.add_argument("--vq-codebook", default=None)
        parser.add_argument("--output-dir", default="data/expression_game")
        parser.add_argument("--resume", default=None)
        parser.add_argument("--num-memory-tokens", type=int, default=8)
        parser.add_argument("--z-hidden-dim", type=int, default=512)
        parser.add_argument("--max-segments", type=int, default=8)
        parser.add_argument("--max-tokens-per-segment", type=int, default=48)
        parser.add_argument("--lambda-p", type=float, default=0.4,
                            help="Geometric prior param (E[K]=1/lambda_p, default: 0.4)")
        parser.add_argument("--kl-beta", type=float, default=0.5,
                            help="KL divergence weight (default: 0.5)")
        parser.add_argument("--encoder-layers", type=int, default=2)
        parser.add_argument("--encoder-heads", type=int, default=8)
        parser.add_argument("--steps", type=int, default=2000)
        parser.add_argument("--batch-size", type=int, default=256)
        parser.add_argument("--gradient-accumulation-steps", type=int, default=1,
                            help="Gradient accumulation steps (default: 1)")
        parser.add_argument("--gru-lr", type=float, default=1e-4)
        parser.add_argument("--receiver-lr", type=float, default=3e-4)
        parser.add_argument("--curriculum-warmup", type=int, default=500)
        parser.add_argument("--no-curriculum", action="store_true")
        parser.add_argument("--z-generator", choices=["gru", "diffusion"], default="gru",
                            help="Z-sequence generator type (default: gru)")
        parser.add_argument("--diffusion-steps", type=int, default=4,
                            help="Diffusion reverse steps (default: 4)")
        parser.add_argument("--diffusion-layers", type=int, default=4,
                            help="Diffusion denoiser layers (default: 4)")
        parser.add_argument("--target-segments", type=float, default=2.5,
                            help="Target E[K] for length regularization (default: 2.5)")
        parser.add_argument("--length-weight", type=float, default=0.5,
                            help="Length distribution loss weight (default: 0.5)")
        parser.add_argument("--no-halt", action="store_true",
                            help="Disable PonderNet halting (always use all segments)")
        parser.add_argument("--z-diversity-weight", type=float, default=0.0,
                            help="z diversity regularization weight (default: 0, disabled)")
        parser.add_argument("--z-diversity-target", type=float, default=None,
                            help="Target z similarity (default: auto from pretrained distribution)")
        parser.add_argument("--z-distribution-weight", type=float, default=0.0,
                            help="z distribution matching weight (default: 0, disabled)")
        parser.add_argument("--hidden-state-weight", type=float, default=1.0,
                            help="Initial weight for hidden-state auxiliary loss (default: 1.0)")
        parser.add_argument("--hidden-state-anneal-steps", type=int, default=1000,
                            help="Steps to anneal hidden-state weight to 0 (default: 1000)")
        parser.add_argument("--device", default="cuda")
        parser.add_argument("--seed", type=int, default=42)
        parser.add_argument("--log-every", type=int, default=50)
        parser.add_argument("--checkpoint-every", type=int, default=100)

    def execute(self, args: argparse.Namespace) -> int:
        from lfm.agents.config import CurriculumConfig, MessageEncoderConfig
        from lfm.agents.games.expression import (
            ExpressionGame,
            ExpressionGameConfig,
        )
        from lfm.agents.trainer import AgentTrainer
        from lfm.faculty.model import LanguageFaculty

        config = ExpressionGameConfig(
            decoder_path=args.decoder_path,
            spm_path=args.spm_path,
            embedding_store_dir=args.embedding_store,
            vq_codebook_path=args.vq_codebook,
            output_dir=args.output_dir,
            num_memory_tokens=args.num_memory_tokens,
            z_hidden_dim=args.z_hidden_dim,
            max_segments=args.max_segments,
            max_tokens_per_segment=args.max_tokens_per_segment,
            z_generator=args.z_generator,
            diffusion_steps=args.diffusion_steps,
            diffusion_layers=args.diffusion_layers,
            target_segments=args.target_segments,
            length_weight=args.length_weight,
            use_halt=not args.no_halt,
            lambda_p=args.lambda_p,
            kl_beta=args.kl_beta,
            z_diversity_weight=args.z_diversity_weight,
            z_diversity_target=args.z_diversity_target,
            z_distribution_weight=args.z_distribution_weight,
            hidden_state_weight=args.hidden_state_weight,
            hidden_state_anneal_steps=args.hidden_state_anneal_steps,
            encoder=MessageEncoderConfig(
                num_layers=args.encoder_layers,
                num_heads=args.encoder_heads,
            ),
            batch_size=args.batch_size,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            steps=args.steps,
            gru_lr=args.gru_lr,
            receiver_lr=args.receiver_lr,
            curriculum=CurriculumConfig(
                enabled=not args.no_curriculum,
                warmup_steps=args.curriculum_warmup,
            ),
            device=args.device,
            seed=args.seed,
            log_every=args.log_every,
            checkpoint_every=args.checkpoint_every,
        )

        import torch
        device = torch.device(config.device)

        faculty = LanguageFaculty(config.build_faculty_config()).to(device)
        game = ExpressionGame(config, faculty).to(device)
        trainer = AgentTrainer(game, config)
        trainer.train(resume=args.resume)

        return 0


def register_agent_group(parent_subparsers: argparse._SubParsersAction) -> None:
    """Register the ``lfm agent`` subcommand group."""
    agent_parser = parent_subparsers.add_parser(
        "agent",
        help="Agent communication games",
        description="Train and evaluate agent communication through the linguistic bottleneck.",
    )
    agent_subparsers = agent_parser.add_subparsers(
        title="agent games",
        description="Available agent games",
        dest="agent_cmd",
    )

    commands = [
        ReferentialCommand(),
        ExpressionCommand(),
    ]

    for cmd in commands:
        sub = agent_subparsers.add_parser(
            cmd.name,
            help=cmd.help,
            description=cmd.description,
        )
        cmd.add_arguments(sub)
        sub.set_defaults(command_handler=cmd)

    agent_parser.set_defaults(
        command_handler=type(
            "_AgentHelp",
            (),
            {"execute": staticmethod(lambda _args: agent_parser.print_help() or 0)},
        )()
    )

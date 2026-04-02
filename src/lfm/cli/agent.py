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
        parser.add_argument("config", nargs="?", default=None,
                            help="Optional YAML config file (CLI args override)")
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
        parser.add_argument("--use-ipa-receiver", action="store_true",
                            help="Score based on IPA token representations instead of raw embeddings")
        parser.add_argument("--ipa-cache-refresh", type=int, default=0,
                            help="Refresh IPA cache every N steps (0=never)")
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
        import yaml

        from lfm.agents.config import CurriculumConfig, MessageEncoderConfig
        from lfm.agents.games.expression import (
            ExpressionGame,
            ExpressionGameConfig,
        )
        from lfm.agents.trainer import AgentTrainer
        from lfm.faculty.model import LanguageFaculty

        # Load YAML config if provided, then let CLI args override
        cfg_dict: dict = {}
        if args.config is not None:
            with open(args.config) as f:
                cfg_dict = yaml.safe_load(f) or {}

        def _get(key, cli_val, default=None):
            """YAML value unless CLI explicitly set."""
            if cli_val != default:
                return cli_val
            return cfg_dict.get(key, cli_val)

        config = ExpressionGameConfig(
            decoder_path=_get("decoder_path", args.decoder_path, "data/vae_decoder.pt"),
            spm_path=_get("spm_path", args.spm_path, "data/spm.model"),
            embedding_store_dir=_get("embedding_store_dir", args.embedding_store, "data/embeddings"),
            vq_codebook_path=_get("vq_codebook_path", args.vq_codebook, None),
            output_dir=_get("output_dir", args.output_dir, "data/expression_game"),
            num_memory_tokens=_get("num_memory_tokens", args.num_memory_tokens, 8),
            z_hidden_dim=_get("z_hidden_dim", args.z_hidden_dim, 512),
            max_segments=_get("max_segments", args.max_segments, 8),
            max_tokens_per_segment=_get("max_tokens_per_segment", args.max_tokens_per_segment, 48),
            use_ipa_receiver=_get("use_ipa_receiver", args.use_ipa_receiver, False),
            ipa_cache_refresh=_get("ipa_cache_refresh", args.ipa_cache_refresh, 0),
            z_generator=_get("z_generator", args.z_generator, "gru"),
            diffusion_steps=_get("diffusion_steps", args.diffusion_steps, 4),
            diffusion_layers=_get("diffusion_layers", args.diffusion_layers, 4),
            target_segments=_get("target_segments", args.target_segments, 2.5),
            length_weight=_get("length_weight", args.length_weight, 0.5),
            use_halt=_get("use_halt", not args.no_halt, True),
            lambda_p=_get("lambda_p", args.lambda_p, 0.4),
            kl_beta=_get("kl_beta", args.kl_beta, 0.5),
            z_diversity_weight=_get("z_diversity_weight", args.z_diversity_weight, 0.0),
            z_diversity_target=_get("z_diversity_target", args.z_diversity_target, None),
            z_distribution_weight=_get("z_distribution_weight", args.z_distribution_weight, 0.0),
            hidden_state_weight=_get("hidden_state_weight", args.hidden_state_weight, 1.0),
            hidden_state_anneal_steps=_get("hidden_state_anneal_steps", args.hidden_state_anneal_steps, 1000),
            batch_size=_get("batch_size", args.batch_size, 256),
            gradient_accumulation_steps=_get("gradient_accumulation_steps", args.gradient_accumulation_steps, 1),
            steps=_get("steps", args.steps, 2000),
            gru_lr=_get("gru_lr", args.gru_lr, 1e-4),
            receiver_lr=_get("receiver_lr", args.receiver_lr, 3e-4),
            device=_get("device", args.device, "cuda"),
            seed=_get("seed", args.seed, 42),
            log_every=_get("log_every", args.log_every, 50),
            checkpoint_every=_get("checkpoint_every", args.checkpoint_every, 100),
            encoder=MessageEncoderConfig(
                num_layers=_get("encoder_layers", args.encoder_layers, 2),
                num_heads=_get("encoder_heads", args.encoder_heads, 8),
            ),
            curriculum=CurriculumConfig(
                enabled=not _get("no_curriculum", args.no_curriculum, False),
                warmup_steps=_get("curriculum_warmup", args.curriculum_warmup, 500),
            ),
        )

        resume = _get("resume", args.resume, None)

        import torch
        device = torch.device(config.device)

        faculty = LanguageFaculty(config.build_faculty_config()).to(device)
        game = ExpressionGame(config, faculty).to(device)
        trainer = AgentTrainer(game, config)
        trainer.train(resume=resume)

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

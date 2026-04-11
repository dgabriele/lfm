"""Agent subcommand group for ``lfm agent {referential,...}``."""

from __future__ import annotations

import argparse
import logging

logger = logging.getLogger(__name__)

from lfm.cli.base import CLICommand


def _auto_detect_embedding_dim(
    cfg_dict: dict, store_dir_key: str = "embedding_store_dir",
) -> dict:
    """Sync ``embedding_dim`` in a config dict with its embedding store.

    Reads ``metadata.json`` from the configured store directory and sets
    ``cfg_dict["embedding_dim"]`` to match — but only when the user has
    NOT explicitly specified a value in the config.  Explicit user
    values are always respected so experiments that want a mismatched
    dim (e.g. for projection layers) can opt in by setting it directly.

    This keeps the dialogue game adaptive: pointing at a new embedding
    store with a different native dim requires no config changes.
    """
    from lfm.embeddings.store import EmbeddingStore

    store_dir = cfg_dict.get(store_dir_key)
    if not store_dir:
        return cfg_dict
    if "embedding_dim" in cfg_dict:
        return cfg_dict  # respect explicit override
    try:
        meta = EmbeddingStore.read_metadata(store_dir)
    except FileNotFoundError:
        return cfg_dict
    detected = meta.get("embedding_dim")
    if detected is None:
        return cfg_dict
    logger.info(
        "Auto-detected embedding_dim=%d from store at %s",
        detected, store_dir,
    )
    return {**cfg_dict, "embedding_dim": int(detected)}


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

        extra_kwargs: dict = {}
        from lfm.embeddings.store import EmbeddingStore
        try:
            meta = EmbeddingStore.read_metadata(args.embedding_store)
            if "embedding_dim" in meta:
                extra_kwargs["embedding_dim"] = int(meta["embedding_dim"])
                logger.info(
                    "Auto-detected embedding_dim=%d from store at %s",
                    extra_kwargs["embedding_dim"], args.embedding_store,
                )
        except FileNotFoundError:
            pass

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
        parser.add_argument("--max-phrases", type=int, default=8)
        parser.add_argument("--max-tokens-per-phrase", type=int, default=48)
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
        parser.add_argument("--target-phrases", type=float, default=2.5,
                            help="Target E[K] for length regularization (default: 2.5)")
        parser.add_argument("--length-weight", type=float, default=0.5,
                            help="Length distribution loss weight (default: 0.5)")
        parser.add_argument("--phase2-mode", choices=["decoder", "refinement"], default="decoder",
                            help="Phase 2 mode: 'decoder' (frozen re-run) or 'refinement' (diffusion denoiser)")
        parser.add_argument("--refinement-layers", type=int, default=4)
        parser.add_argument("--refinement-steps", type=int, default=4)
        parser.add_argument("--use-ipa-receiver", action="store_true",
                            help="Score based on IPA token representations instead of raw embeddings")
        parser.add_argument("--ipa-cache-refresh", type=int, default=0,
                            help="Refresh IPA cache every N steps (0=never)")
        parser.add_argument("--no-halt", action="store_true",
                            help="Disable PonderNet halting (always use all phrases)")
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

        resolved_store = _get("embedding_store_dir", args.embedding_store, "data/embeddings")
        # Auto-detect embedding_dim from store unless YAML explicitly sets it.
        extra_kwargs: dict = {}
        if "embedding_dim" not in cfg_dict:
            from lfm.embeddings.store import EmbeddingStore
            try:
                meta = EmbeddingStore.read_metadata(resolved_store)
                if "embedding_dim" in meta:
                    extra_kwargs["embedding_dim"] = int(meta["embedding_dim"])
                    logger.info(
                        "Auto-detected embedding_dim=%d from store at %s",
                        extra_kwargs["embedding_dim"], resolved_store,
                    )
            except FileNotFoundError:
                pass
        else:
            extra_kwargs["embedding_dim"] = cfg_dict["embedding_dim"]

        config = ExpressionGameConfig(
            decoder_path=_get("decoder_path", args.decoder_path, "data/vae_decoder.pt"),
            spm_path=_get("spm_path", args.spm_path, "data/spm.model"),
            embedding_store_dir=resolved_store,
            vq_codebook_path=_get("vq_codebook_path", args.vq_codebook, None),
            output_dir=_get("output_dir", args.output_dir, "data/expression_game"),
            num_memory_tokens=_get("num_memory_tokens", args.num_memory_tokens, 8),
            z_hidden_dim=_get("z_hidden_dim", args.z_hidden_dim, 512),
            max_phrases=_get("max_phrases", args.max_phrases, 8),
            max_tokens_per_phrase=_get("max_tokens_per_phrase", args.max_tokens_per_phrase, 48),
            phase2_mode=_get("phase2_mode", args.phase2_mode, "decoder"),
            refinement_layers=_get("refinement_layers", args.refinement_layers, 4),
            refinement_steps=_get("refinement_steps", args.refinement_steps, 4),
            use_ipa_receiver=_get("use_ipa_receiver", args.use_ipa_receiver, False),
            ipa_cache_refresh=_get("ipa_cache_refresh", args.ipa_cache_refresh, 0),
            z_generator=_get("z_generator", args.z_generator, "gru"),
            diffusion_steps=_get("diffusion_steps", args.diffusion_steps, 4),
            diffusion_layers=_get("diffusion_layers", args.diffusion_layers, 4),
            target_phrases=_get("target_phrases", args.target_phrases, 2.5),
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
            **extra_kwargs,
        )

        resume = _get("resume", args.resume, None)

        import torch
        device = torch.device(config.device)

        faculty = LanguageFaculty(config.build_faculty_config()).to(device)
        game = ExpressionGame(config, faculty).to(device)
        trainer = AgentTrainer(game, config)
        trainer.train(resume=resume)

        return 0


class DialogueCommand(CLICommand):
    """Train dialogue game with multi-turn self-play."""

    @property
    def name(self) -> str:
        return "dialogue"

    @property
    def help(self) -> str:
        return "Train multi-turn dialogue game through frozen decoder"

    def add_arguments(self, parser: argparse.ArgumentParser) -> None:
        parser.add_argument("config", nargs="?", default=None,
                            help="Optional YAML config file")
        parser.add_argument("--decoder-path", default="data/vae_decoder.pt")
        parser.add_argument("--spm-path", default="data/spm.model")
        parser.add_argument("--embedding-store", default="data/embeddings")
        parser.add_argument("--output-dir", default="data/dialogue_game")
        parser.add_argument("--resume", default=None)
        parser.add_argument("--num-turns", type=int, default=4)
        parser.add_argument("--max-phrases", type=int, default=4)
        parser.add_argument("--batch-size", type=int, default=20)
        parser.add_argument("--gradient-accumulation-steps", type=int, default=12)
        parser.add_argument("--steps", type=int, default=4000)
        parser.add_argument("--device", default="cuda")
        parser.add_argument("--seed", type=int, default=42)
        parser.add_argument("--log-every", type=int, default=50)
        parser.add_argument("--checkpoint-every", type=int, default=100)

    def execute(self, args: argparse.Namespace) -> int:
        import yaml

        from lfm.agents.config import CurriculumConfig
        from lfm.agents.games.dialogue import DialogueGame, DialogueGameConfig
        from lfm.agents.trainer import AgentTrainer
        from lfm.faculty.model import LanguageFaculty

        cfg_dict: dict = {}
        if args.config is not None:
            with open(args.config) as f:
                cfg_dict = yaml.safe_load(f) or {}

        # Build config from YAML with CLI overrides.
        # All YAML fields are passed through to DialogueGameConfig;
        # CLI args override when explicitly set.
        cli_overrides = {}
        cli_map = {
            "decoder_path": ("decoder_path", "data/vae_decoder.pt"),
            "spm_path": ("spm_path", "data/spm.model"),
            "embedding_store": ("embedding_store_dir", "data/embeddings"),
            "output_dir": ("output_dir", "data/dialogue_game"),
            "num_turns": ("num_turns", 4),
            "max_phrases": ("max_phrases", 4),
            "batch_size": ("batch_size", 20),
            "gradient_accumulation_steps": ("gradient_accumulation_steps", 12),
            "steps": ("steps", 4000),
            "device": ("device", "cuda"),
            "seed": ("seed", 42),
            "log_every": ("log_every", 50),
            "checkpoint_every": ("checkpoint_every", 100),
        }
        for arg_name, (cfg_name, default) in cli_map.items():
            val = getattr(args, arg_name, default)
            if val != default:
                cli_overrides[cfg_name] = val

        # Handle curriculum from YAML
        curriculum_kwargs = {}
        for ck in ("enabled", "warmup_steps", "start_hard_ratio", "end_hard_ratio", "medium_ratio"):
            if ck in cfg_dict:
                curriculum_kwargs[ck] = cfg_dict.pop(ck)
        if getattr(args, "no_curriculum", False):
            curriculum_kwargs["enabled"] = False
        elif "no_curriculum" in cfg_dict:
            curriculum_kwargs["enabled"] = not cfg_dict.pop("no_curriculum")
        if curriculum_kwargs:
            cfg_dict["curriculum"] = CurriculumConfig(**curriculum_kwargs)

        import torch
        resume = cfg_dict.get("resume", None) if args.resume is None else args.resume

        # On resume, restore training config from checkpoint to prevent
        # regime shifts.  Only operational fields (batch_size, device,
        # log_every, checkpoint_every) are taken from the YAML/CLI.
        if resume is not None:
            ckpt = torch.load(resume, map_location="cpu", weights_only=False)
            if "training_config" in ckpt:
                saved_cfg = ckpt["training_config"]
                # Operational overrides that are safe to change on resume
                operational = {
                    "batch_size", "gradient_accumulation_steps", "device",
                    "log_every", "checkpoint_every", "output_dir",
                    "phase2_vram_budget_mb", "phase2_min_chunk",
                    "embedding_store_dir",
                }
                merged = dict(saved_cfg)
                for key in operational:
                    if key in cfg_dict:
                        merged[key] = cfg_dict[key]
                    if key in cli_overrides:
                        merged[key] = cli_overrides[key]
                # Restore curriculum from saved config
                if "curriculum" in merged and isinstance(merged["curriculum"], dict):
                    merged["curriculum"] = CurriculumConfig(**merged["curriculum"])
                merged = _auto_detect_embedding_dim(merged)
                config = DialogueGameConfig(**merged)
                logger.info("Restored training config from checkpoint (step %d)", ckpt.get("step", 0))
            else:
                merged = _auto_detect_embedding_dim({**cfg_dict, **cli_overrides})
                config = DialogueGameConfig(**merged)
            del ckpt
        else:
            merged = _auto_detect_embedding_dim({**cfg_dict, **cli_overrides})
            config = DialogueGameConfig(**merged)

        device = torch.device(config.device)
        faculty = LanguageFaculty(config.build_faculty_config()).to(device)
        game = DialogueGame(config, faculty).to(device)
        trainer = AgentTrainer(game, config)
        trainer.train(resume=resume)
        return 0


class MultiViewCommand(CLICommand):
    """Train the multi-view game — multiple complementary views per embedding."""

    @property
    def name(self) -> str:
        return "multiview"

    @property
    def help(self) -> str:
        return "Train multi-view game: N independent expressions per embedding"

    def add_arguments(self, parser: argparse.ArgumentParser) -> None:
        parser.add_argument("config", nargs="?", default=None)
        parser.add_argument("--resume", default=None)
        parser.add_argument("--steps", type=int, default=None)
        parser.add_argument("--batch-size", type=int, default=None)
        parser.add_argument("--device", default=None)

    def execute(self, args: argparse.Namespace) -> int:
        import yaml

        from lfm.agents.games.multiview import MultiViewGame, MultiViewGameConfig
        from lfm.agents.trainer import AgentTrainer
        from lfm.faculty.model import LanguageFaculty

        cfg_dict: dict = {}
        if args.config is not None:
            with open(args.config) as f:
                cfg_dict = yaml.safe_load(f) or {}

        # Handle curriculum
        curriculum_kwargs = {}
        for ck in ("enabled", "warmup_steps", "start_hard_ratio",
                    "end_hard_ratio", "medium_ratio"):
            if ck in cfg_dict:
                from lfm.agents.config import CurriculumConfig
                curriculum_kwargs[ck] = cfg_dict.pop(ck)
        if "no_curriculum" in cfg_dict:
            curriculum_kwargs["enabled"] = not cfg_dict.pop("no_curriculum")
        if curriculum_kwargs:
            from lfm.agents.config import CurriculumConfig
            cfg_dict["curriculum"] = CurriculumConfig(**curriculum_kwargs)

        for key in ("steps", "batch_size", "device"):
            val = getattr(args, key, None)
            if val is not None:
                cfg_dict[key] = val

        import torch
        cfg_dict = _auto_detect_embedding_dim(cfg_dict)
        config = MultiViewGameConfig(**cfg_dict)
        device = torch.device(config.device)
        faculty = LanguageFaculty(config.build_faculty_config()).to(device)
        game = MultiViewGame(config, faculty).to(device)
        trainer = AgentTrainer(game, config)
        trainer.train(resume=args.resume)
        return 0


class DocumentCommand(CLICommand):
    """Train the document game — multi-phrase expression generation."""

    @property
    def name(self) -> str:
        return "document"

    @property
    def help(self) -> str:
        return "Train document game: multi-phrase expressions through frozen decoder"

    def add_arguments(self, parser: argparse.ArgumentParser) -> None:
        parser.add_argument("config", nargs="?", default=None,
                            help="YAML config file")
        parser.add_argument("--resume", default=None)
        parser.add_argument("--steps", type=int, default=None)
        parser.add_argument("--batch-size", type=int, default=None)
        parser.add_argument("--device", default=None)

    def execute(self, args: argparse.Namespace) -> int:
        import yaml

        from lfm.agents.games.document import DocumentGame, DocumentGameConfig
        from lfm.agents.trainer import AgentTrainer
        from lfm.faculty.model import LanguageFaculty

        cfg_dict: dict = {}
        if args.config is not None:
            with open(args.config) as f:
                cfg_dict = yaml.safe_load(f) or {}

        # Handle curriculum from YAML
        curriculum_kwargs = {}
        for ck in ("enabled", "warmup_steps", "start_hard_ratio",
                    "end_hard_ratio", "medium_ratio"):
            if ck in cfg_dict:
                from lfm.agents.config import CurriculumConfig
                curriculum_kwargs[ck] = cfg_dict.pop(ck)
        if "no_curriculum" in cfg_dict:
            curriculum_kwargs["enabled"] = not cfg_dict.pop("no_curriculum")
        if curriculum_kwargs:
            from lfm.agents.config import CurriculumConfig
            cfg_dict["curriculum"] = CurriculumConfig(**curriculum_kwargs)

        # CLI overrides
        for key in ("steps", "batch_size", "device"):
            val = getattr(args, key, None)
            if val is not None:
                cfg_dict[key] = val

        import torch
        cfg_dict = _auto_detect_embedding_dim(cfg_dict)
        config = DocumentGameConfig(**cfg_dict)
        device = torch.device(config.device)
        faculty = LanguageFaculty(config.build_faculty_config()).to(device)
        game = DocumentGame(config, faculty).to(device)
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
        MultiViewCommand(),
        DialogueCommand(),
        DocumentCommand(),
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

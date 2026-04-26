"""Agent subcommand group: ``lfm agent {referential, contrastive}``."""

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
    """
    from lfm.embeddings.store import EmbeddingStore

    store_dir = cfg_dict.get(store_dir_key)
    if not store_dir:
        return cfg_dict
    if "embedding_dim" in cfg_dict:
        return cfg_dict
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
        parser.add_argument("--decoder-path", default="data/vae_decoder.pt")
        parser.add_argument("--spm-path", default="data/spm.model")
        parser.add_argument("--embedding-store", default="data/embeddings")
        parser.add_argument("--vq-codebook", default=None)
        parser.add_argument("--output-dir", default="data/referential_game")
        parser.add_argument("--resume", default=None)
        parser.add_argument("--num-memory-tokens", type=int, default=8)
        parser.add_argument("--encoder-layers", type=int, default=2)
        parser.add_argument("--encoder-heads", type=int, default=8)
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
            ReferentialGame, ReferentialGameConfig,
        )
        from lfm.agents.trainer import AgentTrainer
        from lfm.embeddings.store import EmbeddingStore
        from lfm.faculty.model import LanguageFaculty

        extra: dict = {}
        try:
            meta = EmbeddingStore.read_metadata(args.embedding_store)
            if "embedding_dim" in meta:
                extra["embedding_dim"] = int(meta["embedding_dim"])
                logger.info(
                    "Auto-detected embedding_dim=%d from %s",
                    extra["embedding_dim"], args.embedding_store,
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
            **extra,
        )

        import torch
        device = torch.device(config.device)
        faculty = LanguageFaculty(config.build_faculty_config()).to(device)
        game = ReferentialGame(config, faculty).to(device)
        trainer = AgentTrainer(game, config)
        trainer.train(resume=args.resume)
        return 0


class ContrastiveCommand(CLICommand):
    """Train the contrastive expression game with the six-term loss."""

    @property
    def name(self) -> str:
        return "contrastive"

    @property
    def help(self) -> str:
        return "Train contrastive game (single-Expression, surface-grounded)"

    def add_arguments(self, parser: argparse.ArgumentParser) -> None:
        parser.add_argument(
            "config", nargs="?", default=None, help="YAML config file",
        )
        parser.add_argument("--resume", default=None)
        parser.add_argument("--steps", type=int, default=None)
        parser.add_argument("--batch-size", type=int, default=None)
        parser.add_argument("--device", default=None)

    def execute(self, args: argparse.Namespace) -> int:
        import yaml

        from lfm.agents.config import CurriculumConfig
        from lfm.agents.games.contrastive import (
            ContrastiveGame, ContrastiveGameConfig,
        )
        from lfm.agents.trainer import AgentTrainer
        from lfm.generator.dep_tree_vae.config import DepTreeVAEConfig
        from lfm.generator.dep_tree_vae.model import DepTreeVAE

        cfg_dict: dict = {}
        if args.config is not None:
            with open(args.config) as f:
                cfg_dict = yaml.safe_load(f) or {}

        # Curriculum block: lift from top-level into nested CurriculumConfig.
        curriculum_kwargs = {}
        for ck in (
            "enabled", "warmup_steps",
            "start_hard_ratio", "end_hard_ratio", "medium_ratio",
        ):
            if ck in cfg_dict:
                curriculum_kwargs[ck] = cfg_dict.pop(ck)
        if "no_curriculum" in cfg_dict:
            curriculum_kwargs["enabled"] = not cfg_dict.pop("no_curriculum")
        if curriculum_kwargs:
            cfg_dict["curriculum"] = CurriculumConfig(**curriculum_kwargs)

        for key in ("steps", "batch_size", "device"):
            val = getattr(args, key, None)
            if val is not None:
                cfg_dict[key] = val

        cfg_dict = _auto_detect_embedding_dim(cfg_dict)
        config = ContrastiveGameConfig(**cfg_dict)

        import torch
        device = torch.device(config.device)

        # Build a frozen DepTreeVAE from its own checkpoint.  The vae's
        # trainer-side config (latent dims, decoder shape, etc.) lives
        # in a separate YAML so the game config doesn't have to mirror
        # every architectural knob.
        vae_cfg_dict = yaml.safe_load(open(config.vae_config))
        vae_cfg_dict.pop("model_type", None)
        vae_cfg_dict["spm_model_path"] = config.spm_path
        vae_cfg_dict["output_dir"] = config.output_dir
        vae_cfg_dict["corpus_unigram_path"] = ""  # vae-internal KL; agent uses its own

        # Sniff the checkpoint to align optional-head config with what's
        # actually saved.  DepTreeVAEConfig is frozen, so we adjust the
        # dict before construction rather than mutating the instance.
        ckpt = torch.load(config.vae_checkpoint, map_location=device, weights_only=False)
        state = ckpt["model_state"]
        if not any(k.startswith("length_head") for k in state):
            vae_cfg_dict["length_pred_weight"] = 0.0
            vae_cfg_dict["use_predicted_length_at_decode"] = False
        if not any(k.startswith("tokens_per_role_head") for k in state):
            vae_cfg_dict["tokens_per_role_weight"] = 0.0
        vae_cfg_dict["use_decoder_role_emb"] = any(
            k.startswith("decoder_role_emb") for k in state
        )
        vae_cfg = DepTreeVAEConfig(**vae_cfg_dict)

        # Vocab size matches the cache-derived 8050 the trainer uses.
        vae = DepTreeVAE(vae_cfg, vocab_size=vae_cfg.spm_vocab_size + 50).to(device)
        vae.load_state_dict(state, strict=False)
        logger.info(
            "Loaded vae from %s @ step %s val_best=%.4f",
            config.vae_checkpoint, ckpt.get("global_step", "?"),
            ckpt.get("best_val_loss", float("nan")),
        )

        game = ContrastiveGame(config, vae).to(device)
        trainer = AgentTrainer(game, config)
        trainer.train(resume=args.resume)
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

        from lfm.agents.config import CurriculumConfig
        from lfm.agents.games.multiview import MultiViewGame, MultiViewGameConfig
        from lfm.agents.trainer import AgentTrainer
        from lfm.faculty.model import LanguageFaculty

        cfg_dict: dict = {}
        if args.config is not None:
            with open(args.config) as f:
                cfg_dict = yaml.safe_load(f) or {}

        curriculum_kwargs = {}
        for ck in (
            "enabled", "warmup_steps",
            "start_hard_ratio", "end_hard_ratio", "medium_ratio",
        ):
            if ck in cfg_dict:
                curriculum_kwargs[ck] = cfg_dict.pop(ck)
        if "no_curriculum" in cfg_dict:
            curriculum_kwargs["enabled"] = not cfg_dict.pop("no_curriculum")
        if curriculum_kwargs:
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
        parser.add_argument("config", nargs="?", default=None)
        parser.add_argument("--resume", default=None)
        parser.add_argument("--steps", type=int, default=None)
        parser.add_argument("--batch-size", type=int, default=None)
        parser.add_argument("--device", default=None)

    def execute(self, args: argparse.Namespace) -> int:
        import yaml

        from lfm.agents.config import CurriculumConfig
        from lfm.agents.games.document import DocumentGame, DocumentGameConfig
        from lfm.agents.trainer import AgentTrainer
        from lfm.faculty.model import LanguageFaculty

        cfg_dict: dict = {}
        if args.config is not None:
            with open(args.config) as f:
                cfg_dict = yaml.safe_load(f) or {}

        curriculum_kwargs = {}
        for ck in (
            "enabled", "warmup_steps",
            "start_hard_ratio", "end_hard_ratio", "medium_ratio",
        ):
            if ck in cfg_dict:
                curriculum_kwargs[ck] = cfg_dict.pop(ck)
        if "no_curriculum" in cfg_dict:
            curriculum_kwargs["enabled"] = not cfg_dict.pop("no_curriculum")
        if curriculum_kwargs:
            cfg_dict["curriculum"] = CurriculumConfig(**curriculum_kwargs)

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
        ContrastiveCommand(),
        MultiViewCommand(),
        DocumentCommand(),
    ]

    for cmd in commands:
        sub = agent_subparsers.add_parser(
            cmd.name, help=cmd.help, description=cmd.description,
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

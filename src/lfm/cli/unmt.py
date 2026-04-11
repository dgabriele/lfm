"""``lfm unmt`` subcommand group — unsupervised NMT pipeline.

Stage-based commands that mirror the pipeline phases:

* ``tokenize`` — train the two per-language sentencepiece models.
* ``embed`` — train monolingual fasttext for each language (Stage 2).
* ``align`` — MUSE Procrustes alignment between the two embedding
  spaces (Stage 2).
* ``train`` — DAE + backtranslation seq2seq training (Stages 3-4).
* ``translate`` — apply a trained model to new input (Stage 4).

Only ``tokenize`` is wired in this initial commit; the others are
registered as stubs that print a helpful "not implemented yet" message
so the CLI surface is stable from the start.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import yaml

from lfm.cli.base import CLICommand
from lfm.unmt.config import UNMTConfig

logger = logging.getLogger(__name__)


def _load_config(config_path: str | None) -> UNMTConfig:
    """Load a ``UNMTConfig`` from YAML, falling back to defaults."""
    if config_path is None:
        return UNMTConfig()
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {path}")
    with open(path) as f:
        cfg_dict = yaml.safe_load(f) or {}
    return UNMTConfig(**cfg_dict)


class TokenizeCommand(CLICommand):
    """Train the two per-language sentencepiece tokenizers."""

    @property
    def name(self) -> str:
        return "tokenize"

    @property
    def help(self) -> str:
        return "Train Neuroglot + English sentencepiece tokenizers"

    @property
    def description(self) -> str:
        return (
            "Train two per-language BPE sentencepiece tokenizers over "
            "the Neuroglot and English corpora specified in the config. "
            "Existing trained models are reused if present."
        )

    def add_arguments(self, parser: argparse.ArgumentParser) -> None:
        parser.add_argument(
            "config", nargs="?", default=None,
            help="YAML config file (default: use package defaults)",
        )

    def execute(self, args: argparse.Namespace) -> int:
        from lfm.unmt.tokenizer import train_tokenizers, load_tokenizer

        config = _load_config(args.config)
        artifacts = train_tokenizers(config)
        logger.info("Tokenizer artifacts:")
        logger.info("  Neuroglot: %s", artifacts.neuroglot_model)
        logger.info("  English:   %s", artifacts.english_model)

        tokenizer = load_tokenizer(config)
        ng_start, ng_end = tokenizer.neuroglot_range
        en_start, en_end = tokenizer.english_range
        logger.info(
            "Global vocabulary layout: size=%d",
            tokenizer.global_vocab_size,
        )
        logger.info(
            "  specials   [0 .. %d)", ng_start,
        )
        logger.info(
            "  neuroglot  [%d .. %d)  (%d BPE units)",
            ng_start, ng_end, ng_end - ng_start,
        )
        logger.info(
            "  english    [%d .. %d)  (%d BPE units)",
            en_start, en_end, en_end - en_start,
        )
        return 0


class EmbedCommand(CLICommand):
    """Train monolingual skip-gram embeddings for both languages."""

    @property
    def name(self) -> str:
        return "embed"

    @property
    def help(self) -> str:
        return "Train monolingual skip-gram BPE embeddings for each language"

    @property
    def description(self) -> str:
        return (
            "Train skip-gram with negative sampling on BPE-tokenized "
            "Neuroglot and English corpora.  Produces one embedding "
            "matrix per language, saved under the config output_dir.  "
            "These matrices are the input to MUSE alignment."
        )

    def add_arguments(self, parser: argparse.ArgumentParser) -> None:
        parser.add_argument(
            "config", nargs="?", default=None,
            help="YAML config file",
        )
        parser.add_argument("--embed-dim", type=int, default=None,
                            help="Override model_dim as the embedding size")
        parser.add_argument("--window", type=int, default=5)
        parser.add_argument("--neg-samples", type=int, default=5)
        parser.add_argument("--subsample-t", type=float, default=1e-4)
        parser.add_argument("--batch-size", type=int, default=8192)
        parser.add_argument("--steps", type=int, default=50_000)
        parser.add_argument("--lr", type=float, default=1e-3)
        parser.add_argument("--max-lines", type=int, default=200_000)

    def execute(self, args: argparse.Namespace) -> int:
        from lfm.unmt.embeddings.skipgram import train_embeddings

        config = _load_config(args.config)
        artifacts = train_embeddings(
            config,
            embed_dim=args.embed_dim,
            window=args.window,
            neg_samples=args.neg_samples,
            subsample_t=args.subsample_t,
            batch_size=args.batch_size,
            total_steps=args.steps,
            lr=args.lr,
            max_lines_per_language=args.max_lines,
        )
        logger.info("Embedding artifacts:")
        logger.info("  Neuroglot: %s", artifacts.neuroglot_embeddings)
        logger.info("  English:   %s", artifacts.english_embeddings)
        return 0


class AlignCommand(CLICommand):
    """Run MUSE adversarial + Procrustes alignment on the embeddings."""

    @property
    def name(self) -> str:
        return "align"

    @property
    def help(self) -> str:
        return "Align Neuroglot and English embedding spaces via MUSE"

    @property
    def description(self) -> str:
        return (
            "Learn an orthogonal rotation W such that source-language "
            "embeddings mapped by W land near semantically-similar "
            "target-language embeddings.  Uses adversarial training "
            "followed by iterative CSLS-refined Procrustes.  Operates "
            "on the monolingual embeddings produced by `lfm unmt embed`."
        )

    def add_arguments(self, parser: argparse.ArgumentParser) -> None:
        parser.add_argument(
            "config", nargs="?", default=None,
            help="YAML config file",
        )
        parser.add_argument("--source-lang", default="ng", choices=["ng", "en"])
        parser.add_argument("--target-lang", default="en", choices=["ng", "en"])
        parser.add_argument("--adv-epochs", type=int, default=5)
        parser.add_argument("--adv-epoch-size", type=int, default=1_000_000)
        parser.add_argument("--adv-batch-size", type=int, default=128)
        parser.add_argument("--adv-lr", type=float, default=0.1)
        parser.add_argument("--refine-rounds", type=int, default=5)
        parser.add_argument("--dico-max-rank", type=int, default=15_000)
        parser.add_argument("--k-csls", type=int, default=10)

    def execute(self, args: argparse.Namespace) -> int:
        from lfm.unmt.embeddings.muse_align import run_muse_alignment

        if args.source_lang == args.target_lang:
            print("Error: source and target languages must differ")
            return 1

        config = _load_config(args.config)
        result = run_muse_alignment(
            config,
            source_lang=args.source_lang,
            target_lang=args.target_lang,
            n_adv_epochs=args.adv_epochs,
            adv_epoch_size=args.adv_epoch_size,
            adv_batch_size=args.adv_batch_size,
            adv_lr=args.adv_lr,
            n_refine_rounds=args.refine_rounds,
            dico_max_rank=args.dico_max_rank,
            k_csls=args.k_csls,
        )
        logger.info("Alignment complete:")
        logger.info("  direction: %s → %s", result.source_lang, result.target_lang)
        logger.info("  W shape:   %s", tuple(result.W.shape))
        logger.info("  adv loss:  %.4f", result.adversarial_loss)
        logger.info("  refine rounds: %d", result.refinement_rounds)
        logger.info("  final mean CSLS: %.4f", result.final_csls_mean)
        return 0


class _NotImplementedCommand(CLICommand):
    """Stub for commands scheduled for later stages."""

    def __init__(self, name: str, help_text: str, stage: str) -> None:
        self._name = name
        self._help = help_text
        self._stage = stage

    @property
    def name(self) -> str:
        return self._name

    @property
    def help(self) -> str:
        return self._help

    def add_arguments(self, parser: argparse.ArgumentParser) -> None:
        parser.add_argument(
            "config", nargs="?", default=None,
            help="YAML config file",
        )

    def execute(self, args: argparse.Namespace) -> int:
        print(
            f"`lfm unmt {self._name}` is not implemented yet — scheduled for {self._stage}.",
        )
        return 1


def register_unmt_group(
    parent_subparsers: argparse._SubParsersAction,
) -> None:
    """Register the ``unmt`` subcommand group with sub-subparsers."""
    unmt_parser = parent_subparsers.add_parser(
        "unmt",
        help="Unsupervised NMT pipeline (Neuroglot ↔ English)",
        description=(
            "Build and train an unsupervised neural machine translator "
            "between Neuroglot and English using monolingual corpora "
            "only — no paired examples."
        ),
    )
    unmt_subparsers = unmt_parser.add_subparsers(
        title="unmt commands",
        description="Pipeline stages",
        dest="unmt_cmd",
    )

    commands: list[CLICommand] = [
        TokenizeCommand(),
        EmbedCommand(),
        AlignCommand(),
        _NotImplementedCommand(
            "train",
            "Shared-weight seq2seq DAE + backtranslation training (Stage 4)",
            "Stage 4",
        ),
        _NotImplementedCommand(
            "translate",
            "Translate with a trained UNMT model (Stage 4)",
            "Stage 4",
        ),
    ]

    for cmd in commands:
        sub = unmt_subparsers.add_parser(
            cmd.name,
            help=cmd.help,
            description=cmd.description,
        )
        cmd.add_arguments(sub)
        sub.set_defaults(command_handler=cmd)

    unmt_parser.set_defaults(
        command_handler=type(
            "_UnmtHelp",
            (),
            {"execute": staticmethod(lambda _args: unmt_parser.print_help() or 0)},
        )()
    )

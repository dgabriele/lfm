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
        _NotImplementedCommand(
            "embed",
            "Train monolingual subword embeddings (Stage 2)",
            "Stage 2",
        ),
        _NotImplementedCommand(
            "align",
            "MUSE Procrustes alignment between embedding spaces (Stage 2)",
            "Stage 2",
        ),
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

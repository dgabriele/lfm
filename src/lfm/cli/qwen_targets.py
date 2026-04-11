"""``lfm qwen-targets`` CLI group.

Subcommands:

* ``build``   — run the full corpus → extractor → density → cluster
                → store pipeline from a YAML config.
* ``inspect`` — run the cheap sanity checks (cluster structure of the
                target LLM space, and Neuroglot coupling from a
                dialogue-game checkpoint) using the same extractor
                settings as a config.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import yaml

from lfm.cli.base import CLICommand
from lfm.qwen_targets.config import (
    CorpusSourceConfig,
    ExtractorConfig,
    QwenTargetsConfig,
)

logger = logging.getLogger(__name__)


def _load_config(config_path: str | None) -> QwenTargetsConfig:
    if config_path is None:
        raise ValueError("Config path is required")
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {path}")
    with open(path, encoding="utf-8") as fh:
        raw = yaml.safe_load(fh) or {}
    # Pydantic will recurse into nested model types automatically
    return QwenTargetsConfig(**raw)


class BuildCommand(CLICommand):
    """Build a Qwen-latent :class:`EmbeddingStore` from a config."""

    @property
    def name(self) -> str:
        return "build"

    @property
    def help(self) -> str:
        return "Build a Qwen-latent EmbeddingStore from a mixed corpus"

    @property
    def description(self) -> str:
        return (
            "Run the full pipeline: load corpora → extract LLM hidden "
            "states → optionally apply density-aware resampling → "
            "cluster → save a drop-in EmbeddingStore compatible with "
            "the dialogue game trainer."
        )

    def add_arguments(self, parser: argparse.ArgumentParser) -> None:
        parser.add_argument("config", help="YAML config file")

    def execute(self, args: argparse.Namespace) -> int:
        from lfm.qwen_targets.builder import QwenEmbeddingStoreBuilder

        config = _load_config(args.config)
        builder = QwenEmbeddingStoreBuilder(config)
        store = builder.build()
        print(
            f"Store ready at {config.output_dir}: "
            f"{store.num_passages} passages, "
            f"{store.num_clusters} clusters, "
            f"dim={store.embedding_dim}"
        )
        return 0


class PrefetchCommand(CLICommand):
    """Prefetch HF streaming sources to local JSONL caches."""

    @property
    def name(self) -> str:
        return "prefetch"

    @property
    def help(self) -> str:
        return "Materialize HF streaming sources to local JSONL caches"

    @property
    def description(self) -> str:
        return (
            "Iterate every HuggingFace streaming source in the config "
            "and write its records to a local JSONL file in "
            "`prefetch_dir`.  Caches are atomic: partial writes leave a "
            "`.tmp` file rather than a half-complete cache.  Once a "
            "cache exists, subsequent `build` runs read from disk "
            "instead of re-streaming — much faster and resumable."
        )

    def add_arguments(self, parser: argparse.ArgumentParser) -> None:
        parser.add_argument("config", help="YAML config file")
        parser.add_argument(
            "--force", action="store_true",
            help="Re-download even if a cache exists",
        )

    def execute(self, args: argparse.Namespace) -> int:
        from lfm.qwen_targets.prefetch import prefetch_all

        config = _load_config(args.config)
        prefetch_dir = Path(config.prefetch_dir)
        logger.info("Prefetch directory: %s", prefetch_dir.resolve())
        paths = prefetch_all(
            sources=config.sources,
            prefetch_dir=prefetch_dir,
            force=args.force,
        )
        print(f"Prefetched {len(paths)} sources to {prefetch_dir}:")
        for p in paths:
            print(f"  {p}")
        return 0


class InspectCommand(CLICommand):
    """Run cheap sanity checks on the target latent space."""

    @property
    def name(self) -> str:
        return "inspect"

    @property
    def help(self) -> str:
        return "Run cluster-structure and Neuroglot-coupling sanity checks"

    @property
    def description(self) -> str:
        return (
            "Two cheap empirical checks before committing to a full "
            "build or training run.  Check 1 clusters LLM hidden "
            "states from a sample of texts to see whether the target "
            "space carries usable structure.  Check 2 feeds Neuroglot "
            "from a dialogue-game checkpoint back into the LLM to see "
            "whether its hidden state couples to input."
        )

    def add_arguments(self, parser: argparse.ArgumentParser) -> None:
        parser.add_argument("config", help="YAML config file")
        parser.add_argument(
            "--checkpoint", default="data/dialogue_game_v7/best.pt",
            help="Dialogue-game checkpoint for Check 2 (default: baseline)",
        )
        parser.add_argument(
            "--decoder-path", default="data/models/v7/vae_decoder.pt",
        )
        parser.add_argument(
            "--spm-path", default="data/models/v7/spm.model",
        )
        parser.add_argument(
            "--embedding-store", default="data/embeddings",
            help="Existing store used to source diverse sentences for "
                 "the Neuroglot-coupling test",
        )
        parser.add_argument("--k-clusters", type=int, default=20)
        parser.add_argument("--samples-per-cluster", type=int, default=5)
        parser.add_argument("--max-texts", type=int, default=1000)
        parser.add_argument("--n-coupling", type=int, default=20)
        parser.add_argument("--skip-check1", action="store_true")
        parser.add_argument("--skip-check2", action="store_true")

    def execute(self, args: argparse.Namespace) -> int:
        from lfm.qwen_targets.corpora import JSONLCorpusSource
        from lfm.qwen_targets.inspect import (
            inspect_cluster_structure,
            inspect_neuroglot_coupling,
        )

        config = _load_config(args.config)

        if not args.skip_check1:
            first_source = config.sources[0]
            src_path = Path(first_source.path)
            src = JSONLCorpusSource(
                path=src_path,
                text_field=first_source.text_field or "text",
                name=first_source.name or src_path.stem,
                max_samples=args.max_texts,
            )
            inspect_cluster_structure(
                source=src,
                extractor_config=config.extractor,
                k=args.k_clusters,
                samples_per_cluster=args.samples_per_cluster,
                max_texts=args.max_texts,
                seed=config.shuffle_seed,
                device=config.device,
            )

        if not args.skip_check2:
            ckpt = Path(args.checkpoint)
            if not ckpt.exists():
                print(f"Warning: checkpoint {ckpt} not found, skipping Check 2")
                return 0
            inspect_neuroglot_coupling(
                checkpoint_path=str(ckpt),
                extractor_config=config.extractor,
                decoder_path=args.decoder_path,
                spm_path=args.spm_path,
                embedding_store_dir=args.embedding_store,
                n_coupling=args.n_coupling,
                seed=config.shuffle_seed,
                device=config.device,
            )
        return 0


def register_qwen_targets_group(
    parent_subparsers: argparse._SubParsersAction,
) -> None:
    """Register the ``qwen-targets`` subcommand group."""
    parent = parent_subparsers.add_parser(
        "qwen-targets",
        help="Build and inspect LLM-latent target embeddings",
        description=(
            "Build dialogue-game target embeddings in a pretrained "
            "LLM's latent space, with density-aware resampling."
        ),
    )
    sub = parent.add_subparsers(
        title="qwen-targets commands",
        description="Pipeline stages",
        dest="qwen_targets_cmd",
    )

    for cmd in [PrefetchCommand(), BuildCommand(), InspectCommand()]:
        p = sub.add_parser(cmd.name, help=cmd.help, description=cmd.description)
        cmd.add_arguments(p)
        p.set_defaults(command_handler=cmd)

    parent.set_defaults(
        command_handler=type(
            "_QwenTargetsHelp", (),
            {"execute": staticmethod(lambda _args: parent.print_help() or 0)},
        )()
    )

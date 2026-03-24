"""CLI subcommand: ``lfm setup`` — data acquisition and directory setup."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from lfm.cli.base import CLICommand

logger = logging.getLogger(__name__)

# Leipzig download URLs (direct links to 100K sentence tarballs)
_LEIPZIG_BASE = "https://downloads.wortschatz-leipzig.de/corpora"
_LEIPZIG_LANGUAGES = {
    "ara": "ara_news_2020_100K",
    "ces": "ces_news_2022_100K",
    "deu": "deu_news_2023_100K",
    "eng": "eng_news_2023_100K",
    "est": "est_newscrawl_2017_100K",
    "fin": "fin_news_2022_100K",
    "hin": "hin_news_2022_100K",
    "hun": "hun_news_2022_100K",
    "ind": "ind_news_2022_100K",
    "kor": "kor_news_2022_100K",
    "pol": "pol_news_2022_100K",
    "por": "por_news_2022_100K",
    "rus": "rus_news_2022_100K",
    "spa": "spa_news_2023_100K",
    "tur": "tur_news_2022_100K",
    "vie": "vie_news_2022_100K",
}


class SetupCommand(CLICommand):
    @property
    def name(self) -> str:
        return "setup"

    @property
    def help(self) -> str:
        return "Download and set up required data"

    @property
    def description(self) -> str:
        return (
            "Download corpus data and set up the data directory.\n\n"
            "Examples:\n"
            "  lfm setup data --corpus leipzig    # Download Leipzig corpora\n"
            "  lfm setup data --embeddings        # Precompute sentence embeddings\n"
            "  lfm setup data --all               # Everything\n"
        )

    def add_arguments(self, parser: argparse.ArgumentParser) -> None:
        parser.add_argument(
            "target",
            choices=["data"],
            help="What to set up",
        )
        parser.add_argument(
            "--corpus",
            choices=["leipzig"],
            default=None,
            help="Corpus to download",
        )
        parser.add_argument(
            "--embeddings",
            action="store_true",
            help="Precompute sentence embeddings for agent game",
        )
        parser.add_argument(
            "--all",
            action="store_true",
            help="Set up everything",
        )
        parser.add_argument(
            "--data-dir",
            default="data",
            help="Base data directory (default: data)",
        )

    def execute(self, args: argparse.Namespace) -> int:
        data_dir = Path(args.data_dir)

        if args.all or args.corpus == "leipzig":
            self._setup_leipzig(data_dir)

        if args.all or args.embeddings:
            self._setup_embeddings(data_dir)

        if not args.all and not args.corpus and not args.embeddings:
            print("Specify --corpus, --embeddings, or --all")
            return 1

        return 0

    def _setup_leipzig(self, data_dir: Path) -> None:
        """Download and extract Leipzig corpora."""
        import tarfile
        import urllib.request

        leipzig_dir = data_dir / "leipzig"
        leipzig_dir.mkdir(parents=True, exist_ok=True)

        for lang, name in sorted(_LEIPZIG_LANGUAGES.items()):
            target_dir = leipzig_dir / name
            sentence_file = target_dir / f"{name}-sentences.txt"

            if sentence_file.exists():
                print(f"  Skip {lang}: {sentence_file} exists")
                continue

            url = f"{_LEIPZIG_BASE}/{name}.tar.gz"
            tarball = leipzig_dir / f"{name}.tar.gz"

            print(f"  Downloading {lang}: {url}")
            try:
                urllib.request.urlretrieve(url, tarball)
            except Exception as e:
                print(f"  FAILED {lang}: {e}")
                print(
                    f"    Manual download: visit https://wortschatz.uni-leipzig.de/en/download/"
                )
                continue

            print(f"  Extracting {tarball.name}...")
            try:
                with tarfile.open(tarball) as tf:
                    tf.extractall(path=leipzig_dir)
            except Exception as e:
                print(f"  FAILED to extract {lang}: {e}")
                continue

            if sentence_file.exists():
                print(f"  OK: {sentence_file}")
            else:
                print(f"  WARNING: extracted but {sentence_file} not found")

        # Summary
        found = sum(
            1
            for name in _LEIPZIG_LANGUAGES.values()
            if (leipzig_dir / name / f"{name}-sentences.txt").exists()
        )
        print(f"\nLeipzig: {found}/{len(_LEIPZIG_LANGUAGES)} languages ready")

    def _setup_embeddings(self, data_dir: Path) -> None:
        """Precompute sentence embeddings."""
        import subprocess

        emb_dir = data_dir / "embeddings"
        if (emb_dir / "embeddings.npy").exists():
            print("  Embeddings already exist, skipping")
            return

        print("  Running precompute_embeddings.py...")
        result = subprocess.run(
            ["poetry", "run", "python", "scripts/precompute_embeddings.py"],
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            print(f"  FAILED: {result.stderr[:500]}")
        else:
            print("  Embeddings ready")

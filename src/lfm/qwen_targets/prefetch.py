"""Materialize HuggingFace streaming corpora to local JSONL caches.

HuggingFace streaming pulls parquet shards lazily over the network —
fine for throughput, but brittle for multi-hour extraction runs: a
mid-run crash loses everything, and dataset shard rotation between
runs can silently change the input.

The :func:`prefetch_source` helper runs a HF source once, writes its
records to a local JSONL file, and leaves an atomic cache at
``<prefetch_dir>/<source-name>.jsonl``.  The builder then prefers
that cache over re-streaming.

Caches are written to ``<path>.tmp`` first and renamed on success so
that a crashed prefetch leaves no partial file that looks complete.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path

from lfm.qwen_targets.config import CorpusSourceConfig
from lfm.qwen_targets.corpora import HFStreamingCorpusSource

logger = logging.getLogger(__name__)


def cache_path_for(source: CorpusSourceConfig, prefetch_dir: Path) -> Path:
    """Return the canonical cache file path for a source."""
    fallback = (source.hf_dataset or source.name or "source").replace("/", "-")
    name = source.name or fallback
    return prefetch_dir / f"{name}.jsonl"


def is_cached(source: CorpusSourceConfig, prefetch_dir: Path) -> bool:
    """Check whether a non-empty cache already exists for this source."""
    path = cache_path_for(source, prefetch_dir)
    return path.exists() and path.stat().st_size > 0


def prefetch_source(
    source: CorpusSourceConfig,
    prefetch_dir: Path,
    *,
    force: bool = False,
) -> Path:
    """Stream a HF source once and write its records to a local JSONL.

    Args:
        source: A :class:`CorpusSourceConfig` with ``hf_dataset`` set.
        prefetch_dir: Directory to write cache files into (created if
            missing).
        force: If ``True``, always re-fetch even when a cache exists.

    Returns:
        Path to the materialized cache file.

    Raises:
        ValueError: If the source has no ``hf_dataset`` field.
    """
    if source.hf_dataset is None:
        raise ValueError(
            f"Source '{source.name}' has no hf_dataset; cannot prefetch",
        )

    prefetch_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_path_for(source, prefetch_dir)

    if cache_path.exists() and not force:
        with cache_path.open("r", encoding="utf-8") as fh:
            existing = sum(1 for _ in fh)
        logger.info(
            "Cache hit %s: %d records already cached — skipping "
            "(pass --force to re-fetch)",
            cache_path.name, existing,
        )
        return cache_path

    stream = HFStreamingCorpusSource(
        dataset_name=source.hf_dataset,
        config_name=source.hf_config,
        split=source.hf_split,
        text_field=source.text_field or "text",
        name=source.name or source.hf_dataset.replace("/", "-"),
        max_samples=source.max_samples,
        min_length=source.min_length,
        max_length=source.max_length,
        trust_remote_code=source.hf_trust_remote_code,
    )

    tmp_path = cache_path.with_suffix(cache_path.suffix + ".tmp")
    logger.info(
        "Prefetching %s -> %s (target=%s)",
        source.name, cache_path, source.max_samples,
    )

    start = time.time()
    written = 0
    try:
        with tmp_path.open("w", encoding="utf-8") as fh:
            for rec in stream.iterate():
                fh.write(
                    json.dumps(
                        {"text": rec.text, "source_index": rec.source_index},
                        ensure_ascii=False,
                    )
                )
                fh.write("\n")
                written += 1
                if written % 10_000 == 0:
                    elapsed = time.time() - start
                    rate = written / max(elapsed, 1e-6)
                    logger.info(
                        "  %s: %d / %s records (%.0f/s)",
                        source.name, written,
                        source.max_samples or "?",
                        rate,
                    )
    except Exception:
        # Leave the .tmp in place so we can inspect it, but do not
        # rename over the canonical cache path — incomplete prefetches
        # must not masquerade as valid caches.
        logger.exception("Prefetch failed after %d records", written)
        raise

    tmp_path.replace(cache_path)
    elapsed = time.time() - start
    logger.info(
        "Prefetched %s: %d records in %.1fs (%.0f/s) -> %s",
        source.name, written, elapsed,
        written / max(elapsed, 1e-6),
        cache_path,
    )
    return cache_path


def prefetch_all(
    sources: list[CorpusSourceConfig],
    prefetch_dir: Path,
    *,
    force: bool = False,
) -> list[Path]:
    """Prefetch every HF source in a list, skipping local-file entries."""
    prefetch_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    for source in sources:
        if source.hf_dataset is None:
            logger.info(
                "Source '%s' is a local file; not prefetching",
                source.name,
            )
            continue
        paths.append(prefetch_source(source, prefetch_dir, force=force))
    return paths

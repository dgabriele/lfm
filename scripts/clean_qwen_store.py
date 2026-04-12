#!/usr/bin/env python
"""Post-hoc cleanup of a built Qwen-latent EmbeddingStore.

Drops three classes of junk that the v2 build's prefetch step let
through:

1. **Project Gutenberg front-matter boilerplate** — the cached
   gutenberg-en records are mostly the "Project Gutenberg eBook, X,
   by Y" preamble rather than actual book content, and they duplicate
   across editions.  Chunks containing multiple Gutenberg marker
   phrases are dropped.

2. **Software-license comment headers** — the-stack-smol-xl records
   typically open with a license block (Apache/BSD/GPL/MIT) that
   gets chunked as its own 200+ token span and repeats across many
   files, producing a dominant "license" cluster that crowds out
   actual code.

3. **Exact-text duplicates** — any chunk whose whitespace-normalized
   text has already been seen.

The script then re-runs MiniBatchKMeans on the surviving embeddings
and writes a fresh :class:`EmbeddingStore` into ``<output_dir>``,
leaving the input untouched.

Usage:
    poetry run python scripts/clean_qwen_store.py \
        data/embeddings_qwen_v2 \
        data/embeddings_qwen_v2_clean

The script is intentionally standalone (no CLI group registration)
because it's designed to be rsync'd to a vast.ai instance and run
there against a freshly-built store, then the cleaned output rsync'd
back.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np

from lfm.embeddings.store import EmbeddingStore
from lfm.qwen_targets.cluster import run_minibatch_kmeans
from lfm.qwen_targets.config import ClusterConfig
from lfm.qwen_targets.filters import (
    DuplicateFilter,
    is_gutenberg_boilerplate,
    is_license_boilerplate,
)

logger = logging.getLogger(__name__)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Drop Gutenberg/license/duplicate rows from a store and re-cluster.",
    )
    parser.add_argument("input_dir", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--num-clusters", type=int, default=2047)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--cluster-batch-size", type=int, default=4096)
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Count drops and print summary without writing the output store.",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )

    # ------------------------------------------------------------------
    # Load the original store
    # ------------------------------------------------------------------
    logger.info("Loading input store: %s", args.input_dir)
    store = EmbeddingStore(args.input_dir)
    store.load()

    # Copy embeddings out of the mmap so we can index freely.
    embeddings = np.array(store._embeddings)
    dim = embeddings.shape[1]
    n_orig = embeddings.shape[0]

    passages_path = args.input_dir / "passages.jsonl"
    if not passages_path.exists():
        raise FileNotFoundError(f"Missing passages.jsonl in {args.input_dir}")
    passages: list[dict] = []
    with passages_path.open(encoding="utf-8") as fh:
        for line in fh:
            passages.append(json.loads(line))
    if len(passages) != n_orig:
        raise RuntimeError(
            f"Passage count ({len(passages)}) does not match embedding count "
            f"({n_orig}) — store is corrupt.",
        )
    logger.info("Loaded %d rows, dim=%d", n_orig, dim)

    # ------------------------------------------------------------------
    # Filter
    # ------------------------------------------------------------------
    keep_mask = np.ones(n_orig, dtype=bool)
    dup_filter = DuplicateFilter()

    dropped_gutenberg = 0
    dropped_license = 0
    dropped_duplicate = 0

    per_source_kept: dict[str, int] = {}
    per_source_dropped: dict[str, int] = {}

    for i, rec in enumerate(passages):
        text = rec.get("text", "")
        source = rec.get("source", "?")

        if is_gutenberg_boilerplate(text):
            keep_mask[i] = False
            dropped_gutenberg += 1
            per_source_dropped[source] = per_source_dropped.get(source, 0) + 1
            continue

        if is_license_boilerplate(text):
            keep_mask[i] = False
            dropped_license += 1
            per_source_dropped[source] = per_source_dropped.get(source, 0) + 1
            continue

        if not dup_filter(text):
            keep_mask[i] = False
            dropped_duplicate += 1
            per_source_dropped[source] = per_source_dropped.get(source, 0) + 1
            continue

        per_source_kept[source] = per_source_kept.get(source, 0) + 1

    n_keep = int(keep_mask.sum())

    # ------------------------------------------------------------------
    # Report
    # ------------------------------------------------------------------
    logger.info("=" * 72)
    logger.info("FILTER SUMMARY")
    logger.info("=" * 72)
    logger.info("Input rows:        %d", n_orig)
    logger.info("Kept:              %d (%.1f%%)", n_keep, 100 * n_keep / n_orig)
    logger.info("Dropped:           %d (%.1f%%)",
                n_orig - n_keep, 100 * (n_orig - n_keep) / n_orig)
    logger.info("  gutenberg:       %d", dropped_gutenberg)
    logger.info("  license header:  %d", dropped_license)
    logger.info("  duplicate:       %d", dropped_duplicate)
    logger.info("-" * 72)
    logger.info("Per-source breakdown (kept / dropped):")
    sources = sorted(set(per_source_kept) | set(per_source_dropped))
    for s in sources:
        k = per_source_kept.get(s, 0)
        d = per_source_dropped.get(s, 0)
        tot = k + d
        pct = (100 * d / tot) if tot else 0.0
        logger.info("  %-20s  kept=%7d  dropped=%6d  (%.1f%% dropped)",
                    s, k, d, pct)
    logger.info("=" * 72)

    if args.dry_run:
        logger.info("--dry-run set; not writing output store")
        return 0

    if n_keep == 0:
        raise RuntimeError("Nothing survived filtering — refusing to write empty store")

    # ------------------------------------------------------------------
    # Re-cluster
    # ------------------------------------------------------------------
    filtered_embeddings = embeddings[keep_mask].astype(np.float32)
    filtered_passages = [passages[i] for i in range(n_orig) if keep_mask[i]]

    logger.info("Re-clustering %d points into %d clusters...",
                n_keep, args.num_clusters)
    cluster_cfg = ClusterConfig(
        num_clusters=args.num_clusters,
        random_state=args.random_state,
        batch_size=args.cluster_batch_size,
    )
    cluster_labels = run_minibatch_kmeans(filtered_embeddings, cluster_cfg)

    # ------------------------------------------------------------------
    # Save the cleaned store
    # ------------------------------------------------------------------
    new_metadata = dict(store.metadata)
    new_metadata["cleaned_from"] = str(args.input_dir)
    new_metadata["cleanup_filter"] = {
        "gutenberg_dropped": dropped_gutenberg,
        "license_dropped": dropped_license,
        "duplicate_dropped": dropped_duplicate,
        "kept": n_keep,
        "original": n_orig,
    }

    EmbeddingStore.create(
        store_dir=args.output_dir,
        embeddings=filtered_embeddings,
        cluster_labels=cluster_labels,
        metadata=new_metadata,
        passages=filtered_passages,
    )
    logger.info("Wrote cleaned store: %s", args.output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Cluster a multi-position embedding store and emit standard EmbeddingStore artifacts.

Designed for the ``last_k_concat`` Qwen embedding store at
``data/embeddings_qwen[*]/`` whose ``embeddings.npy`` is currently a
*raw* float16 memmap of shape ``(N, n_pos, dim)``. This script:

  1. Reads the raw memmap by explicit shape.
  2. Mean-pools across positions → ``(N, dim)`` for clustering only
     (per-position structure is preserved in the saved store).
  3. Runs MiniBatchKMeans into ``--n-clusters`` clusters.
  4. Re-saves embeddings.npy in *standard* ``.npy`` format (with header)
     so it's loadable via ``np.load(mmap_mode='r')`` — what
     ``EmbeddingStore.load()`` expects.
  5. Writes ``cluster_labels.npy``, ``cluster_index.json``, and
     ``metadata.json`` alongside.

After this runs, ``EmbeddingStore(store_dir).load()`` works and the
``AgentTrainer``'s vectorized hard-negative sampler can hit our store
with no further plumbing.

Usage:
    poetry run python scripts/build_synth_clusters.py \\
        data/embeddings_qwen_subset \\
        --n-pos 8 --dim 896 --n-clusters 2048 [--seed 42]

If ``embeddings.npy`` is already in standard format, the conversion is a
no-op (np.save round-trip preserves data).
"""

from __future__ import annotations

import argparse
import json
import logging
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.cluster import MiniBatchKMeans


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("store_dir")
    p.add_argument("--n-pos", type=int, required=True)
    p.add_argument("--dim", type=int, required=True)
    p.add_argument("--n-clusters", type=int, default=2048)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--batch-size", type=int, default=4096,
                   help="MiniBatchKMeans batch size")
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    log = logging.getLogger(__name__)

    store_dir = Path(args.store_dir)
    emb_path = store_dir / "embeddings.npy"
    bytes_per_sample = args.n_pos * args.dim * 2  # float16

    file_size = emb_path.stat().st_size
    # Detect format: standard .npy starts with magic '\x93NUMPY'.
    with emb_path.open("rb") as fh:
        head = fh.read(6)
    is_standard_npy = head.startswith(b"\x93NUMPY")
    log.info("embeddings.npy: %d bytes, standard=%s", file_size, is_standard_npy)

    if is_standard_npy:
        embeddings = np.load(str(emb_path), mmap_mode="r")
    else:
        if file_size % bytes_per_sample:
            raise ValueError(
                f"raw memmap size {file_size} not divisible by per-sample bytes "
                f"{bytes_per_sample}"
            )
        n_total = file_size // bytes_per_sample
        embeddings = np.memmap(
            emb_path, dtype=np.float16, mode="r",
            shape=(n_total, args.n_pos, args.dim),
        )
    log.info("loaded shape=%s dtype=%s", embeddings.shape, embeddings.dtype)
    n_total = int(embeddings.shape[0])

    # Mean-pool positions for clustering. KMeans operates on (N, dim).
    log.info("mean-pooling positions for clustering ...")
    pooled = np.asarray(embeddings.astype(np.float32).mean(axis=1))  # (N, dim)

    log.info("MiniBatchKMeans: n_clusters=%d, n=%d, dim=%d",
             args.n_clusters, n_total, args.dim)
    km = MiniBatchKMeans(
        n_clusters=args.n_clusters, random_state=args.seed,
        batch_size=args.batch_size, n_init=10, max_iter=200,
        verbose=0,
    )
    labels = km.fit_predict(pooled).astype(np.int32)
    sizes = np.bincount(labels, minlength=args.n_clusters)
    log.info("cluster sizes: min=%d  median=%d  max=%d",
             int(sizes.min()), int(np.median(sizes)), int(sizes.max()))

    # Save (or re-save) embeddings.npy in standard format. Required for
    # EmbeddingStore.load() to use np.load(mmap_mode='r').
    if not is_standard_npy:
        log.info("converting embeddings.npy to standard .npy format ...")
        # Materialize full array (716 MB at our scale) then save with header.
        full = np.array(embeddings, dtype=embeddings.dtype, copy=True)
        # NB: write via open file handle so np.save doesn't auto-append ".npy"
        # to a non-".npy" path (it does for path strings).
        tmp = emb_path.with_name(emb_path.name + ".tmp")
        with tmp.open("wb") as fh:
            np.save(fh, full)
        tmp.replace(emb_path)
        log.info("rewrote %s as standard .npy (%d bytes)",
                 emb_path, emb_path.stat().st_size)

    # Save cluster_labels.npy
    labels_path = store_dir / "cluster_labels.npy"
    np.save(str(labels_path), labels)
    log.info("saved %s", labels_path)

    # Save cluster_index.json
    index: dict[int, list[int]] = defaultdict(list)
    for idx, lbl in enumerate(labels):
        index[int(lbl)].append(idx)
    serializable = {str(k): v for k, v in sorted(index.items())}
    index_path = store_dir / "cluster_index.json"
    with index_path.open("w", encoding="utf-8") as fh:
        json.dump(serializable, fh)
    log.info("saved %s (%d clusters)", index_path, len(index))

    # Save metadata.json
    meta = {
        "num_passages": n_total,
        "embedding_dim": int(args.dim),
        "embedding_shape": [int(args.n_pos), int(args.dim)],
        "embedding_dtype": "float16",
        "num_clusters": int(args.n_clusters),
        "kmeans": {
            "algorithm": "MiniBatchKMeans",
            "pooling": "mean_over_positions",
            "seed": int(args.seed),
        },
    }
    meta_path = store_dir / "metadata.json"
    with meta_path.open("w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=2)
    log.info("saved %s", meta_path)


if __name__ == "__main__":
    main()

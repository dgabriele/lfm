"""Cluster multi-position embeddings for the EmbeddingStore.

Expects:  data/embeddings_qwen/embeddings.npy  shape (N, P, D) float16
Writes:   cluster_labels.npy   (N,) int32
          cluster_index.json   {cluster_id: [sample_idx, ...]}
          metadata.json        {n_samples, n_clusters, embedding_dim, n_positions}
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from sklearn.cluster import MiniBatchKMeans

STORE = Path("data/embeddings_qwen")
N_CLUSTERS = 2048
BATCH = 65536
SEED = 42


def main() -> None:
    emb_path = STORE / "embeddings.npy"
    print(f"loading {emb_path} ...", flush=True)
    emb = np.load(str(emb_path), mmap_mode="r")  # (N, P, D) float16
    print(f"  shape={emb.shape} dtype={emb.dtype}", flush=True)

    # Mean-pool positions for clustering
    if emb.ndim == 3:
        flat = emb.mean(axis=1).astype(np.float32)  # (N, D)
    else:
        flat = emb.astype(np.float32)

    N, D = flat.shape
    print(f"  flat shape={flat.shape}", flush=True)

    print(f"clustering N={N} into K={N_CLUSTERS} ...", flush=True)
    km = MiniBatchKMeans(
        n_clusters=N_CLUSTERS,
        batch_size=BATCH,
        n_init=1,
        init="random",
        max_iter=50,
        random_state=SEED,
        verbose=0,
    )
    labels = km.fit_predict(flat).astype(np.int32)
    print(f"  done. inertia={km.inertia_:.2f}", flush=True)

    # Save labels
    out_labels = STORE / "cluster_labels.npy"
    np.save(str(out_labels), labels)
    print(f"saved {out_labels}", flush=True)

    # Build index: cluster_id -> list of sample indices
    index: dict[str, list[int]] = {str(k): [] for k in range(N_CLUSTERS)}
    for i, lbl in enumerate(labels.tolist()):
        index[str(lbl)].append(i)
    out_index = STORE / "cluster_index.json"
    with open(out_index, "w") as f:
        json.dump(index, f)
    print(f"saved {out_index}", flush=True)

    # Metadata
    meta = {
        "n_samples": int(N),
        "n_clusters": int(N_CLUSTERS),
        "embedding_dim": int(D),
        "n_positions": int(emb.shape[1]) if emb.ndim == 3 else 1,
    }
    out_meta = STORE / "metadata.json"
    with open(out_meta, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"saved {out_meta}", flush=True)
    print("cluster_embeddings done.", flush=True)


if __name__ == "__main__":
    main()

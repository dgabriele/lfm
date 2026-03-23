"""Generate synthetic embeddings for testing the agent game pipeline.

Creates a mock EmbeddingStore with clustered Gaussian embeddings that
have known structure — different clusters represent different "topics"
with controlled inter/intra-cluster distances. This lets us validate
the full game pipeline without needing an LLM encoder or text corpus.

Usage::

    python scripts/generate_synthetic_embeddings.py
"""

from __future__ import annotations

import numpy as np

from lfm.embeddings.store import EmbeddingStore


def generate_synthetic_embeddings(
    store_dir: str = "data/embeddings",
    num_clusters: int = 32,
    samples_per_cluster: int = 200,
    embedding_dim: int = 1024,
    cluster_spread: float = 0.3,
    seed: int = 42,
) -> EmbeddingStore:
    """Generate and save synthetic clustered embeddings.

    Creates ``num_clusters`` Gaussian clusters in ``embedding_dim``-space.
    Cluster centers are random unit vectors (well-separated in high dims).
    Samples within each cluster are center + Gaussian noise scaled by
    ``cluster_spread``.

    Args:
        store_dir: Directory to save the embedding store.
        num_clusters: Number of distinct clusters.
        samples_per_cluster: Embeddings per cluster.
        embedding_dim: Embedding dimensionality.
        cluster_spread: Standard deviation of within-cluster noise.
        seed: Random seed.

    Returns:
        Loaded ``EmbeddingStore`` ready for use.
    """
    rng = np.random.RandomState(seed)
    total = num_clusters * samples_per_cluster

    # Generate cluster centers as random unit vectors
    centers = rng.randn(num_clusters, embedding_dim).astype(np.float32)
    centers /= np.linalg.norm(centers, axis=1, keepdims=True)
    # Scale centers apart
    centers *= 3.0

    # Generate samples around centers
    embeddings = np.zeros((total, embedding_dim), dtype=np.float32)
    cluster_labels = np.zeros(total, dtype=np.int32)

    for c in range(num_clusters):
        start = c * samples_per_cluster
        end = start + samples_per_cluster
        noise = rng.randn(samples_per_cluster, embedding_dim).astype(
            np.float32
        )
        embeddings[start:end] = centers[c] + cluster_spread * noise
        cluster_labels[start:end] = c

    # Shuffle
    perm = rng.permutation(total)
    embeddings = embeddings[perm]
    cluster_labels = cluster_labels[perm]

    # Normalize to unit vectors (like real sentence embeddings)
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    embeddings = embeddings / np.clip(norms, 1e-8, None)

    metadata = {
        "num_passages": total,
        "embedding_dim": embedding_dim,
        "num_clusters": num_clusters,
        "source": "synthetic_gaussian",
        "cluster_spread": cluster_spread,
    }

    print(
        f"Generated {total} embeddings "
        f"({num_clusters} clusters × {samples_per_cluster}), "
        f"dim={embedding_dim}"
    )

    store = EmbeddingStore.create(
        store_dir=store_dir,
        embeddings=embeddings,
        cluster_labels=cluster_labels,
        metadata=metadata,
    )
    print(f"Saved to {store_dir}/")
    return store


if __name__ == "__main__":
    generate_synthetic_embeddings()

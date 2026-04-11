"""K-means clustering for the target embeddings.

Matches the MiniBatchKMeans pattern already used by
:mod:`lfm.embeddings.pipeline` so the downstream dialogue game sees a
store with the same cluster layout and index structure it's been
trained against.
"""

from __future__ import annotations

import logging

import numpy as np

from lfm.qwen_targets.config import ClusterConfig

logger = logging.getLogger(__name__)


def run_minibatch_kmeans(
    embeddings: np.ndarray, config: ClusterConfig,
) -> np.ndarray:
    """Cluster ``embeddings`` into ``config.num_clusters`` groups.

    Returns ``(N,)`` int32 cluster labels.  Uses MiniBatchKMeans for
    scalability — same algorithm as the existing sentence-transformer
    store pipeline.
    """
    from sklearn.cluster import MiniBatchKMeans

    n = embeddings.shape[0]
    k = config.num_clusters
    batch_size = min(config.batch_size, n)
    logger.info(
        "MiniBatchKMeans: N=%d dim=%d k=%d batch=%d",
        n, embeddings.shape[1], k, batch_size,
    )
    kmeans = MiniBatchKMeans(
        n_clusters=k,
        random_state=config.random_state,
        batch_size=batch_size,
        verbose=0,
    )
    labels = kmeans.fit_predict(embeddings)
    return labels.astype(np.int32)

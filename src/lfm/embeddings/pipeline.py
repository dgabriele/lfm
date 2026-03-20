"""Offline precomputation pipeline for the embeddings subpackage.

Orchestrates the full chunk -> encode -> cluster -> store workflow in a
single reproducible run.  Each step checks for existing outputs on disk so
that interrupted runs can be resumed without re-doing expensive work.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from lfm._registry import create as registry_create
from lfm.embeddings.chunker import TextChunker
from lfm.embeddings.config import PrecomputePipelineConfig
from lfm.embeddings.encoder import TextEncoder
from lfm.embeddings.store import EmbeddingStore
from lfm.utils.logging import get_logger

logger = get_logger(__name__)


class PrecomputePipeline:
    """End-to-end offline pipeline: chunk corpus, encode, cluster, save.

    Args:
        config: Full pipeline configuration.
    """

    def __init__(self, config: PrecomputePipelineConfig) -> None:
        self.config = config

    def run(self) -> EmbeddingStore:
        """Execute the full precomputation pipeline.

        Steps:
            1. Chunk the corpus into overlapping passages.
            2. Encode all passages into dense embeddings.
            3. Cluster the embeddings.
            4. Write everything to the on-disk store.

        If a completed store already exists at the configured path, it is
        loaded and returned immediately.

        Returns:
            A loaded :class:`EmbeddingStore` ready for training-time sampling.
        """
        store_dir = Path(self.config.store.store_dir)

        # -- Resumption: skip everything if the store already exists ----
        if self._store_exists(store_dir):
            logger.info(
                "Existing store found at %s -- loading instead of re-running",
                store_dir,
            )
            store = EmbeddingStore(store_dir)
            store.load()
            return store

        # -- Step 1: Chunk ---------------------------------------------
        passages = self._chunk()
        if not passages:
            raise RuntimeError(
                "Chunking produced zero passages. Check corpus_paths and "
                "corpus_format in the pipeline config."
            )
        logger.info("Chunking complete: %d passages", len(passages))

        # -- Step 2: Encode --------------------------------------------
        texts = [p["text"] for p in passages]
        embeddings = self._encode(texts)
        logger.info(
            "Encoding complete: shape=%s, dtype=%s",
            embeddings.shape,
            embeddings.dtype,
        )

        # -- Step 3: Cluster -------------------------------------------
        labels = self._cluster(embeddings)
        n_unique = len(np.unique(labels))
        logger.info(
            "Clustering complete: %d unique clusters from %d passages",
            n_unique,
            len(labels),
        )

        # -- Step 4: Save ----------------------------------------------
        store = self._save(embeddings, labels, passages)
        logger.info("Pipeline complete. Store written to %s", store_dir)
        return store

    # ------------------------------------------------------------------
    # Private step methods
    # ------------------------------------------------------------------

    def _chunk(self) -> list[dict[str, Any]]:
        """Create encoder (for its tokenizer) and chunk the corpus."""
        encoder = self._build_encoder()
        tokenizer = encoder.get_tokenizer()
        chunker = TextChunker(self.config.chunker, tokenizer)
        return chunker.chunk_corpus(
            self.config.corpus_paths,
            corpus_format=self.config.corpus_format,
        )

    def _encode(self, texts: list[str]) -> np.ndarray:
        """Encode all passage texts into dense vectors."""
        encoder = self._build_encoder()
        embeddings = encoder.encode_batched(texts, show_progress=True)

        # Cast to the configured storage dtype.
        target_dtype = np.dtype(self.config.store.embedding_dtype)
        if embeddings.dtype != target_dtype:
            embeddings = embeddings.astype(target_dtype)

        return embeddings

    def _cluster(self, embeddings: np.ndarray) -> np.ndarray:
        """Cluster embeddings on CPU using the configured algorithm.

        Supports ``"kmeans"`` (via scikit-learn's MiniBatchKMeans) and
        ``"hdbscan"`` (via the hdbscan library).  Both are imported lazily.

        Args:
            embeddings: Dense array of shape ``(N, dim)``.

        Returns:
            Cluster label array of shape ``(N,)`` with dtype ``int32``.
        """
        cfg = self.config.cluster
        method = cfg.method.lower()

        # Clustering operates on float32 on CPU regardless of storage dtype.
        data = embeddings.astype(np.float32)

        if method == "kmeans":
            return self._cluster_kmeans(data, cfg)
        elif method == "hdbscan":
            return self._cluster_hdbscan(data, cfg)
        else:
            raise ValueError(
                f"Unknown clustering method: {method!r}. Supported: 'kmeans', 'hdbscan'."
            )

    def _save(
        self,
        embeddings: np.ndarray,
        labels: np.ndarray,
        passages: list[dict[str, Any]],
    ) -> EmbeddingStore:
        """Build metadata and persist to the embedding store."""
        metadata: dict[str, Any] = {
            "encoder_model_id": self.config.encoder.model_id,
            "encoder_name": self.config.encoder.name,
            "cluster_method": self.config.cluster.method,
            "cluster_num_clusters": self.config.cluster.num_clusters,
            "chunker_max_tokens": self.config.chunker.max_tokens,
            "chunker_overlap_tokens": self.config.chunker.overlap_tokens,
            "corpus_paths": self.config.corpus_paths,
            "corpus_format": self.config.corpus_format,
        }

        save_passages = passages if self.config.store.save_text else None

        return EmbeddingStore.create(
            store_dir=self.config.store.store_dir,
            embeddings=embeddings,
            cluster_labels=labels,
            metadata=metadata,
            passages=save_passages,
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _build_encoder(self) -> TextEncoder:
        """Instantiate the configured text encoder via the registry."""
        return registry_create("text_encoder", self.config.encoder.name, self.config.encoder)

    @staticmethod
    def _cluster_kmeans(data: np.ndarray, cfg: Any) -> np.ndarray:
        """Run MiniBatchKMeans clustering."""
        try:
            from sklearn.cluster import MiniBatchKMeans
        except ImportError as exc:
            raise ImportError(
                "scikit-learn is required for k-means clustering. "
                "Install it with: pip install scikit-learn"
            ) from exc

        logger.info(
            "Running MiniBatchKMeans: num_clusters=%d, N=%d",
            cfg.num_clusters,
            data.shape[0],
        )
        kmeans = MiniBatchKMeans(
            n_clusters=cfg.num_clusters,
            random_state=cfg.random_state,
            batch_size=min(4096, data.shape[0]),
            verbose=0,
        )
        labels = kmeans.fit_predict(data)
        return labels.astype(np.int32)

    @staticmethod
    def _cluster_hdbscan(data: np.ndarray, cfg: Any) -> np.ndarray:
        """Run HDBSCAN density-based clustering."""
        try:
            import hdbscan as hdbscan_lib
        except ImportError as exc:
            raise ImportError(
                "hdbscan is required for density-based clustering. "
                "Install it with: pip install hdbscan"
            ) from exc

        logger.info(
            "Running HDBSCAN: min_cluster_size=%d, N=%d",
            cfg.min_cluster_size,
            data.shape[0],
        )
        clusterer = hdbscan_lib.HDBSCAN(
            min_cluster_size=cfg.min_cluster_size,
            metric="euclidean",
        )
        labels = clusterer.fit_predict(data)

        # HDBSCAN labels noise points as -1. Reassign them to the nearest
        # non-noise cluster centroid so every passage has a valid cluster.
        noise_mask = labels == -1
        n_noise = int(noise_mask.sum())
        if n_noise > 0:
            logger.info(
                "HDBSCAN: reassigning %d noise points to nearest cluster",
                n_noise,
            )
            unique_labels = np.unique(labels[~noise_mask])
            if len(unique_labels) == 0:
                # All points are noise -- fall back to a single cluster.
                logger.warning("HDBSCAN found no clusters; assigning all points to cluster 0")
                return np.zeros(data.shape[0], dtype=np.int32)

            # Compute cluster centroids and assign noise to nearest.
            centroids = np.stack([data[labels == c].mean(axis=0) for c in unique_labels])
            noise_data = data[noise_mask]
            # Pairwise L2 distance: (n_noise, n_clusters)
            dists = np.linalg.norm(noise_data[:, None, :] - centroids[None, :, :], axis=2)
            nearest = unique_labels[dists.argmin(axis=1)]
            labels[noise_mask] = nearest

        return labels.astype(np.int32)

    @staticmethod
    def _store_exists(store_dir: Path) -> bool:
        """Check whether a complete store already exists on disk."""
        required_files = [
            "embeddings.npy",
            "cluster_labels.npy",
            "cluster_index.json",
            "metadata.json",
        ]
        return all((store_dir / f).exists() for f in required_files)

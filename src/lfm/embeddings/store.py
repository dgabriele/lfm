"""Memory-mapped on-disk embedding store.

Persists precomputed embeddings, cluster assignments, and metadata to disk
in a layout optimized for fast random access during training.  Embeddings
are memory-mapped so that only the accessed pages are loaded into RAM.

On-disk layout::

    store_dir/
        embeddings.npy       (N, dim) float16/32 -- memory-mappable
        cluster_labels.npy   (N,) int32
        cluster_index.json   {cluster_id_str: [int indices]}
        metadata.json        {N, dim, dtype, num_clusters, ...}
        passages.jsonl       (optional) one JSON object per line
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from lfm.utils.logging import get_logger

logger = get_logger(__name__)


class EmbeddingStore:
    """On-disk embedding store with memory-mapped random access.

    Use :meth:`create` to write a new store from precomputed arrays, then
    instantiate and call :meth:`load` (or use the loaded instance returned
    by ``create``) to access embeddings at training time.

    Args:
        store_dir: Path to the store directory.
    """

    def __init__(self, store_dir: str | Path) -> None:
        self.store_dir = Path(store_dir)
        self._embeddings: np.ndarray | None = None
        self._cluster_labels: np.ndarray | None = None
        self._cluster_index: dict[int, list[int]] | None = None
        self._metadata: dict[str, Any] | None = None

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    @classmethod
    def create(
        cls,
        store_dir: str | Path,
        embeddings: np.ndarray,
        cluster_labels: np.ndarray,
        metadata: dict[str, Any],
        passages: list[dict[str, Any]] | None = None,
    ) -> EmbeddingStore:
        """Write a new embedding store to disk and return the loaded instance.

        Args:
            store_dir: Directory to write into (created if absent).
            embeddings: Dense embedding matrix, shape ``(N, dim)``.
            cluster_labels: Cluster assignment for each passage, shape ``(N,)``.
            metadata: Arbitrary metadata dict persisted as JSON.
            passages: Optional list of passage dicts to save as JSONL.

        Returns:
            An :class:`EmbeddingStore` with all data loaded and ready.
        """
        store_dir = Path(store_dir)
        store_dir.mkdir(parents=True, exist_ok=True)

        # -- embeddings ------------------------------------------------
        emb_path = store_dir / "embeddings.npy"
        np.save(str(emb_path), embeddings)
        logger.info(
            "Saved embeddings: %s, shape=%s, dtype=%s",
            emb_path,
            embeddings.shape,
            embeddings.dtype,
        )

        # -- cluster labels --------------------------------------------
        labels = cluster_labels.astype(np.int32)
        labels_path = store_dir / "cluster_labels.npy"
        np.save(str(labels_path), labels)
        logger.info("Saved cluster labels: %s", labels_path)

        # -- inverted cluster index ------------------------------------
        cluster_index: dict[int, list[int]] = defaultdict(list)
        for idx, label in enumerate(labels):
            cluster_index[int(label)].append(idx)
        cluster_index = dict(cluster_index)  # drop defaultdict behaviour

        index_path = store_dir / "cluster_index.json"
        # JSON keys must be strings; convert and sort for determinism.
        serializable_index = {str(k): v for k, v in sorted(cluster_index.items())}
        with index_path.open("w", encoding="utf-8") as fh:
            json.dump(serializable_index, fh)
        logger.info("Saved cluster index: %d clusters", len(cluster_index))

        # -- metadata --------------------------------------------------
        full_metadata: dict[str, Any] = {
            "num_passages": int(embeddings.shape[0]),
            "embedding_dim": int(embeddings.shape[1]),
            "embedding_dtype": str(embeddings.dtype),
            "num_clusters": len(cluster_index),
        }
        full_metadata.update(metadata)
        meta_path = store_dir / "metadata.json"
        with meta_path.open("w", encoding="utf-8") as fh:
            json.dump(full_metadata, fh, indent=2)
        logger.info("Saved metadata: %s", meta_path)

        # -- passages (optional) ---------------------------------------
        if passages is not None:
            passages_path = store_dir / "passages.jsonl"
            with passages_path.open("w", encoding="utf-8") as fh:
                for p in passages:
                    fh.write(json.dumps(p, ensure_ascii=False) + "\n")
            logger.info("Saved %d passages to %s", len(passages), passages_path)

        # Return a loaded instance
        store = cls(store_dir)
        store.load()
        return store

    @classmethod
    def read_metadata(cls, store_dir: str | Path) -> dict[str, Any]:
        """Read ``metadata.json`` without loading the full store.

        Useful for configuration auto-detection (e.g. pulling the
        ``embedding_dim`` before constructing a game config that has
        to match).  No embeddings, cluster labels, or other heavy
        resources are touched.
        """
        path = Path(store_dir) / "metadata.json"
        if not path.exists():
            raise FileNotFoundError(
                f"metadata.json not found in {store_dir}; is it an EmbeddingStore?"
            )
        with path.open("r", encoding="utf-8") as fh:
            return json.load(fh)

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    def load(self) -> None:
        """Memory-map embeddings and load cluster index and metadata.

        Raises:
            FileNotFoundError: If the store directory or required files are
                missing.
        """
        if not self.store_dir.exists():
            raise FileNotFoundError(f"Store directory does not exist: {self.store_dir}")

        # Memory-map embeddings for zero-copy random access.
        emb_path = self.store_dir / "embeddings.npy"
        self._embeddings = np.load(str(emb_path), mmap_mode="r")
        logger.info(
            "Loaded embeddings (mmap): shape=%s, dtype=%s",
            self._embeddings.shape,
            self._embeddings.dtype,
        )

        # Cluster labels
        labels_path = self.store_dir / "cluster_labels.npy"
        self._cluster_labels = np.load(str(labels_path), mmap_mode="r")

        # Cluster index (keys are strings in JSON; convert to int)
        index_path = self.store_dir / "cluster_index.json"
        with index_path.open("r", encoding="utf-8") as fh:
            raw_index = json.load(fh)
        self._cluster_index = {int(k): v for k, v in raw_index.items()}

        # Pre-allocate numpy arrays for each cluster so rng.choice never
        # has to re-convert Python lists at sampling time.
        self._cluster_arrays: dict[int, np.ndarray] = {
            cid: np.array(members, dtype=np.intp)
            for cid, members in self._cluster_index.items()
        }

        # Metadata
        meta_path = self.store_dir / "metadata.json"
        with meta_path.open("r", encoding="utf-8") as fh:
            self._metadata = json.load(fh)

        logger.info(
            "Store loaded: %d passages, %d clusters, dim=%d",
            self.num_passages,
            self.num_clusters,
            self.embedding_dim,
        )

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def num_passages(self) -> int:
        """Total number of passages in the store."""
        self._require_loaded()
        assert self._embeddings is not None
        return int(self._embeddings.shape[0])

    @property
    def embedding_dim(self) -> int:
        """Dimensionality of the embedding vectors."""
        self._require_loaded()
        assert self._embeddings is not None
        return int(self._embeddings.shape[1])

    @property
    def num_clusters(self) -> int:
        """Number of distinct clusters."""
        self._require_loaded()
        assert self._cluster_index is not None
        return len(self._cluster_index)

    @property
    def metadata(self) -> dict[str, Any]:
        """Return the stored metadata dict."""
        self._require_loaded()
        assert self._metadata is not None
        return self._metadata

    # ------------------------------------------------------------------
    # Access
    # ------------------------------------------------------------------

    def get_embeddings(self, indices: np.ndarray | list[int]) -> np.ndarray:
        """Fetch embeddings by index via memory-mapped random access.

        Args:
            indices: 1-D array or list of passage indices.

        Returns:
            Dense array of shape ``(len(indices), embedding_dim)``.
        """
        self._require_loaded()
        assert self._embeddings is not None
        idx = np.asarray(indices, dtype=np.intp)
        # Explicitly copy out of the mmap so the caller owns the memory.
        return np.array(self._embeddings[idx])

    def get_cluster_members(self, cluster_id: int) -> np.ndarray:
        """Return all passage indices belonging to a cluster.

        Args:
            cluster_id: The cluster identifier.

        Returns:
            1-D int array of passage indices.
        """
        self._require_loaded()
        assert self._cluster_index is not None
        members = self._cluster_index.get(cluster_id)
        if members is None:
            raise KeyError(f"Unknown cluster_id: {cluster_id}")
        return np.array(members, dtype=np.intp)

    def sample_from_cluster(
        self, cluster_id: int, n: int, rng: np.random.Generator | None = None
    ) -> np.ndarray:
        """Randomly sample ``n`` passage indices from a single cluster.

        If the cluster contains fewer than ``n`` members, sampling is done
        with replacement.

        Args:
            cluster_id: Cluster to sample from.
            n: Number of indices to draw.
            rng: Optional NumPy random generator for reproducibility.

        Returns:
            1-D int array of length ``n``.
        """
        if rng is None:
            rng = np.random.default_rng()
        members = self.get_cluster_members(cluster_id)
        replace = len(members) < n
        chosen = rng.choice(members, size=n, replace=replace)
        return chosen

    def sample_from_different_cluster(
        self,
        exclude_cluster: int,
        n: int,
        rng: np.random.Generator | None = None,
    ) -> np.ndarray:
        """Sample ``n`` passage indices from clusters other than ``exclude_cluster``.

        A cluster is chosen uniformly at random from eligible clusters, then
        an index is drawn from that cluster.  This is repeated ``n`` times.

        Args:
            exclude_cluster: Cluster ID to exclude.
            n: Number of indices to draw.
            rng: Optional NumPy random generator.

        Returns:
            1-D int array of length ``n``.
        """
        if rng is None:
            rng = np.random.default_rng()
        self._require_loaded()
        assert self._cluster_index is not None

        eligible_ids = [cid for cid in self._cluster_index if cid != exclude_cluster]
        if not eligible_ids:
            raise ValueError(f"No clusters available after excluding cluster {exclude_cluster}")

        chosen_clusters = rng.choice(eligible_ids, size=n, replace=True)
        result = np.empty(n, dtype=np.intp)
        for i, cid in enumerate(chosen_clusters):
            members = self._cluster_index[cid]
            result[i] = rng.choice(members)
        return result

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _require_loaded(self) -> None:
        """Raise if :meth:`load` has not been called."""
        if self._embeddings is None:
            raise RuntimeError("Store not loaded. Call .load() or use EmbeddingStore.create().")

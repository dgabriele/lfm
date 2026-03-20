"""Stratified batch sampler with curriculum difficulty for embedding games.

Reads precomputed embeddings from an ``EmbeddingStore`` and produces
GPU-ready batches for training the language faculty via embedding
reconstruction and referential games.  All sampling logic runs on the CPU
using NumPy; conversion to PyTorch tensors happens once at the end of each
``sample_batch`` / ``sample_referential_batch`` call.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import torch

from lfm.utils.logging import get_logger

if TYPE_CHECKING:
    from lfm.embeddings.config import SamplerConfig
    from lfm.embeddings.store import EmbeddingStore

logger = get_logger(__name__)


class StratifiedSampler:
    """GPU-centric stratified sampler with curriculum difficulty control.

    Implements ``__iter__`` and ``__next__`` for infinite iteration.  Each
    call to ``__next__`` returns a reconstruction batch and advances the
    curriculum by one step.

    The sampler round-robins across clusters so that every cluster is
    visited equally, then mixes breadth (cross-cluster) and depth
    (within-cluster) samples according to ``within_cluster_ratio``.

    Args:
        config: Sampler hyper-parameters (batch size, negatives, curriculum).
        store: A loaded ``EmbeddingStore`` instance.
    """

    def __init__(self, config: SamplerConfig, store: EmbeddingStore) -> None:
        self._config = config
        self._store = store
        self._cluster_ids = list(range(store.num_clusters))
        self._cluster_cursor: int = 0
        self._step: int = 0

    # ------------------------------------------------------------------
    # Curriculum
    # ------------------------------------------------------------------

    @property
    def curriculum_difficulty(self) -> float:
        """Linear ramp from ``curriculum_start`` to ``curriculum_end``.

        Returns a float in ``[start, end]`` that indicates how hard the
        current negative sampling should be.  ``0`` means easy (cross-cluster
        negatives), ``1`` means hard (within-cluster negatives).
        """
        warmup = max(self._config.curriculum_warmup_steps, 1)
        progress = min(self._step / warmup, 1.0)
        start = self._config.curriculum_start
        end = self._config.curriculum_end
        return start + (end - start) * progress

    def step_curriculum(self) -> None:
        """Advance the curriculum by one step."""
        self._step += 1

    # ------------------------------------------------------------------
    # Reconstruction batches
    # ------------------------------------------------------------------

    def sample_batch(self) -> dict[str, torch.Tensor]:
        """Sample one reconstruction batch.

        Uses ``within_cluster_ratio`` to control the mix between
        within-cluster (depth) and cross-cluster (breadth) samples.

        Returns:
            Dictionary with key ``"agent_state"`` mapping to a tensor of
            shape ``(batch_size, embedding_dim)``.
        """
        bs = self._config.batch_size
        n_within = int(bs * self._config.within_cluster_ratio)
        n_breadth = bs - n_within

        index_parts: list[np.ndarray] = []

        # Depth: samples from the current cluster (round-robin)
        if n_within > 0:
            cluster_id = self._cluster_ids[self._cluster_cursor]
            within_indices = self._store.sample_from_cluster(cluster_id, n_within)
            index_parts.append(within_indices)
            self._advance_cursor()

        # Breadth: stratified samples across all clusters
        if n_breadth > 0:
            breadth_indices, _ = self._select_anchors(n_breadth)
            index_parts.append(breadth_indices)

        all_indices = (
            np.concatenate(index_parts, axis=0) if len(index_parts) > 1 else index_parts[0]
        )

        # Shuffle so within/breadth order is not predictable
        perm = np.random.permutation(len(all_indices))
        all_indices = all_indices[perm]

        embeddings = self._store.get_embeddings(all_indices)

        return {"agent_state": torch.from_numpy(embeddings.astype(np.float32))}

    # ------------------------------------------------------------------
    # Referential batches
    # ------------------------------------------------------------------

    def sample_referential_batch(self) -> dict[str, torch.Tensor]:
        """Sample one referential batch with hard negatives.

        For each anchor, negatives are drawn from the same cluster with
        probability equal to ``curriculum_difficulty`` and from a different
        cluster otherwise.

        Returns:
            Dictionary with keys:

            - ``"agent_state"``: ``(batch_size, dim)`` anchor embeddings.
            - ``"distractors"``: ``(batch_size, num_negatives, dim)``
              distractor embeddings.
            - ``"target_idx"``: ``(batch_size,)`` zeros (target is always
              at position 0 before the phase shuffles candidates).
        """
        bs = self._config.batch_size
        num_neg = self._config.num_negatives
        difficulty = self.curriculum_difficulty

        anchor_indices, anchor_clusters = self._select_anchors(bs)
        anchor_emb = self._store.get_embeddings(anchor_indices)

        neg_indices = self._select_hard_negatives(
            anchor_indices, anchor_clusters, difficulty
        )  # (bs, num_neg)

        # Gather negative embeddings
        neg_flat = neg_indices.reshape(-1)
        neg_emb_flat = self._store.get_embeddings(neg_flat)  # (bs * num_neg, dim)
        neg_emb = neg_emb_flat.reshape(bs, num_neg, -1)

        return {
            "agent_state": torch.from_numpy(anchor_emb.astype(np.float32)),
            "distractors": torch.from_numpy(neg_emb.astype(np.float32)),
            "target_idx": torch.zeros(bs, dtype=torch.long),
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _select_anchors(self, n: int) -> tuple[np.ndarray, np.ndarray]:
        """Stratified anchor selection via round-robin across clusters.

        Args:
            n: Number of anchors to select.

        Returns:
            Tuple of ``(indices, cluster_ids)`` each of shape ``(n,)``.
        """
        indices_list: list[int] = []
        clusters_list: list[int] = []

        num_clusters = len(self._cluster_ids)
        per_cluster = max(n // num_clusters, 1)
        remaining = n

        while remaining > 0:
            cluster_id = self._cluster_ids[self._cluster_cursor]
            take = min(per_cluster, remaining)

            members = self._store.get_cluster_members(cluster_id)
            if len(members) == 0:
                self._advance_cursor()
                continue

            replace = len(members) < take
            chosen = np.random.choice(members, size=take, replace=replace)
            indices_list.extend(chosen.tolist())
            clusters_list.extend([cluster_id] * take)

            remaining -= take
            self._advance_cursor()

        return (
            np.array(indices_list[:n], dtype=np.int64),
            np.array(clusters_list[:n], dtype=np.int32),
        )

    def _select_hard_negatives(
        self,
        anchor_indices: np.ndarray,
        anchor_clusters: np.ndarray,
        difficulty: float,
    ) -> np.ndarray:
        """Select negatives with hardness controlled by difficulty.

        For each anchor, each negative is independently drawn from:
        - the *same* cluster with probability ``difficulty`` (hard)
        - a *different* cluster with probability ``1 - difficulty`` (easy)

        Args:
            anchor_indices: ``(n,)`` passage indices of anchors.
            anchor_clusters: ``(n,)`` cluster IDs of anchors.
            difficulty: Float in ``[0, 1]`` controlling hard-negative ratio.

        Returns:
            Array of shape ``(n, num_negatives)`` with passage indices.
        """
        n = len(anchor_indices)
        num_neg = self._config.num_negatives
        result = np.empty((n, num_neg), dtype=np.int64)

        for i in range(n):
            cluster_id = int(anchor_clusters[i])
            anchor_idx = int(anchor_indices[i])

            for j in range(num_neg):
                use_hard = np.random.random() < difficulty
                if use_hard:
                    # Same cluster (hard negative)
                    members = self._store.get_cluster_members(cluster_id)
                    # Exclude the anchor itself
                    members = members[members != anchor_idx]
                    if len(members) == 0:
                        # Fall back to cross-cluster if cluster is too small
                        result[i, j] = self._sample_index_from_different_cluster(cluster_id)
                    else:
                        result[i, j] = np.random.choice(members)
                else:
                    # Different cluster (easy negative)
                    result[i, j] = self._sample_index_from_different_cluster(cluster_id)

        return result

    def _sample_index_from_different_cluster(self, exclude_cluster: int) -> int:
        """Sample a single passage index from any cluster except the excluded one.

        Args:
            exclude_cluster: Cluster ID to exclude.

        Returns:
            A single passage index.
        """
        other_clusters = [c for c in self._cluster_ids if c != exclude_cluster]
        if not other_clusters:
            # Only one cluster -- fall back to random index from the store
            return int(np.random.randint(0, self._store.num_passages))
        chosen_cluster = int(np.random.choice(other_clusters))
        members = self._store.get_cluster_members(chosen_cluster)
        if len(members) == 0:
            return int(np.random.randint(0, self._store.num_passages))
        return int(np.random.choice(members))

    def _advance_cursor(self) -> None:
        """Advance the round-robin cluster cursor by one."""
        self._cluster_cursor = (self._cluster_cursor + 1) % len(self._cluster_ids)

    # ------------------------------------------------------------------
    # Iterator protocol
    # ------------------------------------------------------------------

    def __iter__(self) -> StratifiedSampler:
        return self

    def __next__(self) -> dict[str, torch.Tensor]:
        """Return the next reconstruction batch and advance the curriculum."""
        batch = self.sample_batch()
        self.step_curriculum()
        return batch

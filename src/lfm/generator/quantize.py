"""Vector quantization for VQ-VAE and Residual VQ-VAE.

Implements the discrete latent space used in VQ-VAE (van den Oord et al.,
2017) and Residual VQ (Zeghidour et al., SoundStream 2021).  Replaces
the continuous Gaussian latent of the standard VAE with discrete codebook
lookups, eliminating posterior collapse by construction.

The decoder receives quantized vectors (sum of codebook entries across
levels), so the entire decoder architecture is unchanged.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn


class VectorQuantizer(nn.Module):
    """Single-codebook vector quantizer with straight-through estimator.

    Maps continuous vectors to their nearest codebook entry.  Supports
    both gradient-based and exponential moving average (EMA) codebook
    updates.

    Args:
        codebook_size: Number of codebook entries (K).
        embedding_dim: Dimension of each entry (D = latent_dim).
        commitment_weight: Weight for commitment loss ``||z_e - sg(e)||²``.
        ema_update: Use EMA codebook updates (stable) vs gradient (noisy).
        ema_decay: Decay rate for EMA updates.
        epsilon: Laplace smoothing for EMA cluster sizes.
    """

    def __init__(
        self,
        codebook_size: int,
        embedding_dim: int,
        commitment_weight: float = 0.25,
        ema_update: bool = True,
        ema_decay: float = 0.99,
        epsilon: float = 1e-5,
    ) -> None:
        super().__init__()
        self.codebook_size = codebook_size
        self.embedding_dim = embedding_dim
        self.commitment_weight = commitment_weight
        self.ema_update = ema_update
        self.ema_decay = ema_decay
        self.epsilon = epsilon

        # Codebook: K entries of dimension D
        self.embedding = nn.Embedding(codebook_size, embedding_dim)
        self.embedding.weight.data.uniform_(
            -1.0 / codebook_size, 1.0 / codebook_size,
        )

        if ema_update:
            # EMA state — not optimized, updated manually during forward
            self.register_buffer(
                "_ema_cluster_size", torch.zeros(codebook_size),
            )
            self.register_buffer(
                "_ema_embedding_sum", self.embedding.weight.data.clone(),
            )

    def forward(self, z: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """Quantize input vectors to nearest codebook entries.

        Args:
            z: Continuous input ``(batch, embedding_dim)``.

        Returns:
            Tuple of ``(quantized, commitment_loss, indices)``:
              - quantized: ``(batch, embedding_dim)`` with straight-through grad.
              - commitment_loss: Scalar loss (already weighted).
              - indices: ``(batch,)`` integer codebook indices.
        """
        # Pairwise distances: ||z - e||² = ||z||² + ||e||² - 2·z@eᵀ
        z_flat = z.float()
        codebook = self.embedding.weight.float()

        distances = (
            z_flat.pow(2).sum(dim=-1, keepdim=True)
            + codebook.pow(2).sum(dim=-1, keepdim=False)
            - 2 * z_flat @ codebook.T
        )  # (B, K)

        # Nearest codebook entry
        indices = distances.argmin(dim=-1)  # (B,)
        quantized = self.embedding(indices)  # (B, D)

        # Commitment loss: encourage encoder to commit to codebook entries
        commitment_loss = (z - quantized.detach()).pow(2).mean()

        # EMA codebook update (during training only)
        if self.ema_update and self.training:
            with torch.no_grad():
                # One-hot encoding of assignments
                encodings = torch.zeros(
                    z.size(0), self.codebook_size,
                    device=z.device, dtype=z.dtype,
                )
                encodings.scatter_(1, indices.unsqueeze(1), 1.0)

                # Update cluster sizes
                batch_cluster_size = encodings.sum(dim=0)
                self._ema_cluster_size.mul_(self.ema_decay).add_(
                    batch_cluster_size, alpha=1 - self.ema_decay,
                )

                # Update embedding sums
                batch_embedding_sum = encodings.T @ z_flat
                self._ema_embedding_sum.mul_(self.ema_decay).add_(
                    batch_embedding_sum, alpha=1 - self.ema_decay,
                )

                # Laplace smoothing + normalize
                n = self._ema_cluster_size.sum()
                smoothed = (
                    (self._ema_cluster_size + self.epsilon)
                    / (n + self.codebook_size * self.epsilon)
                    * n
                )
                self.embedding.weight.data.copy_(
                    self._ema_embedding_sum / smoothed.unsqueeze(1)
                )

        # Straight-through estimator: forward uses quantized,
        # backward passes gradient to z as if quantized == z
        quantized = z + (quantized - z).detach()

        return quantized, self.commitment_weight * commitment_loss, indices

    def reset_dead_codes(self, threshold: float = 1.0, epsilon: float = 0.01) -> int:
        """Replace dead codebook entries by splitting high-usage codes.

        Dead codes (usage below ``threshold``) are replaced with
        perturbations of the most-used codes, scaled by each parent
        code's internal variance.  This targets the information
        bottleneck where it's tightest — codes covering too many
        diverse inputs get split into finer-grained representations.

        EMA state is split proportionally so the new codes don't
        immediately decay back to the dead position.

        Args:
            threshold: Minimum EMA cluster size to be considered alive.
            epsilon: Perturbation scale multiplier.

        Returns:
            Number of codes reset.
        """
        if not self.ema_update:
            return 0

        alive_mask = self._ema_cluster_size > threshold
        dead_indices = (~alive_mask).nonzero(as_tuple=True)[0]
        n_dead = dead_indices.size(0)

        if n_dead == 0:
            return 0

        # Rank alive codes by usage (highest first)
        alive_indices = alive_mask.nonzero(as_tuple=True)[0]
        if alive_indices.size(0) == 0:
            return 0
        alive_usage = self._ema_cluster_size[alive_indices]
        _, sorted_order = alive_usage.sort(descending=True)
        alive_sorted = alive_indices[sorted_order]

        with torch.no_grad():
            for i, dead_idx in enumerate(dead_indices):
                # Pick highest-usage parent (cycle if more dead than alive)
                parent_idx = alive_sorted[i % alive_sorted.size(0)]
                parent_embed = self.embedding.weight.data[parent_idx]

                # Compute perturbation from parent's internal variance.
                # variance ≈ E[x²] - E[x]² estimated from EMA sums.
                parent_count = self._ema_cluster_size[parent_idx].clamp(min=1)
                parent_mean = self._ema_embedding_sum[parent_idx] / parent_count
                # Use parent_mean as the centroid; perturbation is random
                # scaled by the distance from mean to embedding (a proxy
                # for internal spread when exact variance isn't tracked).
                spread = (parent_embed - parent_mean).abs().clamp(min=1e-4)
                noise = torch.randn_like(parent_embed) * spread * epsilon

                # Split: dead = parent + noise, parent = parent - noise
                self.embedding.weight.data[dead_idx] = parent_embed + noise
                self.embedding.weight.data[parent_idx] = parent_embed - noise

                # Split EMA state proportionally
                half_count = parent_count / 2
                self._ema_cluster_size[dead_idx] = half_count
                self._ema_cluster_size[parent_idx] = half_count
                parent_sum = self._ema_embedding_sum[parent_idx].clone()
                self._ema_embedding_sum[dead_idx] = parent_sum / 2
                self._ema_embedding_sum[parent_idx] = parent_sum / 2

        return n_dead

    def encode(self, z: Tensor) -> Tensor:
        """Encode to codebook indices (inference, no grad).

        Args:
            z: ``(batch, embedding_dim)``.

        Returns:
            ``(batch,)`` integer codebook indices.
        """
        with torch.no_grad():
            distances = (
                z.float().pow(2).sum(dim=-1, keepdim=True)
                + self.embedding.weight.float().pow(2).sum(dim=-1)
                - 2 * z.float() @ self.embedding.weight.float().T
            )
            return distances.argmin(dim=-1)

    def decode(self, indices: Tensor) -> Tensor:
        """Look up codebook entries by index.

        Args:
            indices: ``(batch,)`` integer indices.

        Returns:
            ``(batch, embedding_dim)`` codebook vectors.
        """
        return self.embedding(indices)


class ResidualVQ(nn.Module):
    """Residual vector quantization with N codebook levels.

    Level 0 captures coarse structure.  Each subsequent level
    quantizes the residual error from the previous level, capturing
    progressively finer detail.  The final quantized output is the
    sum of all codebook entries across levels.

    With ``num_levels=1``, this is equivalent to standard VQ-VAE.

    Args:
        num_levels: Number of quantization levels (N).
        codebook_size: Entries per codebook (K).
        embedding_dim: Dimension of each entry (D = latent_dim).
        commitment_weight: Per-level commitment loss weight.
        ema_update: Use EMA codebook updates.
        ema_decay: EMA decay rate.
    """

    def __init__(
        self,
        num_levels: int = 4,
        codebook_size: int = 512,
        embedding_dim: int = 256,
        commitment_weight: float = 0.25,
        ema_update: bool = True,
        ema_decay: float = 0.99,
    ) -> None:
        super().__init__()
        self.num_levels = num_levels
        self.codebook_size = codebook_size
        self.embedding_dim = embedding_dim

        self.levels = nn.ModuleList([
            VectorQuantizer(
                codebook_size=codebook_size,
                embedding_dim=embedding_dim,
                commitment_weight=commitment_weight,
                ema_update=ema_update,
                ema_decay=ema_decay,
            )
            for _ in range(num_levels)
        ])

        # Running codebook utilization tracking (per level)
        self.register_buffer(
            "_usage_counts",
            torch.zeros(num_levels, codebook_size),
            persistent=False,
        )
        self._usage_total = 0

    def forward(self, z: Tensor) -> tuple[Tensor, Tensor, list[Tensor]]:
        """Residual quantization across all levels.

        Args:
            z: Continuous input ``(batch, embedding_dim)``.

        Returns:
            Tuple of ``(quantized, total_loss, all_indices)``:
              - quantized: ``(batch, embedding_dim)`` sum of all levels.
              - total_loss: Scalar sum of per-level commitment losses.
              - all_indices: List of N ``(batch,)`` index tensors.
        """
        residual = z
        quantized_sum = torch.zeros_like(z)
        total_loss = torch.tensor(0.0, device=z.device, dtype=z.dtype)
        all_indices: list[Tensor] = []

        for i, level in enumerate(self.levels):
            quantized_level, loss_level, indices_level = level(residual)
            # Each level trains on its own residual — detach prevents
            # gradients from flowing through the residual chain
            residual = residual - quantized_level.detach()
            quantized_sum = quantized_sum + quantized_level
            total_loss = total_loss + loss_level
            all_indices.append(indices_level)

            # Track codebook utilization
            if self.training:
                with torch.no_grad():
                    self._usage_counts[i].scatter_add_(
                        0, indices_level, torch.ones_like(indices_level, dtype=torch.float),
                    )
                    self._usage_total += indices_level.size(0)

        return quantized_sum, total_loss, all_indices

    def encode(self, z: Tensor) -> list[Tensor]:
        """Encode to codebook indices at all levels (inference).

        Args:
            z: ``(batch, embedding_dim)``.

        Returns:
            List of N ``(batch,)`` index tensors.
        """
        all_indices: list[Tensor] = []
        residual = z

        with torch.no_grad():
            for level in self.levels:
                indices = level.encode(residual)
                quantized = level.decode(indices)
                residual = residual - quantized
                all_indices.append(indices)

        return all_indices

    def decode(self, all_indices: list[Tensor]) -> Tensor:
        """Reconstruct from multi-level codebook indices.

        Args:
            all_indices: List of N ``(batch,)`` index tensors.

        Returns:
            ``(batch, embedding_dim)`` sum of codebook vectors.
        """
        quantized = torch.zeros(
            all_indices[0].size(0), self.embedding_dim,
            device=all_indices[0].device,
            dtype=self.levels[0].embedding.weight.dtype,
        )
        for level, indices in zip(self.levels, all_indices):
            quantized = quantized + level.decode(indices)
        return quantized

    @property
    def utilization(self) -> list[float]:
        """Fraction of codebook entries used at each level (0.0–1.0).

        Based on running counts since last ``reset_usage()``.
        """
        if self._usage_total == 0:
            return [0.0] * self.num_levels
        return [
            (self._usage_counts[i] > 0).float().mean().item()
            for i in range(self.num_levels)
        ]

    def reset_usage(self) -> None:
        """Reset codebook utilization counters (call per epoch)."""
        self._usage_counts.zero_()
        self._usage_total = 0

    def reset_dead_codes(self, threshold: float = 1.0, epsilon: float = 0.01) -> list[int]:
        """Reset dead codes at all levels by splitting high-usage codes.

        Args:
            threshold: Minimum EMA cluster size to be considered alive.
            epsilon: Perturbation scale multiplier.

        Returns:
            List of reset counts per level.
        """
        return [level.reset_dead_codes(threshold, epsilon) for level in self.levels]

    @property
    def total_codebook_size(self) -> int:
        """Total number of possible code combinations."""
        return self.codebook_size ** self.num_levels


class GroupedVQ(nn.Module):
    """Product-quantized VQ with independent codebooks per latent group.

    Splits the latent vector into ``num_groups`` contiguous slices, each
    quantized independently by its own codebook.  The quantized output
    is the concatenation of all group outputs, preserving the original
    dimensionality.

    Diversity is **multiplicative**: with G groups of K codes each,
    the combinatorial space is K^G.  Even modest per-group utilization
    yields enormous effective diversity.

    Example: 8 groups × 64 codes × 32 dims each = 256-dim latent
    with 64^8 ≈ 2.8 × 10^14 possible code combinations from only
    512 total codebook vectors.

    Args:
        num_groups: Number of independent groups (G).
        codebook_size: Entries per group codebook (K).
        embedding_dim: Total latent dimension (must be divisible by num_groups).
        commitment_weight: Per-group commitment loss weight.
        ema_update: Use EMA codebook updates.
        ema_decay: EMA decay rate.
    """

    def __init__(
        self,
        num_groups: int = 8,
        codebook_size: int = 64,
        embedding_dim: int = 256,
        commitment_weight: float = 0.25,
        ema_update: bool = True,
        ema_decay: float = 0.99,
    ) -> None:
        super().__init__()

        if embedding_dim % num_groups != 0:
            raise ValueError(
                f"embedding_dim ({embedding_dim}) must be divisible by "
                f"num_groups ({num_groups})"
            )

        self.num_groups = num_groups
        self.codebook_size = codebook_size
        self.embedding_dim = embedding_dim
        self.group_dim = embedding_dim // num_groups

        self.groups = nn.ModuleList([
            VectorQuantizer(
                codebook_size=codebook_size,
                embedding_dim=self.group_dim,
                commitment_weight=commitment_weight,
                ema_update=ema_update,
                ema_decay=ema_decay,
            )
            for _ in range(num_groups)
        ])

        # Utilization tracking
        self.register_buffer(
            "_usage_counts",
            torch.zeros(num_groups, codebook_size),
            persistent=False,
        )
        self._usage_total = 0

    def forward(self, z: Tensor) -> tuple[Tensor, Tensor, list[Tensor]]:
        """Quantize each group independently and concatenate.

        Args:
            z: Continuous input ``(batch, embedding_dim)``.

        Returns:
            Tuple of ``(quantized, total_loss, all_indices)``:
              - quantized: ``(batch, embedding_dim)`` concatenation of groups.
              - total_loss: Scalar sum of per-group commitment losses.
              - all_indices: List of G ``(batch,)`` index tensors.
        """
        chunks = z.split(self.group_dim, dim=-1)  # G tensors of (B, group_dim)
        quantized_parts: list[Tensor] = []
        total_loss = torch.tensor(0.0, device=z.device, dtype=z.dtype)
        all_indices: list[Tensor] = []

        for i, (group, chunk) in enumerate(zip(self.groups, chunks)):
            quantized_g, loss_g, indices_g = group(chunk)
            quantized_parts.append(quantized_g)
            total_loss = total_loss + loss_g
            all_indices.append(indices_g)

            if self.training:
                with torch.no_grad():
                    self._usage_counts[i].scatter_add_(
                        0, indices_g,
                        torch.ones_like(indices_g, dtype=torch.float),
                    )
                    self._usage_total += indices_g.size(0)

        quantized = torch.cat(quantized_parts, dim=-1)  # (B, embedding_dim)
        return quantized, total_loss, all_indices

    def encode(self, z: Tensor) -> list[Tensor]:
        """Encode to per-group codebook indices (inference).

        Args:
            z: ``(batch, embedding_dim)``.

        Returns:
            List of G ``(batch,)`` index tensors.
        """
        chunks = z.split(self.group_dim, dim=-1)
        return [group.encode(chunk) for group, chunk in zip(self.groups, chunks)]

    def decode(self, all_indices: list[Tensor]) -> Tensor:
        """Reconstruct from per-group codebook indices.

        Args:
            all_indices: List of G ``(batch,)`` index tensors.

        Returns:
            ``(batch, embedding_dim)`` concatenation of group decodings.
        """
        parts = [
            group.decode(indices)
            for group, indices in zip(self.groups, all_indices)
        ]
        return torch.cat(parts, dim=-1)

    @property
    def utilization(self) -> list[float]:
        """Fraction of codebook entries used per group."""
        if self._usage_total == 0:
            return [0.0] * self.num_groups
        return [
            (self._usage_counts[i] > 0).float().mean().item()
            for i in range(self.num_groups)
        ]

    def reset_usage(self) -> None:
        """Reset utilization counters."""
        self._usage_counts.zero_()
        self._usage_total = 0

    def reset_dead_codes(
        self, threshold: float = 1.0, epsilon: float = 0.01,
    ) -> list[int]:
        """Reset dead codes in all groups."""
        return [g.reset_dead_codes(threshold, epsilon) for g in self.groups]

    @property
    def total_codebook_size(self) -> int:
        """Total combinatorial space across all groups."""
        return self.codebook_size ** self.num_groups

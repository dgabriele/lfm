"""Latent space smoothness (Lipschitz continuity) analysis.

Measures how smoothly the decoder maps nearby latent vectors to similar
outputs.  Three figures:

1. **Lipschitz Continuity Curve** — z L2 distance vs output edit distance.
2. **Token Jaccard vs z Distance** — content overlap vs latent distance.
3. **Interpolation Continuity** — edit distance from origin along 50
   interpolation paths.
"""

from __future__ import annotations

import logging
from collections import Counter

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.figure import Figure
from scipy.stats import spearmanr

from lfm.visualize import BaseVisualization
from lfm.visualize.config import VisualizeConfig
from lfm.visualize.loader import decode_z
from lfm.visualize.style import FIGSIZE_SINGLE, FIGSIZE_WIDE, apply_style

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _edit_distance(a: list[int], b: list[int]) -> int:
    """Space-optimised Levenshtein distance (O(min(m,n)) memory)."""
    if not a:
        return len(b)
    if not b:
        return len(a)
    m, n = len(a), len(b)
    dp = list(range(n + 1))
    for i in range(1, m + 1):
        prev, dp[0] = dp[0], i
        for j in range(1, n + 1):
            temp = dp[j]
            dp[j] = min(
                dp[j] + 1,
                dp[j - 1] + 1,
                prev + (0 if a[i - 1] == b[j - 1] else 1),
            )
            prev = temp
    return dp[n]


def _jaccard_distance(a: list[int], b: list[int]) -> float:
    """Token multiset Jaccard distance: 1 - |intersection| / |union|."""
    if not a and not b:
        return 0.0
    ca, cb = Counter(a), Counter(b)
    intersection = sum((ca & cb).values())
    union = sum((ca | cb).values())
    if union == 0:
        return 0.0
    return 1.0 - intersection / union


class SmoothnessVisualization(BaseVisualization):
    """Measure and visualise Lipschitz continuity of the z -> output mapping."""

    @property
    def name(self) -> str:
        return "smoothness"

    # ------------------------------------------------------------------
    # Pair-wise analysis
    # ------------------------------------------------------------------

    def _pairwise_distances(
        self, data: dict, n_pairs: int = 500,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Sample z pairs at *varying* distances and compute output distances.

        Creates pairs at a range of latent distances by mixing:
        - Very close pairs (small perturbations of the same z)
        - Medium pairs (random corpus pairs from same language bucket)
        - Far pairs (random corpus pairs)

        Returns:
            (z_dists, edit_dists, jaccard_dists) — each shape ``(n_pairs,)``.
        """
        z = data["z"]
        n = z.size(0)
        rng = np.random.RandomState(self.config.seed)

        n_close = n_pairs // 3
        n_medium = n_pairs // 3
        n_far = n_pairs - n_close - n_medium

        z_a_list: list[torch.Tensor] = []
        z_b_list: list[torch.Tensor] = []

        # Close pairs: z + small perturbation (varying scales)
        base_idx = rng.choice(n, size=n_close, replace=True)
        scales = np.geomspace(0.01, 1.0, n_close)  # log-spaced noise scales
        z_std = z.std(dim=0, keepdim=True)
        for i, idx in enumerate(base_idx):
            z_a_list.append(z[idx].unsqueeze(0))
            noise = torch.randn_like(z[idx]) * z_std.squeeze(0) * scales[i]
            z_b_list.append((z[idx] + noise).unsqueeze(0))

        # Medium pairs: random pairs (moderate distance)
        idx_a = rng.choice(n, size=n_medium, replace=True)
        idx_b = rng.choice(n, size=n_medium, replace=True)
        for a, b in zip(idx_a, idx_b):
            z_a_list.append(z[a].unsqueeze(0))
            # Interpolate halfway toward b for medium distance
            z_b_list.append(((z[a] + z[b]) / 2).unsqueeze(0))

        # Far pairs: fully random pairs
        idx_a = rng.choice(n, size=n_far, replace=True)
        idx_b = rng.choice(n, size=n_far, replace=True)
        for a, b in zip(idx_a, idx_b):
            z_a_list.append(z[a].unsqueeze(0))
            z_b_list.append(z[b].unsqueeze(0))

        z_a = torch.cat(z_a_list, dim=0)
        z_b = torch.cat(z_b_list, dim=0)
        z_dists = torch.norm(z_a - z_b, dim=-1).cpu().numpy()

        # Decode all (interleave a and b for batched decoding)
        z_all = torch.cat([z_a, z_b], dim=0)  # (2*n_pairs, D)
        logger.info("Decoding %d z vectors for pairwise analysis...", z_all.size(0))
        decoded = decode_z(z_all, data, self.config, temperature=0.8, top_p=0.9)

        total = z_a.size(0)
        edit_dists = np.empty(total, dtype=np.float64)
        jaccard_dists = np.empty(total, dtype=np.float64)

        for i in range(total):
            seq_a = decoded[i]
            seq_b = decoded[total + i]
            edit_dists[i] = _edit_distance(seq_a, seq_b)
            jaccard_dists[i] = _jaccard_distance(seq_a, seq_b)

        return z_dists, edit_dists, jaccard_dists

    # ------------------------------------------------------------------
    # Interpolation continuity
    # ------------------------------------------------------------------

    def _interpolation_continuity(
        self, data: dict, n_interp_pairs: int = 50, n_steps: int = 11,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Interpolate between random z pairs and measure edit distance from t=0.

        Returns:
            ts: shape ``(n_steps,)`` — interpolation parameter 0..1.
            all_curves: shape ``(n_interp_pairs, n_steps)`` — edit distances.
        """
        z = data["z"]
        n = z.size(0)
        rng = np.random.RandomState(self.config.seed)
        idx_a = rng.choice(n, size=n_interp_pairs, replace=False)
        idx_b = rng.choice(n, size=n_interp_pairs, replace=False)

        ts = np.linspace(0.0, 1.0, n_steps)

        # Build all interpolation points: (n_interp_pairs * n_steps, latent_dim)
        interp_z_list: list[torch.Tensor] = []
        for i in range(n_interp_pairs):
            za = z[idx_a[i]].unsqueeze(0)  # (1, D)
            zb = z[idx_b[i]].unsqueeze(0)
            for t in ts:
                interp_z_list.append((1 - t) * za + t * zb)

        interp_z = torch.cat(interp_z_list, dim=0)  # (n_pairs * n_steps, D)

        logger.info(
            "Decoding %d interpolation points (%d pairs x %d steps)...",
            interp_z.size(0), n_interp_pairs, n_steps,
        )
        decoded = decode_z(interp_z, data, self.config, temperature=0.8, top_p=0.9)

        # Compute edit distance from t=0 for each pair
        all_curves = np.zeros((n_interp_pairs, n_steps), dtype=np.float64)
        for i in range(n_interp_pairs):
            base_idx = i * n_steps
            origin_seq = decoded[base_idx]  # t=0
            for j in range(n_steps):
                all_curves[i, j] = _edit_distance(origin_seq, decoded[base_idx + j])

        return ts, all_curves

    # ------------------------------------------------------------------
    # Figures
    # ------------------------------------------------------------------

    def _fig_lipschitz(
        self, z_dists: np.ndarray, edit_dists: np.ndarray,
    ) -> Figure:
        """Figure 1: Lipschitz Continuity Curve (scatter + regression)."""
        fig, ax = plt.subplots(figsize=FIGSIZE_SINGLE)

        ax.scatter(z_dists, edit_dists, s=10, alpha=0.4, c="#1f77b4", edgecolors="none")

        # Regression line
        coeffs = np.polyfit(z_dists, edit_dists, deg=1)
        x_line = np.linspace(z_dists.min(), z_dists.max(), 200)
        y_line = np.polyval(coeffs, x_line)
        rho, pval = spearmanr(z_dists, edit_dists)
        ax.plot(
            x_line, y_line, color="#d62728", linewidth=2,
            label=f"OLS fit (Spearman r={rho:.3f}, p={pval:.2e})",
        )

        ax.set_xlabel("z L2 Distance")
        ax.set_ylabel("Output Edit Distance")
        ax.set_title("Latent Space Smoothness (z distance vs output edit distance)")
        ax.legend(loc="upper left")
        fig.tight_layout()
        return fig

    def _fig_jaccard(
        self, z_dists: np.ndarray, jaccard_dists: np.ndarray,
    ) -> Figure:
        """Figure 2: Token Jaccard distance vs z distance."""
        fig, ax = plt.subplots(figsize=FIGSIZE_SINGLE)

        ax.scatter(
            z_dists, jaccard_dists, s=10, alpha=0.4, c="#2ca02c", edgecolors="none",
        )

        coeffs = np.polyfit(z_dists, jaccard_dists, deg=1)
        x_line = np.linspace(z_dists.min(), z_dists.max(), 200)
        y_line = np.polyval(coeffs, x_line)
        rho, pval = spearmanr(z_dists, jaccard_dists)
        ax.plot(
            x_line, y_line, color="#d62728", linewidth=2,
            label=f"OLS fit (Spearman r={rho:.3f}, p={pval:.2e})",
        )

        ax.set_xlabel("z L2 Distance")
        ax.set_ylabel("Token Jaccard Distance")
        ax.set_title("Token Overlap vs Latent Distance")
        ax.legend(loc="upper left")
        fig.tight_layout()
        return fig

    def _fig_interpolation_continuity(
        self, ts: np.ndarray, all_curves: np.ndarray,
    ) -> Figure:
        """Figure 3: Interpolation continuity — edit distance from origin."""
        fig, ax = plt.subplots(figsize=FIGSIZE_WIDE)

        # Individual curves in light gray
        for i in range(all_curves.shape[0]):
            ax.plot(ts, all_curves[i], color="#cccccc", alpha=0.3, linewidth=0.8)

        # Mean curve in bold
        mean_curve = all_curves.mean(axis=0)
        ax.plot(
            ts, mean_curve, color="#1f77b4", linewidth=2.5,
            label=f"Mean (n={all_curves.shape[0]})",
        )

        ax.set_xlabel("Interpolation t")
        ax.set_ylabel("Edit Distance from t=0")
        ax.set_title("Interpolation Continuity (edit distance from origin)")
        ax.legend(loc="upper left")
        fig.tight_layout()
        return fig

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate(self, data: dict) -> list[Figure]:
        """Generate smoothness analysis figures.

        Args:
            data: Shared data dict with ``z``, ``token_ids_list``, ``vocab_size``,
                  ``modules``, ``device``, ``cfg``, ``rope_freqs``,
                  ``cached_mask``, ``full_vocab``, ``bos_id``, ``eos_id``.

        Returns:
            Three figures: [Lipschitz curve, Jaccard scatter,
            interpolation continuity].
        """
        apply_style()

        # --- Pairwise analysis ---
        logger.info("Smoothness: computing pairwise distances (500 pairs)...")
        z_dists, edit_dists, jaccard_dists = self._pairwise_distances(data, n_pairs=500)
        logger.info(
            "Pairwise stats — z dist: mean=%.3f, edit dist: mean=%.1f, "
            "jaccard: mean=%.3f",
            z_dists.mean(), edit_dists.mean(), jaccard_dists.mean(),
        )

        # --- Interpolation continuity ---
        logger.info("Smoothness: computing interpolation continuity (50 pairs)...")
        ts, all_curves = self._interpolation_continuity(
            data, n_interp_pairs=50, n_steps=11,
        )

        # --- Figures ---
        fig1 = self._fig_lipschitz(z_dists, edit_dists)
        fig2 = self._fig_jaccard(z_dists, jaccard_dists)
        fig3 = self._fig_interpolation_continuity(ts, all_curves)

        return [fig1, fig2, fig3]

    def save(
        self, figures: list[Figure], suffixes: list[str] | None = None,
    ) -> list:
        """Save with descriptive suffixes."""
        return super().save(
            figures, suffixes=["lipschitz", "jaccard", "interpolation_continuity"],
        )

"""Surface diversity analysis — unique decoded forms per z vector.

Measures whether different z vectors produce different token sequences,
which is the prerequisite for discriminative surface forms in the
expression game and learnable structure for LLM translation.

Figures:
1. **Diversity ratio** — unique sequences / total, with z-distance
   correlation scatter.
2. **Pairwise edit distance vs z distance** — token-level Lipschitz
   at the surface (not hidden-state) level.
3. **Per-position token entropy** — how much each output position
   varies across different z inputs.
"""

from __future__ import annotations

import logging

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.figure import Figure
from scipy.stats import spearmanr

from lfm.visualize import BaseVisualization
from lfm.visualize.loader import decode_z
from lfm.visualize.style import FIGSIZE_SINGLE, FIGSIZE_WIDE, apply_style

logger = logging.getLogger(__name__)


def _edit_distance(a: list[int], b: list[int]) -> int:
    """Space-optimised Levenshtein distance."""
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


class SurfaceDiversityVisualization(BaseVisualization):
    """Analyze surface-level diversity of decoded output."""

    @property
    def name(self) -> str:
        return "surface_diversity"

    def generate(self, data: dict) -> list[Figure]:
        cfg = self.config
        device = data["device"]

        # Sample z vectors from the pretrained distribution
        n = min(cfg.max_samples, 2000)
        z_mean = data.get("z_mean")
        z_std = data.get("z_std")

        if z_mean is not None:
            z_mean = z_mean.to(device).float()
            z_std = z_std.to(device).float()
            z = torch.randn(n, z_mean.shape[0], device=device) * z_std + z_mean
        else:
            latent_dim = data["cfg"].latent_dim
            z = torch.randn(n, latent_dim, device=device)

        logger.info("Decoding %d random z vectors for surface diversity...", n)
        token_lists = decode_z(z, data, cfg)

        # Compute metrics
        unique_seqs = set(tuple(t) for t in token_lists)
        n_unique = len(unique_seqs)
        diversity_ratio = n_unique / n

        logger.info(
            "Surface diversity: %d/%d unique (%.1f%%)",
            n_unique, n, diversity_ratio * 100,
        )

        # Pairwise z distance vs edit distance (subsample)
        import random
        rng = random.Random(cfg.seed)
        n_pairs = min(1000, n * (n - 1) // 2)
        z_np = z.cpu().numpy()

        z_dists = []
        edit_dists = []
        for _ in range(n_pairs):
            i, j = rng.sample(range(n), 2)
            zd = np.linalg.norm(z_np[i] - z_np[j])
            ed = _edit_distance(token_lists[i], token_lists[j])
            z_dists.append(zd)
            edit_dists.append(ed)

        z_dists = np.array(z_dists)
        edit_dists = np.array(edit_dists)

        # Per-position token entropy
        max_len = max(len(t) for t in token_lists) if token_lists else 0
        max_len = min(max_len, cfg.max_samples)
        vocab_size = data.get("full_vocab", data["cfg"].spm_vocab_size + 2)

        position_entropy = []
        for pos in range(min(max_len, 100)):
            counts: dict[int, int] = {}
            total = 0
            for toks in token_lists:
                if pos < len(toks):
                    t = toks[pos]
                    counts[t] = counts.get(t, 0) + 1
                    total += 1
            if total > 0:
                probs = np.array(list(counts.values())) / total
                ent = -np.sum(probs * np.log2(probs + 1e-10))
                position_entropy.append(ent)
            else:
                position_entropy.append(0.0)

        # Generate figures
        apply_style()
        figs = [
            self._fig_diversity_summary(n_unique, n, diversity_ratio, position_entropy),
            self._fig_edit_vs_z_distance(z_dists, edit_dists),
            self._fig_position_entropy(position_entropy),
        ]
        return figs

    def _fig_diversity_summary(
        self, n_unique: int, n_total: int, ratio: float,
        position_entropy: list[float],
    ) -> Figure:
        fig, axes = plt.subplots(1, 2, figsize=FIGSIZE_WIDE)

        # Bar: unique vs total
        ax = axes[0]
        ax.bar(["Unique", "Total"], [n_unique, n_total], color=["#2ecc71", "#e74c3c"])
        ax.set_ylabel("Count")
        ax.set_title(f"Surface Diversity: {ratio:.1%}")
        for i, v in enumerate([n_unique, n_total]):
            ax.text(i, v + n_total * 0.02, str(v), ha="center", fontweight="bold")

        # Mean entropy across positions
        ax = axes[1]
        if position_entropy:
            mean_ent = np.mean(position_entropy)
            ax.axhline(mean_ent, color="#e74c3c", ls="--", label=f"Mean={mean_ent:.2f}")
            ax.bar(range(len(position_entropy)), position_entropy, color="#3498db", alpha=0.7)
            ax.set_xlabel("Output position")
            ax.set_ylabel("Token entropy (bits)")
            ax.set_title("Per-Position Token Entropy")
            ax.legend()

        fig.suptitle("Surface Diversity Analysis", fontsize=14, fontweight="bold")
        fig.tight_layout()
        return fig

    def _fig_edit_vs_z_distance(
        self, z_dists: np.ndarray, edit_dists: np.ndarray,
    ) -> Figure:
        fig, ax = plt.subplots(figsize=FIGSIZE_SINGLE)

        ax.scatter(z_dists, edit_dists, alpha=0.15, s=8, color="#3498db")

        if len(z_dists) > 10:
            r, p = spearmanr(z_dists, edit_dists)
            ax.set_title(f"Surface Edit Distance vs z Distance (Spearman r={r:.3f}, p={p:.1e})")
        else:
            ax.set_title("Surface Edit Distance vs z Distance")

        ax.set_xlabel("z L2 distance")
        ax.set_ylabel("Token edit distance")
        fig.tight_layout()
        return fig

    def _fig_position_entropy(self, position_entropy: list[float]) -> Figure:
        fig, ax = plt.subplots(figsize=FIGSIZE_WIDE)

        if position_entropy:
            colors = plt.cm.viridis(np.linspace(0, 1, len(position_entropy)))
            ax.bar(range(len(position_entropy)), position_entropy, color=colors, alpha=0.8)
            ax.set_xlabel("Output position")
            ax.set_ylabel("Token entropy (bits)")
            ax.set_title("Per-Position Token Entropy (higher = more diverse)")
            mean_ent = np.mean(position_entropy)
            ax.axhline(mean_ent, color="#e74c3c", ls="--", lw=2, label=f"Mean={mean_ent:.2f} bits")
            ax.legend()

        fig.tight_layout()
        return fig

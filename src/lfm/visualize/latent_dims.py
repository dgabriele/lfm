"""Per-dimension analysis of the 384-dim VAE latent space.

Generates four figures:
1. Per-dimension variance (sorted bar chart) — effective dimensionality.
2. Per-dimension mean by language (heatmap) — language-specific activation.
3. Cumulative PCA explained variance — intrinsic dimensionality.
4. F-statistic per dimension (ANOVA) — language discriminativeness.
"""

from __future__ import annotations

import logging
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from matplotlib.figure import Figure
from scipy.stats import f_oneway
from sklearn.decomposition import PCA

from lfm.visualize import BaseVisualization
from lfm.visualize.config import VisualizeConfig
from lfm.visualize.languages import LANGUAGES
from lfm.visualize.style import FIGSIZE_SINGLE, FIGSIZE_WIDE, LANG_COLORS, apply_style

logger = logging.getLogger(__name__)


class LatentDimsVisualization(BaseVisualization):
    """Per-dimension properties of the VAE latent space."""

    name = "latent_dims"

    def __init__(self, config: VisualizeConfig) -> None:
        super().__init__(config)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate(self, data: dict) -> list[Figure]:
        """Generate four latent-dimension analysis figures.

        Args:
            data: Dict with ``z`` (N, 384) tensor and ``languages`` list[str].

        Returns:
            Four figures: [variance, mean_heatmap, pca, f_statistic].
        """
        apply_style()

        z_tensor = data["z"]
        languages: list[str] = data["languages"]

        z = z_tensor.cpu().numpy() if isinstance(z_tensor, torch.Tensor) else np.asarray(z_tensor)

        fig_variance = self._make_variance_plot(z)
        fig_heatmap = self._make_mean_heatmap(z, languages)
        fig_pca = self._make_pca_plot(z)
        fig_fstat = self._make_fstatistic_plot(z, languages)

        return [fig_variance, fig_heatmap, fig_pca, fig_fstat]

    def save(self, figures: list[Figure], suffixes: list[str] | None = None) -> list:
        """Save with descriptive suffixes."""
        return super().save(
            figures,
            suffixes=["variance", "mean_heatmap", "pca", "f_statistic"],
        )

    # ------------------------------------------------------------------
    # 1. Per-dimension variance
    # ------------------------------------------------------------------

    @staticmethod
    def _make_variance_plot(z: np.ndarray) -> Figure:
        """Bar chart of per-dimension variance, sorted descending."""
        variances = np.var(z, axis=0)
        sorted_idx = np.argsort(variances)[::-1]
        sorted_var = variances[sorted_idx]

        fig, ax = plt.subplots(figsize=FIGSIZE_WIDE)
        ax.bar(range(len(sorted_var)), sorted_var, width=1.0, color="#1f77b4", edgecolor="none")
        ax.set_xlabel("Dimension (sorted by variance)")
        ax.set_ylabel("Variance")
        ax.set_title("Per-Dimension Variance of Latent Space")
        ax.set_xlim(-0.5, len(sorted_var) - 0.5)
        fig.tight_layout()
        return fig

    # ------------------------------------------------------------------
    # 2. Per-dimension mean by language (heatmap)
    # ------------------------------------------------------------------

    @staticmethod
    def _make_mean_heatmap(z: np.ndarray, languages: list[str]) -> Figure:
        """Heatmap of per-language, per-dimension means."""
        # Group vectors by language
        buckets: dict[str, list[np.ndarray]] = defaultdict(list)
        for vec, lang in zip(z, languages):
            buckets[lang].append(vec)

        codes = sorted(buckets.keys())
        labels = [LANGUAGES[c].name if c in LANGUAGES else c for c in codes]
        means = np.stack([np.mean(buckets[c], axis=0) for c in codes])

        fig, ax = plt.subplots(figsize=(max(14, means.shape[1] // 20), max(6, len(codes) * 0.5)))
        sns.heatmap(
            means,
            xticklabels=False,
            yticklabels=labels,
            center=0,
            cmap="RdBu_r",
            cbar_kws={"label": "Mean Activation", "shrink": 0.8},
            ax=ax,
            rasterized=True,
        )

        ax.set_xlabel("Latent Dimension")
        ax.set_ylabel("Language")
        ax.set_title("Per-Dimension Mean Activation by Language")
        ax.tick_params(axis="y", rotation=0)
        fig.tight_layout()
        return fig

    # ------------------------------------------------------------------
    # 3. Cumulative PCA explained variance
    # ------------------------------------------------------------------

    @staticmethod
    def _make_pca_plot(z: np.ndarray) -> Figure:
        """Cumulative explained variance from PCA with threshold annotations."""
        n_components = min(z.shape[0], z.shape[1])
        pca = PCA(n_components=n_components)
        pca.fit(z)

        cumulative = np.cumsum(pca.explained_variance_ratio_)
        n_dims = len(cumulative)

        fig, ax = plt.subplots(figsize=FIGSIZE_SINGLE)
        ax.plot(range(1, n_dims + 1), cumulative, color="#1f77b4", linewidth=2)
        ax.set_xlabel("Number of Principal Components")
        ax.set_ylabel("Cumulative Explained Variance")
        ax.set_title("PCA Cumulative Explained Variance")
        ax.set_xlim(1, n_dims)
        ax.set_ylim(0, 1.05)

        # Threshold annotations
        thresholds = [0.90, 0.95, 0.99]
        colors = ["#2ca02c", "#ff7f0e", "#d62728"]
        for thresh, color in zip(thresholds, colors):
            idx = np.searchsorted(cumulative, thresh)
            if idx < n_dims:
                n_pcs = idx + 1
                ax.axhline(y=thresh, linestyle="--", color=color, alpha=0.7, linewidth=1)
                ax.annotate(
                    f"{thresh:.0%}: {n_pcs} PCs",
                    xy=(n_pcs, thresh),
                    xytext=(n_pcs + n_dims * 0.05, thresh - 0.03),
                    fontsize=10,
                    color=color,
                    fontweight="bold",
                    arrowprops=dict(arrowstyle="->", color=color, lw=1.2),
                )
                ax.plot(n_pcs, thresh, "o", color=color, markersize=6, zorder=5)

        fig.tight_layout()
        return fig

    # ------------------------------------------------------------------
    # 4. F-statistic per dimension (ANOVA)
    # ------------------------------------------------------------------

    @staticmethod
    def _make_fstatistic_plot(z: np.ndarray, languages: list[str]) -> Figure:
        """Bar chart of per-dimension ANOVA F-statistic across languages."""
        # Group row indices by language
        lang_indices: dict[str, list[int]] = defaultdict(list)
        for i, lang in enumerate(languages):
            lang_indices[lang].append(i)

        # Need at least 2 languages for ANOVA
        groups_by_lang = {
            lang: idx for lang, idx in lang_indices.items() if len(idx) >= 2
        }
        if len(groups_by_lang) < 2:
            logger.warning(
                "Fewer than 2 languages with >=2 samples; cannot compute ANOVA"
            )
            fig, ax = plt.subplots(figsize=FIGSIZE_WIDE)
            ax.set_title("Language Discriminativeness per Latent Dimension")
            ax.text(0.5, 0.5, "Insufficient data", transform=ax.transAxes, ha="center")
            return fig

        n_dims = z.shape[1]
        f_stats = np.zeros(n_dims)
        lang_keys = sorted(groups_by_lang.keys())

        for d in range(n_dims):
            groups = [z[groups_by_lang[lang], d] for lang in lang_keys]
            f_val, _ = f_oneway(*groups)
            f_stats[d] = f_val if np.isfinite(f_val) else 0.0

        sorted_idx = np.argsort(f_stats)[::-1]
        sorted_f = f_stats[sorted_idx]

        fig, ax = plt.subplots(figsize=FIGSIZE_WIDE)
        ax.bar(range(len(sorted_f)), sorted_f, width=1.0, color="#d62728", edgecolor="none")
        ax.set_xlabel("Dimension (sorted by F-statistic)")
        ax.set_ylabel("F-statistic")
        ax.set_title("Language Discriminativeness per Latent Dimension")
        ax.set_xlim(-0.5, len(sorted_f) - 0.5)
        fig.tight_layout()
        return fig

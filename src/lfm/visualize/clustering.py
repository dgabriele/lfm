"""Hierarchical clustering dendrograms and distance heatmaps of latent vectors.

Computes per-language mean latent vectors, then generates:
1. A dendrogram colored by language family.
2. A distance heatmap with hierarchical ordering from the dendrogram.
"""

from __future__ import annotations

import logging
import warnings
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from matplotlib.figure import Figure
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import pdist, squareform

from lfm.visualize import BaseVisualization
from lfm.visualize.config import VisualizeConfig
from lfm.visualize.languages import FAMILIES, LANGUAGES
from lfm.visualize.style import FAMILY_COLORS, FIGSIZE_SINGLE, FIGSIZE_WIDE, apply_style

logger = logging.getLogger(__name__)


class ClusteringVisualization(BaseVisualization):
    """Dendrogram and distance heatmap of per-language mean latent vectors."""

    name = "clustering"

    def __init__(self, config: VisualizeConfig) -> None:
        super().__init__(config)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate(self, data: dict) -> list[Figure]:
        """Generate clustering figures from encoded corpus data.

        Args:
            data: Dict with ``z`` (N, 384) tensor and ``languages`` list[str].

        Returns:
            Two figures: [dendrogram, heatmap].
        """
        apply_style()

        z = data["z"]
        languages = data["languages"]

        # Compute per-language mean vectors
        codes, means = self._per_language_means(z, languages)
        labels = [LANGUAGES[c].name if c in LANGUAGES else c for c in codes]

        # Resolve metric and linkage, handling the ward + cosine incompatibility
        metric, linkage_method = self._resolve_metric_linkage()

        # Pairwise distance matrix
        dist_vec = self._pairwise_distances(means, metric)

        # Hierarchical linkage
        Z = linkage(dist_vec, method=linkage_method)

        # Build figures
        fig_dendro = self._make_dendrogram(Z, codes, labels)
        fig_heatmap = self._make_heatmap(Z, dist_vec, codes, labels, metric)

        return [fig_dendro, fig_heatmap]

    def save(self, figures: list[Figure], suffixes: list[str] | None = None) -> list:
        """Save with descriptive suffixes."""
        return super().save(figures, suffixes=["dendrogram", "heatmap"])

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    @staticmethod
    def _per_language_means(
        z: torch.Tensor, languages: list[str]
    ) -> tuple[list[str], np.ndarray]:
        """Group z vectors by language code and compute per-language means.

        Returns:
            Tuple of (sorted language codes, mean vectors as (L, D) ndarray).
        """
        z_np = z.cpu().numpy() if isinstance(z, torch.Tensor) else np.asarray(z)

        buckets: dict[str, list[np.ndarray]] = defaultdict(list)
        for vec, lang in zip(z_np, languages):
            buckets[lang].append(vec)

        codes = sorted(buckets.keys())
        means = np.stack([np.mean(buckets[c], axis=0) for c in codes])
        return codes, means

    def _resolve_metric_linkage(self) -> tuple[str, str]:
        """Return (distance_metric, linkage_method), auto-correcting conflicts.

        Ward linkage requires Euclidean distance. If the user requests ward
        with cosine, we switch to L2 and emit a warning.
        """
        metric = self.config.metric
        linkage_method = self.config.linkage

        if linkage_method == "ward" and metric == "cosine":
            warnings.warn(
                "Ward linkage requires Euclidean distance; switching linkage "
                "from 'ward' to 'average'.",
                UserWarning,
                stacklevel=2,
            )
            linkage_method = "average"

        # Normalize metric name: config uses "l2" as alias for euclidean
        if metric == "l2":
            metric = "euclidean"

        return metric, linkage_method

    @staticmethod
    def _pairwise_distances(means: np.ndarray, metric: str) -> np.ndarray:
        """Compute condensed pairwise distance vector."""
        return pdist(means, metric=metric)

    # ------------------------------------------------------------------
    # Dendrogram
    # ------------------------------------------------------------------

    def _make_dendrogram(
        self,
        Z: np.ndarray,
        codes: list[str],
        labels: list[str],
    ) -> Figure:
        """Create a dendrogram figure with leaves colored by language family."""
        fig, ax = plt.subplots(figsize=FIGSIZE_WIDE)

        # scipy dendrogram — we disable its default coloring so we can do our own
        dendro = dendrogram(
            Z,
            labels=labels,
            ax=ax,
            leaf_rotation=45,
            leaf_font_size=10,
            above_threshold_color="#999999",
        )

        # Color the x-axis tick labels by family
        leaf_order = dendro["leaves"]
        tick_labels = ax.get_xticklabels()
        for tick, leaf_idx in zip(tick_labels, leaf_order):
            code = codes[leaf_idx]
            family = LANGUAGES[code].family if code in LANGUAGES else None
            color = FAMILY_COLORS.get(family, "#333333")
            tick.set_color(color)
            tick.set_fontweight("bold")

        # Add a family-color legend
        families_present = {
            LANGUAGES[c].family for c in codes if c in LANGUAGES
        }
        handles = [
            plt.Line2D(
                [0], [0],
                marker="s",
                color="w",
                markerfacecolor=FAMILY_COLORS.get(f, "#333"),
                markersize=8,
                label=f,
            )
            for f in sorted(families_present)
        ]
        ax.legend(
            handles=handles,
            loc="upper right",
            title="Language Family",
            frameon=True,
            framealpha=0.9,
        )

        ax.set_title("Hierarchical Clustering of Per-Language Mean Latent Vectors")
        ax.set_ylabel("Distance")
        ax.grid(False)
        fig.tight_layout()
        return fig

    # ------------------------------------------------------------------
    # Heatmap
    # ------------------------------------------------------------------

    def _make_heatmap(
        self,
        Z: np.ndarray,
        dist_vec: np.ndarray,
        codes: list[str],
        labels: list[str],
        metric: str,
    ) -> Figure:
        """Create a distance heatmap with hierarchical ordering."""
        # Get leaf order from the dendrogram (suppress plotting)
        dendro = dendrogram(Z, no_plot=True)
        order = dendro["leaves"]

        # Reorder the square distance matrix
        dist_sq = squareform(dist_vec)
        dist_ordered = dist_sq[np.ix_(order, order)]
        labels_ordered = [labels[i] for i in order]
        codes_ordered = [codes[i] for i in order]

        fig, ax = plt.subplots(figsize=(12, 10))

        sns.heatmap(
            dist_ordered,
            xticklabels=labels_ordered,
            yticklabels=labels_ordered,
            annot=True,
            fmt=".2f",
            annot_kws={"size": 7},
            cmap="YlOrRd",
            square=True,
            linewidths=0.5,
            linecolor="white",
            cbar_kws={"label": f"{metric.capitalize()} Distance", "shrink": 0.8},
            ax=ax,
        )

        # Color tick labels by family
        for tick in ax.get_xticklabels():
            name = tick.get_text()
            code = self._name_to_code(name, codes_ordered, labels_ordered)
            family = LANGUAGES[code].family if code and code in LANGUAGES else None
            tick.set_color(FAMILY_COLORS.get(family, "#333333"))
            tick.set_fontweight("bold")

        for tick in ax.get_yticklabels():
            name = tick.get_text()
            code = self._name_to_code(name, codes_ordered, labels_ordered)
            family = LANGUAGES[code].family if code and code in LANGUAGES else None
            tick.set_color(FAMILY_COLORS.get(family, "#333333"))
            tick.set_fontweight("bold")

        ax.set_title("Pairwise Distance Matrix (Hierarchical Order)")
        ax.tick_params(axis="x", rotation=45)
        ax.tick_params(axis="y", rotation=0)
        fig.tight_layout()
        return fig

    @staticmethod
    def _name_to_code(
        name: str, codes: list[str], labels: list[str]
    ) -> str | None:
        """Reverse-lookup a language code from its display name."""
        for code, label in zip(codes, labels):
            if label == name:
                return code
        return None

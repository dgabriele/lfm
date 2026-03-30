"""t-SNE / UMAP 2D projection of the VAE latent space.

Generates scatter plots of latent vectors colored by individual language,
language family, and morphological type.
"""

from __future__ import annotations

import logging
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure
from sklearn.manifold import TSNE

from lfm.visualize import BaseVisualization
from lfm.visualize.config import VisualizeConfig
from lfm.visualize.languages import LANGUAGES, get_label
from lfm.visualize.style import (
    FIGSIZE_SINGLE,
    SCATTER_ALPHA,
    SCATTER_SIZE,
    apply_style,
    get_color_map,
    language_legend,
)

logger = logging.getLogger(__name__)

# The three groupings we produce, with descriptive titles and legend keys
_VIEWS: list[tuple[str, str]] = [
    ("language", "Latent Space by Language"),
    ("family", "Latent Space by Language Family"),
    ("type", "Latent Space by Morphological Type"),
]


class TSNEVisualization(BaseVisualization):
    """2D scatter of VAE latent codes via t-SNE or UMAP."""

    @property
    def name(self) -> str:
        return "tsne"

    # ------------------------------------------------------------------
    # Dimensionality reduction
    # ------------------------------------------------------------------

    def _reduce(self, z: np.ndarray) -> np.ndarray:
        """Reduce *z* from (N, D) to (N, 2).

        Uses UMAP when ``self.config.method == "umap"`` and the
        ``umap-learn`` package is available; otherwise falls back to
        scikit-learn t-SNE.
        """
        if self.config.method == "umap":
            try:
                import umap  # umap-learn

                logger.info(
                    "Running UMAP (n=%d, metric=%s) ...",
                    z.shape[0],
                    self.config.metric,
                )
                reducer = umap.UMAP(
                    n_components=2,
                    metric=self.config.metric,
                    random_state=self.config.seed,
                )
                return reducer.fit_transform(z)
            except ImportError:
                logger.warning(
                    "umap-learn not installed; falling back to t-SNE"
                )

        logger.info(
            "Running t-SNE (n=%d, perplexity=%d) ...",
            z.shape[0],
            self.config.perplexity,
        )
        tsne = TSNE(
            n_components=2,
            perplexity=self.config.perplexity,
            metric=self.config.metric,
            random_state=self.config.seed,
            init="pca",
            learning_rate="auto",
        )
        return tsne.fit_transform(z)

    # ------------------------------------------------------------------
    # Plotting
    # ------------------------------------------------------------------

    @staticmethod
    def _scatter_by(
        ax: plt.Axes,
        xy: np.ndarray,
        languages: list[str],
        by: str,
    ) -> None:
        """Draw one scatter plot, coloring points by *by* grouping."""
        # Ensure colors exist for all languages/families/types in the data
        unique_codes = sorted(set(languages))
        unique_labels = sorted({get_label(c, by) for c in unique_codes})
        if by == "language":
            cmap = get_color_map(by, keys=unique_codes)
        else:
            cmap = get_color_map(by, keys=unique_labels)

        # Group indices by label so each group gets a single scatter call
        # (cleaner legend, consistent z-order)
        label_to_idx: dict[str, list[int]] = {}
        for i, code in enumerate(languages):
            label = get_label(code, by)
            label_to_idx.setdefault(label, []).append(i)

        for label in sorted(label_to_idx):
            idx = np.array(label_to_idx[label])
            # Resolve color: for "language" grouping the key is the code,
            # for family/type the key is the label itself.
            if by == "language":
                # All codes in this group share the same label (language name),
                # but the color map is keyed by code.
                code = languages[idx[0]]
                color = cmap.get(code, "#333333")
            else:
                color = cmap.get(label, "#333333")

            ax.scatter(
                xy[idx, 0],
                xy[idx, 1],
                s=SCATTER_SIZE,
                alpha=SCATTER_ALPHA,
                c=color,
                label=label,
                edgecolors="none",
                rasterized=True,
            )

    def _make_figure(
        self,
        xy: np.ndarray,
        languages: list[str],
        by: str,
        title: str,
    ) -> Figure:
        """Create a single scatter figure for the given grouping."""
        fig, ax = plt.subplots(figsize=FIGSIZE_SINGLE)
        self._scatter_by(ax, xy, languages, by)
        ax.set_title(title)
        ax.set_xlabel("Dimension 1")
        ax.set_ylabel("Dimension 2")
        language_legend(ax, by=by)
        fig.tight_layout()
        return fig

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate(self, data: dict[str, Any]) -> list[Figure]:
        """Generate three t-SNE/UMAP scatter figures.

        Args:
            data: Must contain ``"z"`` (Tensor of shape (N, 384)) and
                  ``"languages"`` (list[str] of ISO 639-3 codes).

        Returns:
            List of three :class:`~matplotlib.figure.Figure` objects
            (by language, by family, by type).
        """
        apply_style()

        z_tensor = data["z"]
        languages: list[str] = data["languages"]

        z = np.array(z_tensor.numpy())
        xy = self._reduce(z)

        figures: list[Figure] = []
        for by, title in _VIEWS:
            method_label = (
                "UMAP" if self.config.method == "umap" else "t-SNE"
            )
            full_title = f"{title} ({method_label})"
            fig = self._make_figure(xy, languages, by, full_title)
            figures.append(fig)

        return figures

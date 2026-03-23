"""Output length distribution analysis for the VAE decoder.

Generates three figures:
1. Histogram comparing decoded vs. training corpus sequence lengths.
2. Box plot of sequence lengths grouped by source language.
3. Scatter of decoded length vs. latent norm with linear trend.
"""

from __future__ import annotations

import logging

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.figure import Figure

from lfm.visualize import BaseVisualization
from lfm.visualize.config import VisualizeConfig
from lfm.visualize.languages import LANGUAGES
from lfm.visualize.loader import decode_z
from lfm.visualize.style import (
    FIGSIZE_SINGLE,
    FIGSIZE_WIDE,
    LANG_COLORS,
    apply_style,
)

logger = logging.getLogger(__name__)

# Maximum z vectors to decode (keeps runtime reasonable)
_MAX_DECODE = 2000


class LengthDistVisualization(BaseVisualization):
    """Analyze output length distributions from the VAE decoder."""

    @property
    def name(self) -> str:
        return "length_dist"

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _measure_lengths(token_lists: list[list[int]]) -> np.ndarray:
        """Return an array of sequence lengths."""
        return np.array([len(seq) for seq in token_lists])

    # ------------------------------------------------------------------
    # Figures
    # ------------------------------------------------------------------

    def _fig_histogram(
        self,
        decoded_lengths: np.ndarray,
        corpus_lengths: np.ndarray,
    ) -> Figure:
        """Histogram of decoded vs. training corpus sequence lengths."""
        fig, ax = plt.subplots(figsize=FIGSIZE_WIDE)

        bins = np.linspace(
            0,
            max(decoded_lengths.max(), corpus_lengths.max()) + 1,
            50,
        )

        ax.hist(
            corpus_lengths,
            bins=bins,
            alpha=0.5,
            label="Training corpus",
            color="#1f77b4",
            edgecolor="white",
            linewidth=0.5,
        )
        ax.hist(
            decoded_lengths,
            bins=bins,
            alpha=0.5,
            label="Decoded",
            color="#ff7f0e",
            edgecolor="white",
            linewidth=0.5,
        )

        ax.set_xlabel("Sequence length (tokens)")
        ax.set_ylabel("Count")
        ax.set_title("Output Length Distribution")
        ax.legend()
        fig.tight_layout()
        return fig

    def _fig_boxplot_by_language(
        self,
        token_ids_list: list[list[int]],
        languages: list[str] | None,
    ) -> Figure:
        """Box plot of corpus sequence lengths grouped by language."""
        fig, ax = plt.subplots(figsize=FIGSIZE_WIDE)

        if languages is None:
            ax.text(
                0.5,
                0.5,
                "Language labels not available",
                transform=ax.transAxes,
                ha="center",
                va="center",
                fontsize=14,
                color="gray",
            )
            ax.set_title("Sequence Length by Language")
            fig.tight_layout()
            return fig

        # Group lengths by language code
        lang_lengths: dict[str, list[int]] = {}
        for seq, lang in zip(token_ids_list, languages):
            lang_lengths.setdefault(lang, []).append(len(seq))

        # Sort by language code for consistent ordering
        codes = sorted(lang_lengths.keys())
        data = [lang_lengths[c] for c in codes]
        labels = [
            LANGUAGES[c].name if c in LANGUAGES else c for c in codes
        ]
        colors = [LANG_COLORS.get(c, "#333333") for c in codes]

        bp = ax.boxplot(
            data,
            labels=labels,
            patch_artist=True,
            showfliers=False,
            medianprops={"color": "black", "linewidth": 1.5},
        )

        for patch, color in zip(bp["boxes"], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)

        ax.set_xlabel("Language")
        ax.set_ylabel("Sequence length (tokens)")
        ax.set_title("Sequence Length by Language")
        ax.tick_params(axis="x", rotation=45)
        fig.tight_layout()
        return fig

    def _fig_length_vs_norm(
        self,
        z_subset: torch.Tensor,
        decoded_lengths: np.ndarray,
    ) -> Figure:
        """Scatter of decoded length vs. latent L2 norm with trend line."""
        fig, ax = plt.subplots(figsize=FIGSIZE_SINGLE)

        z_np = z_subset.cpu().numpy() if isinstance(z_subset, torch.Tensor) else np.asarray(z_subset)
        norms = np.linalg.norm(z_np, axis=1)

        ax.scatter(
            norms,
            decoded_lengths,
            alpha=0.3,
            s=10,
            color="#1f77b4",
            edgecolors="none",
            rasterized=True,
        )

        # Linear trend line
        if len(norms) > 1:
            coeffs = np.polyfit(norms, decoded_lengths, 1)
            x_line = np.linspace(norms.min(), norms.max(), 100)
            y_line = np.polyval(coeffs, x_line)
            ax.plot(
                x_line,
                y_line,
                color="#d62728",
                linewidth=2,
                label=f"Trend (slope={coeffs[0]:.2f})",
            )
            ax.legend()

        ax.set_xlabel("L2 norm of z")
        ax.set_ylabel("Decoded sequence length (tokens)")
        ax.set_title("Decoded Length vs Latent Norm")
        fig.tight_layout()
        return fig

    # ------------------------------------------------------------------
    # Main entry
    # ------------------------------------------------------------------

    def generate(self, data: dict) -> list[Figure]:
        """Generate length distribution figures.

        Args:
            data: Shared data dict containing z vectors, token_ids_list,
                  decoder modules, config, and optionally languages.

        Returns:
            Three figures: [histogram, boxplot_by_language, length_vs_norm].
        """
        apply_style()

        z = data["z"]
        token_ids_list = data["token_ids_list"]
        languages = data.get("languages")

        # Decode a subset of z vectors through the decoder
        n = min(_MAX_DECODE, z.size(0))
        z_subset = z[:n]

        logger.info("Decoding %d z vectors for length distribution analysis...", n)
        decoded_tokens = decode_z(z_subset, data, self.config)

        decoded_lengths = self._measure_lengths(decoded_tokens)
        corpus_lengths = self._measure_lengths(token_ids_list)

        logger.info(
            "Decoded lengths — mean: %.1f, std: %.1f | "
            "Corpus lengths — mean: %.1f, std: %.1f",
            decoded_lengths.mean(),
            decoded_lengths.std(),
            corpus_lengths.mean(),
            corpus_lengths.std(),
        )

        # Generate figures
        fig1 = self._fig_histogram(decoded_lengths, corpus_lengths)
        fig2 = self._fig_boxplot_by_language(token_ids_list, languages)
        fig3 = self._fig_length_vs_norm(z_subset, decoded_lengths)

        return [fig1, fig2, fig3]

    def save(self, figures: list[Figure], suffixes: list[str] | None = None) -> list:
        """Save with descriptive suffixes."""
        return super().save(
            figures, suffixes=["histogram", "by_language", "vs_norm"]
        )

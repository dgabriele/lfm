"""Zipf's law analysis for VAE decoded output.

Compares the rank-frequency distribution of training corpus tokens
against tokens produced by decoding random latent vectors, and fits
Zipf exponents to quantify how closely each distribution follows
the ideal 1/rank^s power law.
"""

from __future__ import annotations

import logging
from collections import Counter

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.figure import Figure

from lfm.visualize import BaseVisualization
from lfm.visualize.config import VisualizeConfig
from lfm.visualize.loader import decode_z
from lfm.visualize.style import FIGSIZE_SINGLE, FIGSIZE_WIDE, apply_style

logger = logging.getLogger(__name__)


class ZipfVisualization(BaseVisualization):
    """Rank-frequency (Zipf) analysis of corpus vs. decoded token distributions."""

    @property
    def name(self) -> str:
        return "zipf"

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _rank_freq(counter: Counter) -> tuple[np.ndarray, np.ndarray]:
        """Return (ranks, frequencies) sorted by descending frequency."""
        freqs = np.array(sorted(counter.values(), reverse=True), dtype=np.float64)
        ranks = np.arange(1, len(freqs) + 1, dtype=np.float64)
        return ranks, freqs

    @staticmethod
    def _fit_zipf(ranks: np.ndarray, freqs: np.ndarray, top_k: int = 500) -> float:
        """Fit Zipf exponent *s* via linear regression on log-log data.

        Uses the top-*k* tokens (by frequency) to avoid noisy tail.

        The model is ``log(freq) = -s * log(rank) + c``, so the slope
        of ``numpy.polyfit`` gives ``-s``.
        """
        k = min(top_k, len(ranks))
        log_r = np.log(ranks[:k])
        log_f = np.log(freqs[:k])
        slope, _ = np.polyfit(log_r, log_f, 1)
        return -slope  # s (positive)

    # ------------------------------------------------------------------
    # Token counting
    # ------------------------------------------------------------------

    @staticmethod
    def _count_corpus(token_ids_list: list[list[int]]) -> Counter:
        """Count token frequencies across all corpus sequences."""
        counter: Counter = Counter()
        for ids in token_ids_list:
            counter.update(ids)
        return counter

    def _count_decoded(self, data: dict) -> Counter:
        """Sample random z vectors, decode them, and count token frequencies."""
        n_samples = self.config.n_samples
        latent_dim = data["cfg"].latent_dim
        device = data["device"]

        logger.info("Sampling %d random z vectors for Zipf analysis...", n_samples)
        z = torch.randn(n_samples, latent_dim, device=device)

        decoded_ids = decode_z(z, data, self.config)

        counter: Counter = Counter()
        for ids in decoded_ids:
            counter.update(ids)
        return counter

    # ------------------------------------------------------------------
    # Figures
    # ------------------------------------------------------------------

    def _fig_rank_freq(
        self,
        corpus_ranks: np.ndarray,
        corpus_freqs: np.ndarray,
        decoded_ranks: np.ndarray,
        decoded_freqs: np.ndarray,
    ) -> Figure:
        """Log-log rank-frequency plot with ideal Zipf reference."""
        apply_style()
        fig, ax = plt.subplots(figsize=FIGSIZE_WIDE)

        ax.plot(
            np.log(corpus_ranks),
            np.log(corpus_freqs),
            linewidth=1.5,
            alpha=0.8,
            label="Training corpus",
        )
        ax.plot(
            np.log(decoded_ranks),
            np.log(decoded_freqs),
            linewidth=1.5,
            alpha=0.8,
            label="Decoded output",
        )

        # Ideal Zipf reference: freq ~ 1/rank  (s = 1)
        max_rank = max(len(corpus_ranks), len(decoded_ranks))
        ref_ranks = np.arange(1, max_rank + 1, dtype=np.float64)
        ref_freqs = 1.0 / ref_ranks
        # Scale reference to match the corpus peak for visual comparison
        ref_freqs = ref_freqs * corpus_freqs[0]
        ax.plot(
            np.log(ref_ranks),
            np.log(ref_freqs),
            linestyle="--",
            linewidth=1.0,
            color="gray",
            alpha=0.7,
            label="Ideal Zipf ($s=1$)",
        )

        ax.set_xlabel("log(rank)")
        ax.set_ylabel("log(frequency)")
        ax.set_title("Rank-Frequency Distribution (Zipf Analysis)")
        ax.legend()

        fig.tight_layout()
        return fig

    def _fig_exponent(self, s_corpus: float, s_decoded: float) -> Figure:
        """Bar chart comparing fitted Zipf exponents."""
        apply_style()
        fig, ax = plt.subplots(figsize=FIGSIZE_SINGLE)

        labels = ["Corpus", "Decoded"]
        values = [s_corpus, s_decoded]
        colors = ["#1f77b4", "#ff7f0e"]

        bars = ax.bar(labels, values, color=colors, width=0.5)

        # Annotate bars with the fitted exponent values
        for bar, val in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.02,
                f"$s = {val:.3f}$",
                ha="center",
                va="bottom",
                fontweight="bold",
            )

        # Ideal Zipf reference line
        ax.axhline(y=1.0, linestyle="--", color="gray", linewidth=1.0, label="Ideal Zipf ($s=1$)")

        ax.set_ylabel("Zipf exponent $s$")
        ax.set_title("Fitted Zipf Exponent")
        ax.legend()
        ax.set_ylim(0, max(s_corpus, s_decoded, 1.0) * 1.4)

        fig.tight_layout()
        return fig

    # ------------------------------------------------------------------
    # Main entry
    # ------------------------------------------------------------------

    def generate(self, data: dict) -> list[Figure]:
        """Generate Zipf analysis figures.

        Args:
            data: Shared data dict containing z vectors, token_ids_list,
                  decoder modules, and config.

        Returns:
            Two figures: rank-frequency plot and exponent comparison.
        """
        # 1. Count token frequencies in the training corpus
        corpus_counter = self._count_corpus(data["token_ids_list"])
        corpus_ranks, corpus_freqs = self._rank_freq(corpus_counter)

        # 2. Decode random z vectors and count token frequencies
        decoded_counter = self._count_decoded(data)
        decoded_ranks, decoded_freqs = self._rank_freq(decoded_counter)

        # 3. Fit Zipf exponents
        s_corpus = self._fit_zipf(corpus_ranks, corpus_freqs, top_k=500)
        s_decoded = self._fit_zipf(decoded_ranks, decoded_freqs, top_k=500)
        logger.info("Zipf exponents — corpus: %.3f, decoded: %.3f", s_corpus, s_decoded)

        # 4. Generate figures
        fig1 = self._fig_rank_freq(corpus_ranks, corpus_freqs, decoded_ranks, decoded_freqs)
        fig2 = self._fig_exponent(s_corpus, s_decoded)

        return [fig1, fig2]

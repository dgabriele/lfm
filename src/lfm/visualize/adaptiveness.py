"""Decoder adaptiveness analysis for the VAE.

Measures whether the decoder adapts output length and complexity to match
input complexity.  Three figures:

1. Input Length vs Output Length — scatter with regression line.
2. z Norm vs Output Diversity — latent richness → token diversity.
3. Complexity Adaptation Profile — box plots across input complexity quintiles.
"""

from __future__ import annotations

import logging

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.figure import Figure
from scipy.stats import pearsonr

from lfm.visualize import BaseVisualization
from lfm.visualize.config import VisualizeConfig
from lfm.visualize.loader import decode_z
from lfm.visualize.style import FIGSIZE_SINGLE, FIGSIZE_WIDE, apply_style

logger = logging.getLogger(__name__)

# Maximum z vectors to decode (keeps runtime reasonable)
_MAX_DECODE = 2000


class AdaptivenessVisualization(BaseVisualization):
    """Analyze whether the decoder adapts output to input complexity."""

    @property
    def name(self) -> str:
        return "adaptiveness"

    # ------------------------------------------------------------------
    # Complexity metrics
    # ------------------------------------------------------------------

    @staticmethod
    def _input_lengths(token_ids_list: list[list[int]]) -> np.ndarray:
        """Number of tokens in each input sequence."""
        return np.array([len(seq) for seq in token_ids_list])

    @staticmethod
    def _type_token_ratio(token_ids_list: list[list[int]]) -> np.ndarray:
        """Unique tokens / total tokens for each sequence."""
        ratios = []
        for seq in token_ids_list:
            if len(seq) == 0:
                ratios.append(0.0)
            else:
                ratios.append(len(set(seq)) / len(seq))
        return np.array(ratios)

    @staticmethod
    def _z_norms(z: torch.Tensor) -> np.ndarray:
        """L2 norm of each z vector."""
        z_np = z.cpu().numpy() if isinstance(z, torch.Tensor) else np.asarray(z)
        return np.linalg.norm(z_np, axis=1)

    @staticmethod
    def _z_entropy(z: torch.Tensor) -> np.ndarray:
        """Entropy of each z vector treated as a distribution after softmax."""
        probs = torch.softmax(z.float(), dim=-1).cpu().numpy()
        # Clamp to avoid log(0)
        probs = np.clip(probs, 1e-12, None)
        entropy = -np.sum(probs * np.log(probs), axis=1)
        return entropy

    @staticmethod
    def _output_lengths(decoded_tokens: list[list[int]]) -> np.ndarray:
        """Number of tokens in each decoded output (before EOS)."""
        return np.array([len(seq) for seq in decoded_tokens])

    @staticmethod
    def _output_unique_counts(decoded_tokens: list[list[int]]) -> np.ndarray:
        """Number of unique tokens in each decoded output."""
        return np.array([len(set(seq)) if len(seq) > 0 else 0 for seq in decoded_tokens])

    # ------------------------------------------------------------------
    # Figures
    # ------------------------------------------------------------------

    def _fig_input_vs_output_length(
        self,
        input_lens: np.ndarray,
        output_lens: np.ndarray,
    ) -> Figure:
        """Scatter plot: input token count vs decoded output token count."""
        fig, ax = plt.subplots(figsize=FIGSIZE_SINGLE)

        ax.scatter(
            input_lens,
            output_lens,
            alpha=0.3,
            s=10,
            color="#1f77b4",
            edgecolors="none",
            rasterized=True,
        )

        # Regression line + Pearson r
        if len(input_lens) > 2:
            coeffs = np.polyfit(input_lens, output_lens, 1)
            x_line = np.linspace(input_lens.min(), input_lens.max(), 100)
            y_line = np.polyval(coeffs, x_line)
            r, p = pearsonr(input_lens, output_lens)
            ax.plot(
                x_line,
                y_line,
                color="#d62728",
                linewidth=2,
                label=f"slope={coeffs[0]:.3f}, r={r:.3f} (p={p:.2e})",
            )
            ax.legend()

        ax.set_xlabel("Input token count")
        ax.set_ylabel("Decoded output token count")
        ax.set_title("Adaptive Length: Input Complexity \u2192 Output Length")
        fig.tight_layout()
        return fig

    def _fig_z_norm_vs_diversity(
        self,
        norms: np.ndarray,
        unique_counts: np.ndarray,
    ) -> Figure:
        """Scatter plot: L2 norm of z vs output unique token count."""
        fig, ax = plt.subplots(figsize=FIGSIZE_SINGLE)

        ax.scatter(
            norms,
            unique_counts,
            alpha=0.3,
            s=10,
            color="#2ca02c",
            edgecolors="none",
            rasterized=True,
        )

        # Regression line + Pearson r
        if len(norms) > 2:
            coeffs = np.polyfit(norms, unique_counts, 1)
            x_line = np.linspace(norms.min(), norms.max(), 100)
            y_line = np.polyval(coeffs, x_line)
            r, p = pearsonr(norms, unique_counts)
            ax.plot(
                x_line,
                y_line,
                color="#d62728",
                linewidth=2,
                label=f"slope={coeffs[0]:.3f}, r={r:.3f} (p={p:.2e})",
            )
            ax.legend()

        ax.set_xlabel("L2 norm of z")
        ax.set_ylabel("Output unique token count")
        ax.set_title("Latent Richness vs Output Diversity")
        fig.tight_layout()
        return fig

    def _fig_complexity_profile(
        self,
        input_lens: np.ndarray,
        output_lens: np.ndarray,
        output_ttr: np.ndarray,
    ) -> Figure:
        """Box plots of output properties across input complexity quintiles."""
        fig, (ax_len, ax_ttr) = plt.subplots(1, 2, figsize=FIGSIZE_WIDE)

        # Bin inputs into 5 complexity quintiles by input length
        quintile_edges = np.percentile(input_lens, [0, 20, 40, 60, 80, 100])
        # Assign each sample to a quintile
        bins = np.digitize(input_lens, quintile_edges[1:-1])  # 0..4

        quintile_labels = []
        for i in range(5):
            lo = quintile_edges[i]
            hi = quintile_edges[i + 1]
            quintile_labels.append(f"Q{i + 1}\n[{lo:.0f}-{hi:.0f}]")

        # Gather data per quintile
        len_groups = [output_lens[bins == i] for i in range(5)]
        ttr_groups = [output_ttr[bins == i] for i in range(5)]

        # Output length box plot
        bp_len = ax_len.boxplot(
            len_groups,
            labels=quintile_labels,
            patch_artist=True,
            showfliers=False,
            medianprops={"color": "black", "linewidth": 1.5},
        )
        for patch in bp_len["boxes"]:
            patch.set_facecolor("#1f77b4")
            patch.set_alpha(0.7)

        ax_len.set_xlabel("Input complexity quintile")
        ax_len.set_ylabel("Output length (tokens)")
        ax_len.set_title("Output Length by Quintile")

        # Output TTR box plot
        bp_ttr = ax_ttr.boxplot(
            ttr_groups,
            labels=quintile_labels,
            patch_artist=True,
            showfliers=False,
            medianprops={"color": "black", "linewidth": 1.5},
        )
        for patch in bp_ttr["boxes"]:
            patch.set_facecolor("#ff7f0e")
            patch.set_alpha(0.7)

        ax_ttr.set_xlabel("Input complexity quintile")
        ax_ttr.set_ylabel("Output type-token ratio")
        ax_ttr.set_title("Output TTR by Quintile")

        fig.suptitle("Output Properties by Input Complexity Quintile", fontsize=14, y=1.02)
        fig.tight_layout()
        return fig

    # ------------------------------------------------------------------
    # Main entry
    # ------------------------------------------------------------------

    def generate(self, data: dict) -> list[Figure]:
        """Generate adaptiveness analysis figures.

        Args:
            data: Shared data dict containing z vectors, token_ids_list,
                  decoder modules, and config.

        Returns:
            Three figures: [input_vs_output_length, z_norm_vs_diversity,
            complexity_profile].
        """
        apply_style()

        z = data["z"]
        token_ids_list = data["token_ids_list"]

        # Take a subset for decoding
        n = min(_MAX_DECODE, self.config.max_samples, z.size(0))
        z_subset = z[:n]
        input_subset = token_ids_list[:n]

        logger.info("Decoding %d z vectors for adaptiveness analysis...", n)
        decoded_tokens = decode_z(z_subset, data, self.config)

        # Input complexity metrics
        input_lens = self._input_lengths(input_subset)
        input_ttr = self._type_token_ratio(input_subset)
        norms = self._z_norms(z_subset)
        z_ent = self._z_entropy(z_subset)

        # Output metrics
        output_lens = self._output_lengths(decoded_tokens)
        output_unique = self._output_unique_counts(decoded_tokens)
        output_ttr = self._type_token_ratio(decoded_tokens)

        logger.info(
            "Input — mean len: %.1f, mean TTR: %.3f, mean z-norm: %.3f, mean z-ent: %.3f",
            input_lens.mean(),
            input_ttr.mean(),
            norms.mean(),
            z_ent.mean(),
        )
        logger.info(
            "Output — mean len: %.1f, mean TTR: %.3f, mean unique: %.1f",
            output_lens.mean(),
            output_ttr.mean(),
            output_unique.mean(),
        )

        # Correlations summary
        if n > 2:
            r_len, p_len = pearsonr(input_lens, output_lens)
            r_div, p_div = pearsonr(norms, output_unique)
            logger.info(
                "Correlations — input_len↔output_len: r=%.3f (p=%.2e), "
                "z_norm↔output_unique: r=%.3f (p=%.2e)",
                r_len,
                p_len,
                r_div,
                p_div,
            )

        # Generate figures
        fig1 = self._fig_input_vs_output_length(input_lens, output_lens)
        fig2 = self._fig_z_norm_vs_diversity(norms, output_unique)
        fig3 = self._fig_complexity_profile(input_lens, output_lens, output_ttr)

        return [fig1, fig2, fig3]

    def save(self, figures: list[Figure], suffixes: list[str] | None = None) -> list:
        """Save with descriptive suffixes."""
        return super().save(
            figures,
            suffixes=["input_vs_output_length", "z_norm_vs_diversity", "complexity_profile"],
        )

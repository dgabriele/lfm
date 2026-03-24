"""Compositionality analysis of the VAE latent space.

Measures whether specific input features (z dimensions) map to specific
output positions — the hallmark of compositional communication.

Generates three figures:
1. Positional disentanglement heatmap (z dimension x output position correlation).
2. Disentanglement score distribution across all latent dimensions.
3. Feature-position mutual information (top 50 dimensions).
"""

from __future__ import annotations

import logging

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from matplotlib.figure import Figure
from scipy.stats import entropy as scipy_entropy
from lfm.visualize import BaseVisualization
from lfm.visualize.config import VisualizeConfig
from lfm.visualize.loader import decode_z
from lfm.visualize.style import FIGSIZE_SINGLE, FIGSIZE_WIDE, apply_style

logger = logging.getLogger(__name__)

# Padding value for sequences shorter than max length
_PAD_VALUE = -1


class CompositionalityVisualization(BaseVisualization):
    """Measure and visualize compositionality of the learned code."""

    name = "compositionality"

    def __init__(self, config: VisualizeConfig) -> None:
        super().__init__(config)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _pad_sequences(
        sequences: list[list[int]], max_len: int
    ) -> tuple[np.ndarray, np.ndarray]:
        """Pad variable-length token sequences into a fixed-width matrix.

        Args:
            sequences: Decoded token ID lists (variable length).
            max_len: Pad / truncate to this length.

        Returns:
            token_matrix: ``(N, max_len)`` int array, padded with ``_PAD_VALUE``.
            mask: ``(N, max_len)`` bool array, True where tokens are valid.
        """
        n = len(sequences)
        token_matrix = np.full((n, max_len), _PAD_VALUE, dtype=np.int64)
        mask = np.zeros((n, max_len), dtype=bool)

        for i, seq in enumerate(sequences):
            length = min(len(seq), max_len)
            token_matrix[i, :length] = seq[:length]
            mask[i, :length] = True

        return token_matrix, mask

    @staticmethod
    def _pearson_correlation_masked(
        x: np.ndarray, y: np.ndarray, mask: np.ndarray
    ) -> float:
        """Pearson correlation between *x* and *y* using only masked entries.

        Returns 0.0 when fewer than 3 valid entries or zero variance.
        """
        valid = mask.astype(bool)
        if valid.sum() < 3:
            return 0.0

        xv = x[valid].astype(np.float64)
        yv = y[valid].astype(np.float64)

        xv_centered = xv - xv.mean()
        yv_centered = yv - yv.mean()

        denom = np.sqrt((xv_centered ** 2).sum() * (yv_centered ** 2).sum())
        if denom < 1e-12:
            return 0.0

        return float((xv_centered * yv_centered).sum() / denom)

    # ------------------------------------------------------------------
    # Figure 1: Positional Disentanglement Heatmap
    # ------------------------------------------------------------------

    def _fig_disentanglement_heatmap(
        self,
        z: np.ndarray,
        token_matrix: np.ndarray,
        mask: np.ndarray,
        max_seq_len: int,
    ) -> Figure:
        """Heatmap of Pearson correlation between top-variance z dims and output positions."""
        n_dims = z.shape[1]
        n_top = min(20, n_dims)

        # Select top-variance z dimensions
        variances = np.var(z, axis=0)
        top_dims = np.argsort(variances)[::-1][:n_top]

        # Compute correlation matrix: (n_top, max_seq_len)
        corr_matrix = np.zeros((n_top, max_seq_len))
        for row, dim_idx in enumerate(top_dims):
            z_col = z[:, dim_idx]
            for pos in range(max_seq_len):
                corr_matrix[row, pos] = self._pearson_correlation_masked(
                    z_col, token_matrix[:, pos].astype(np.float64), mask[:, pos]
                )

        fig, ax = plt.subplots(figsize=FIGSIZE_WIDE)
        sns.heatmap(
            corr_matrix,
            center=0,
            cmap="RdBu_r",
            xticklabels=list(range(max_seq_len)),
            yticklabels=[f"z[{d}]" for d in top_dims],
            cbar_kws={"label": "Pearson r", "shrink": 0.8},
            ax=ax,
            rasterized=True,
        )
        ax.set_xlabel("Output Position")
        ax.set_ylabel("Latent Dimension (top 20 by variance)")
        ax.set_title(
            "Positional Disentanglement: Input Dimension \u2192 Output Position"
        )

        # Reduce x-tick clutter for long sequences
        if max_seq_len > 30:
            step = max(1, max_seq_len // 15)
            ax.set_xticks(range(0, max_seq_len, step))
            ax.set_xticklabels(range(0, max_seq_len, step))

        fig.tight_layout()
        return fig

    # ------------------------------------------------------------------
    # Figure 2: Disentanglement Score Distribution
    # ------------------------------------------------------------------

    def _fig_functional_compositionality(
        self,
        z: np.ndarray,
        token_matrix: np.ndarray,
        mask: np.ndarray,
        max_seq_len: int,
    ) -> Figure:
        """Heatmap: which z dimensions control which output properties.

        Correlates each of the top-variance z dims with interpretable output
        properties (length, TTR, unique token count, mean token ID).  Shows
        functional compositionality — specific dims controlling specific
        linguistic properties — rather than positional disentanglement.
        """
        import seaborn as sns

        n_samples = z.shape[0]

        # Compute output properties per sample
        lengths = mask.sum(axis=1).astype(np.float64)
        unique_counts = np.array([
            len(set(token_matrix[i, :int(lengths[i])]))
            for i in range(n_samples)
        ], dtype=np.float64)
        ttrs = np.where(lengths > 0, unique_counts / lengths, 0.0)
        mean_tok = np.array([
            token_matrix[i, :int(lengths[i])].mean() if lengths[i] > 0 else 0.0
            for i in range(n_samples)
        ], dtype=np.float64)

        properties = {
            "Output length": lengths,
            "Unique tokens": unique_counts,
            "Type-token ratio": ttrs,
            "Mean token ID": mean_tok,
        }

        # Select top 30 z dims by variance
        z_var = z.var(axis=0)
        top_dims = np.argsort(z_var)[::-1][:30]

        # Compute correlation matrix: (n_props, n_dims)
        prop_names = list(properties.keys())
        corr_matrix = np.zeros((len(prop_names), len(top_dims)))
        for i, name in enumerate(prop_names):
            prop = properties[name]
            for j, dim in enumerate(top_dims):
                valid = np.isfinite(prop) & np.isfinite(z[:, dim])
                if valid.sum() > 10:
                    r = np.corrcoef(z[valid, dim], prop[valid])[0, 1]
                    corr_matrix[i, j] = r if np.isfinite(r) else 0.0

        fig, ax = plt.subplots(figsize=FIGSIZE_WIDE)
        sns.heatmap(
            corr_matrix,
            ax=ax,
            center=0,
            cmap="RdBu_r",
            vmin=-0.5,
            vmax=0.5,
            xticklabels=[str(d) for d in top_dims],
            yticklabels=prop_names,
            annot=True,
            fmt=".2f",
            annot_kws={"fontsize": 7},
        )
        ax.set_xlabel("Latent Dimension (top 30 by variance)")
        ax.set_title(
            "Functional Compositionality: Which z Dims Control Which Properties"
        )
        plt.xticks(fontsize=7, rotation=90)
        fig.tight_layout()
        return fig

    # ------------------------------------------------------------------
    # Figure 3: Feature-Position Mutual Information
    # ------------------------------------------------------------------

    def _fig_probe_r2(
        self,
        z: np.ndarray,
        token_matrix: np.ndarray,
        mask: np.ndarray,
        max_seq_len: int,
    ) -> Figure:
        """Bar chart of per-dimension probe R²: how predictable is each z dim from output."""
        from sklearn.linear_model import Ridge

        n_samples, n_dims = z.shape

        # Build output feature matrix: bag-of-tokens per sample
        # (more informative than position-specific tokens for probe)
        vocab_size = int(token_matrix.max()) + 1
        bow = np.zeros((n_samples, min(vocab_size, 2000)), dtype=np.float32)
        for i in range(n_samples):
            length = mask[i].sum()
            for tok in token_matrix[i, :length]:
                if 0 <= tok < bow.shape[1]:
                    bow[i, tok] += 1
            # Normalize to frequencies
            if length > 0:
                bow[i] /= length

        # Also add output length as a feature
        lengths = mask.sum(axis=1, keepdims=True).astype(np.float32)
        features = np.concatenate([bow, lengths], axis=1)

        # Train/test split
        split = int(n_samples * 0.8)
        X_train, X_test = features[:split], features[split:]

        # Probe each z dimension
        r2_scores = np.zeros(n_dims)
        for dim_idx in range(n_dims):
            y_train = z[:split, dim_idx]
            y_test = z[split:, dim_idx]
            y_var = y_test.var()
            if y_var < 1e-10:
                continue
            model = Ridge(alpha=1.0)
            model.fit(X_train, y_train)
            pred = model.predict(X_test)
            ss_res = ((y_test - pred) ** 2).mean()
            r2_scores[dim_idx] = max(0.0, 1.0 - ss_res / y_var)

        # Sort descending, take top 50
        n_top = min(50, n_dims)
        sorted_idx = np.argsort(r2_scores)[::-1][:n_top]
        sorted_r2 = r2_scores[sorted_idx]

        fig, ax = plt.subplots(figsize=FIGSIZE_WIDE)
        ax.bar(
            range(n_top), sorted_r2, width=0.8,
            color="#2ca02c", edgecolor="none",
        )
        ax.set_xlabel("Latent Dimension (sorted by R²)")
        ax.set_ylabel("Probe R²")
        ax.set_title(
            "Output → Input Probe: How Recoverable Is Each z Dimension?"
        )
        ax.axhline(
            y=r2_scores.mean(), color="#d62728", linestyle="--",
            label=f"Mean R²={r2_scores.mean():.3f}",
        )
        ax.legend()

        if n_top <= 50:
            ax.set_xticks(range(n_top))
            ax.set_xticklabels(
                [str(d) for d in sorted_idx],
                rotation=90, fontsize=7,
            )
        ax.set_xlim(-0.5, n_top - 0.5)
        fig.tight_layout()
        return fig

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate(self, data: dict) -> list[Figure]:
        """Generate compositionality analysis figures.

        Args:
            data: Shared data dict with ``z``, ``token_ids_list``,
                  ``vocab_size``, ``modules``, ``device``, ``cfg``, and
                  decoder prerequisites (``rope_freqs``, ``cached_mask``,
                  ``full_vocab``, ``bos_id``, ``eos_id``).

        Returns:
            Three figures: [heatmap, score_distribution, mutual_information].
        """
        apply_style()

        z_tensor = data["z"]
        cfg = data["cfg"]
        max_seq_len = cfg.max_seq_len

        # Convert z to numpy, subsample if needed
        z_np = (
            z_tensor.cpu().numpy()
            if isinstance(z_tensor, torch.Tensor)
            else np.asarray(z_tensor)
        )
        n_samples = min(len(z_np), self.config.max_samples, 2000)
        if n_samples < len(z_np):
            rng = np.random.RandomState(self.config.seed)
            idx = rng.choice(len(z_np), size=n_samples, replace=False)
            z_sub = z_np[idx]
        else:
            z_sub = z_np

        logger.info(
            "Compositionality analysis: %d samples, %d latent dims, max_seq_len=%d",
            z_sub.shape[0], z_sub.shape[1], max_seq_len,
        )

        # Decode z vectors through the decoder
        z_decode = torch.from_numpy(z_sub).float()
        logger.info("Decoding %d latent vectors...", z_decode.shape[0])
        decoded_seqs = decode_z(z_decode, data, self.config)
        logger.info(
            "Decoded %d sequences (mean length %.1f)",
            len(decoded_seqs),
            np.mean([len(s) for s in decoded_seqs]) if decoded_seqs else 0,
        )

        # Pad to fixed-width matrix
        token_matrix, mask = self._pad_sequences(decoded_seqs, max_seq_len)

        # Figure 1: Positional Disentanglement Heatmap
        logger.info("Computing positional disentanglement heatmap...")
        fig_heatmap = self._fig_disentanglement_heatmap(
            z_sub, token_matrix, mask, max_seq_len
        )

        # Figure 2: Functional Compositionality
        logger.info("Computing functional compositionality...")
        fig_scores = self._fig_functional_compositionality(
            z_sub, token_matrix, mask, max_seq_len
        )

        # Figure 3: Probe R² — how recoverable is each z dim from output
        logger.info("Computing output→input probe R²...")
        fig_mi = self._fig_probe_r2(
            z_sub, token_matrix, mask, max_seq_len
        )

        return [fig_heatmap, fig_scores, fig_mi]

    def save(
        self, figures: list[Figure], suffixes: list[str] | None = None
    ) -> list:
        """Save with descriptive suffixes."""
        return super().save(
            figures,
            suffixes=["heatmap", "functional", "probe_r2"],
        )

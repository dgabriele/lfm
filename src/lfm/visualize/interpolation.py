"""Latent space interpolation between language pairs.

Generates two figures per run:
1. t-SNE scatter with interpolation trajectories overlaid.
2. Decoded IPA text at each interpolation step (text table).
"""

from __future__ import annotations

import logging
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.figure import Figure
from sklearn.manifold import TSNE

from lfm.visualize import BaseVisualization
from lfm.visualize.languages import LANGUAGES
from lfm.visualize.loader import decode_z
from lfm.visualize.style import FIGSIZE_TALL, FIGSIZE_WIDE, LANG_COLORS, apply_style

logger = logging.getLogger(__name__)

# Colors cycled across interpolation pairs when LANG_COLORS is insufficient
_PAIR_COLORS = [
    "#e41a1c",
    "#377eb8",
    "#4daf4a",
    "#984ea3",
    "#ff7f00",
    "#a65628",
    "#f781bf",
    "#999999",
]

# Maximum background points to include in the t-SNE projection
_MAX_BG_POINTS = 5000


class InterpolationVisualization(BaseVisualization):
    """Visualize latent-space interpolation trajectories between language pairs."""

    @property
    def name(self) -> str:
        return "interpolation"

    # ------------------------------------------------------------------
    # Pair parsing
    # ------------------------------------------------------------------

    def _parse_pairs(self) -> list[tuple[str, str]]:
        """Parse ``self.config.pairs`` into a list of (lang_a, lang_b) tuples.

        Format: ``"pol-vie,ara-fin"`` (comma-separated, dash-delimited).
        Defaults to ``"pol-vie,ara-fin"`` when the config value is empty.
        """
        raw = self.config.pairs or "pol-vie,ara-fin"
        pairs: list[tuple[str, str]] = []
        for token in raw.split(","):
            token = token.strip()
            if "-" not in token:
                logger.warning("Skipping malformed pair token: %s", token)
                continue
            a, b = token.split("-", 1)
            pairs.append((a.strip(), b.strip()))
        return pairs

    # ------------------------------------------------------------------
    # Per-language mean z
    # ------------------------------------------------------------------

    @staticmethod
    def _language_means(
        z: torch.Tensor, languages: list[str]
    ) -> dict[str, np.ndarray]:
        """Compute mean z vector for each language code."""
        z_np = z.cpu().numpy() if isinstance(z, torch.Tensor) else np.asarray(z)

        buckets: dict[str, list[np.ndarray]] = defaultdict(list)
        for vec, lang in zip(z_np, languages):
            buckets[lang].append(vec)

        return {
            code: np.mean(np.stack(vecs), axis=0)
            for code, vecs in buckets.items()
        }

    # ------------------------------------------------------------------
    # Interpolation
    # ------------------------------------------------------------------

    @staticmethod
    def _interpolate(
        z_a: np.ndarray, z_b: np.ndarray, steps: int
    ) -> np.ndarray:
        """Linear interpolation from *z_a* to *z_b* in *steps* points.

        Returns:
            Array of shape ``(steps, latent_dim)``.
        """
        ts = np.linspace(0.0, 1.0, steps)
        return np.stack([(1 - t) * z_a + t * z_b for t in ts])

    # ------------------------------------------------------------------
    # Figure 1: t-SNE trajectories
    # ------------------------------------------------------------------

    def _fig_tsne_trajectories(
        self,
        z: torch.Tensor,
        languages: list[str],
        interp_arrays: list[np.ndarray],
        pair_labels: list[tuple[str, str]],
    ) -> Figure:
        """Create a t-SNE scatter with interpolation paths overlaid."""
        z_np = z.cpu().numpy() if isinstance(z, torch.Tensor) else np.asarray(z)

        # Subsample background points for speed
        n_bg = min(len(z_np), _MAX_BG_POINTS)
        rng = np.random.RandomState(self.config.seed)
        bg_idx = rng.choice(len(z_np), size=n_bg, replace=False)
        bg_z = z_np[bg_idx]

        # Concatenate: [background, interp_pair_0, interp_pair_1, ...]
        interp_concat = np.concatenate(interp_arrays, axis=0)  # (total_interp, D)
        combined = np.concatenate([bg_z, interp_concat], axis=0)

        logger.info(
            "Running t-SNE on %d points (%d background + %d interpolation) ...",
            combined.shape[0],
            n_bg,
            interp_concat.shape[0],
        )
        tsne = TSNE(
            n_components=2,
            perplexity=min(self.config.perplexity, combined.shape[0] - 1),
            metric=self.config.metric,
            random_state=self.config.seed,
            init="pca",
            learning_rate="auto",
        )
        xy = tsne.fit_transform(combined)

        bg_xy = xy[:n_bg]
        interp_xy = xy[n_bg:]

        # Plot
        fig, ax = plt.subplots(figsize=FIGSIZE_WIDE)

        # Background scatter (light gray)
        ax.scatter(
            bg_xy[:, 0],
            bg_xy[:, 1],
            s=4,
            alpha=0.15,
            c="#cccccc",
            edgecolors="none",
            rasterized=True,
            label="_bg",
        )

        # Overlay each interpolation path
        offset = 0
        for i, (arr, (lang_a, lang_b)) in enumerate(
            zip(interp_arrays, pair_labels)
        ):
            n_pts = arr.shape[0]
            path_xy = interp_xy[offset : offset + n_pts]
            offset += n_pts

            color = _PAIR_COLORS[i % len(_PAIR_COLORS)]
            name_a = LANGUAGES[lang_a].name if lang_a in LANGUAGES else lang_a
            name_b = LANGUAGES[lang_b].name if lang_b in LANGUAGES else lang_b
            pair_label = f"{name_a} \u2192 {name_b}"

            # Connected line
            ax.plot(
                path_xy[:, 0],
                path_xy[:, 1],
                color=color,
                linewidth=2.0,
                alpha=0.8,
                zorder=3,
            )
            # Dots along the path
            ax.scatter(
                path_xy[:, 0],
                path_xy[:, 1],
                s=30,
                c=color,
                edgecolors="white",
                linewidths=0.5,
                zorder=4,
                label=pair_label,
            )
            # Start marker
            ax.scatter(
                path_xy[0, 0],
                path_xy[0, 1],
                s=120,
                c=color,
                marker="o",
                edgecolors="black",
                linewidths=1.2,
                zorder=5,
            )
            ax.annotate(
                name_a,
                xy=(path_xy[0, 0], path_xy[0, 1]),
                xytext=(8, 8),
                textcoords="offset points",
                fontsize=9,
                fontweight="bold",
                color=color,
                zorder=6,
            )
            # End marker
            ax.scatter(
                path_xy[-1, 0],
                path_xy[-1, 1],
                s=120,
                c=color,
                marker="s",
                edgecolors="black",
                linewidths=1.2,
                zorder=5,
            )
            ax.annotate(
                name_b,
                xy=(path_xy[-1, 0], path_xy[-1, 1]),
                xytext=(8, 8),
                textcoords="offset points",
                fontsize=9,
                fontweight="bold",
                color=color,
                zorder=6,
            )

        ax.set_title("Latent Interpolation Trajectories")
        ax.set_xlabel("t-SNE Dimension 1")
        ax.set_ylabel("t-SNE Dimension 2")
        ax.legend(loc="best", frameon=True, framealpha=0.9)
        fig.tight_layout()
        return fig

    # ------------------------------------------------------------------
    # Figure 2: decoded text table
    # ------------------------------------------------------------------

    def _fig_decoded_table(
        self,
        interp_arrays: list[np.ndarray],
        pair_labels: list[tuple[str, str]],
        data: dict,
    ) -> Figure:
        """Render decoded IPA text at each interpolation step."""
        import sentencepiece as spm_lib

        sp = spm_lib.SentencePieceProcessor(model_file=self.config.spm_model)
        vocab_size = data["vocab_size"]
        steps = interp_arrays[0].shape[0]

        # Decode all interpolation points
        decoded_texts: list[list[str]] = []
        for arr in interp_arrays:
            z_tensor = torch.from_numpy(arr).float()
            token_lists = decode_z(z_tensor, data, self.config)
            texts: list[str] = []
            for tokens in token_lists:
                clean = [t for t in tokens if t < vocab_size]
                texts.append(sp.decode(clean))
            decoded_texts.append(texts)

        n_pairs = len(pair_labels)
        ts = np.linspace(0.0, 1.0, steps)

        # Compute figure height: ~0.35 inches per row, plus margins per subplot
        row_height = 0.35
        subplot_margin = 1.2
        fig_height = max(
            n_pairs * (steps * row_height + subplot_margin) + 1.0,
            FIGSIZE_TALL[1],
        )
        fig, axes = plt.subplots(
            n_pairs, 1,
            figsize=(FIGSIZE_WIDE[0], fig_height),
            squeeze=False,
        )

        for idx, ((lang_a, lang_b), texts) in enumerate(
            zip(pair_labels, decoded_texts)
        ):
            ax = axes[idx, 0]
            ax.set_xlim(0, 1)
            ax.set_ylim(0, steps + 1)
            ax.invert_yaxis()
            ax.axis("off")

            name_a = LANGUAGES[lang_a].name if lang_a in LANGUAGES else lang_a
            name_b = LANGUAGES[lang_b].name if lang_b in LANGUAGES else lang_b
            ax.set_title(
                f"Interpolation: {name_a} \u2192 {name_b}",
                fontsize=12,
                fontweight="bold",
                pad=10,
            )

            # Column header
            ax.text(
                0.05, 0.5, "t", fontsize=10, fontweight="bold",
                verticalalignment="center", family="monospace",
            )
            ax.text(
                0.15, 0.5, "Decoded IPA", fontsize=10, fontweight="bold",
                verticalalignment="center",
            )

            for row, (t_val, text) in enumerate(zip(ts, texts)):
                y = row + 1.0
                # Shade alternating rows
                if row % 2 == 0:
                    ax.axhspan(
                        y - 0.4, y + 0.4,
                        facecolor="#f0f0f0", edgecolor="none", zorder=0,
                    )

                # Color gradient: start language color -> end language color
                frac = t_val
                c_start = np.array(
                    plt.cm.colors.to_rgba(
                        LANG_COLORS.get(lang_a, "#333333")
                    )[:3]
                )
                c_end = np.array(
                    plt.cm.colors.to_rgba(
                        LANG_COLORS.get(lang_b, "#333333")
                    )[:3]
                )
                color = (1 - frac) * c_start + frac * c_end

                ax.text(
                    0.05, y, f"{t_val:.2f}",
                    fontsize=9, verticalalignment="center",
                    family="monospace", color="#555555",
                )
                # Truncate long decoded strings for display
                display_text = text if len(text) <= 100 else text[:97] + "..."
                ax.text(
                    0.15, y, display_text,
                    fontsize=9, verticalalignment="center",
                    color=color,
                )

        fig.tight_layout()
        return fig

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate(self, data: dict) -> list[Figure]:
        """Generate interpolation trajectory and decoded-text figures.

        Args:
            data: Shared data dict with ``z``, ``languages``, model components,
                  ``token_ids_list``, ``vocab_size``, etc.

        Returns:
            Two figures: [t-SNE trajectories, decoded text table].
        """
        apply_style()

        pairs = self._parse_pairs()
        steps = self.config.steps
        logger.info(
            "Interpolation: %d pair(s), %d steps each", len(pairs), steps
        )

        z = data["z"]
        languages = data["languages"]
        lang_means = self._language_means(z, languages)

        # Build interpolation arrays for each pair
        interp_arrays: list[np.ndarray] = []
        valid_pairs: list[tuple[str, str]] = []

        for lang_a, lang_b in pairs:
            if lang_a not in lang_means:
                logger.warning(
                    "Language %s not found in data; skipping pair %s-%s",
                    lang_a, lang_a, lang_b,
                )
                continue
            if lang_b not in lang_means:
                logger.warning(
                    "Language %s not found in data; skipping pair %s-%s",
                    lang_b, lang_a, lang_b,
                )
                continue

            z_a = lang_means[lang_a]
            z_b = lang_means[lang_b]
            interp = self._interpolate(z_a, z_b, steps)
            interp_arrays.append(interp)
            valid_pairs.append((lang_a, lang_b))
            logger.info(
                "  %s -> %s: z_a norm=%.2f, z_b norm=%.2f, distance=%.2f",
                lang_a, lang_b,
                np.linalg.norm(z_a),
                np.linalg.norm(z_b),
                np.linalg.norm(z_a - z_b),
            )

        if not valid_pairs:
            logger.error("No valid language pairs; returning empty figure list")
            return []

        # Figure 1: t-SNE trajectories
        fig_tsne = self._fig_tsne_trajectories(
            z, languages, interp_arrays, valid_pairs
        )

        # Figure 2: decoded text table
        fig_text = self._fig_decoded_table(interp_arrays, valid_pairs, data)

        return [fig_tsne, fig_text]

    def save(
        self, figures: list[Figure], suffixes: list[str] | None = None
    ) -> list:
        """Save with descriptive suffixes."""
        return super().save(figures, suffixes=["trajectories", "decoded"])

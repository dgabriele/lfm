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

    def _parse_pairs(
        self, lang_means: dict[str, np.ndarray] | None = None,
    ) -> list[tuple[str, str]]:
        """Parse pairs or auto-select based on maximum z-space distance.

        If ``self.config.pairs`` is set, parse it (``"pol-vie,ara-fin"``).
        Otherwise, dynamically select 2 pairs that span the greatest
        distance in latent space, ensuring visually distinct trajectories.
        """
        raw = self.config.pairs
        if raw:
            pairs: list[tuple[str, str]] = []
            for token in raw.split(","):
                token = token.strip()
                if "-" not in token:
                    logger.warning("Skipping malformed pair token: %s", token)
                    continue
                a, b = token.split("-", 1)
                pairs.append((a.strip(), b.strip()))
            return pairs

        # Auto-select: find the 2 most distant language pairs in z-space
        if lang_means is None or len(lang_means) < 2:
            return [("pol", "vie"), ("ara", "fin")]

        codes = sorted(lang_means.keys())
        best_pairs: list[tuple[float, str, str]] = []
        for i, a in enumerate(codes):
            for b in codes[i + 1:]:
                dist = np.linalg.norm(lang_means[a] - lang_means[b])
                best_pairs.append((dist, a, b))
        best_pairs.sort(reverse=True)

        # Take the most distant pair, then the most distant pair that
        # doesn't share a language with the first
        selected: list[tuple[str, str]] = []
        used: set[str] = set()
        for _, a, b in best_pairs:
            if len(selected) >= 2:
                break
            if len(selected) == 0 or (a not in used and b not in used):
                selected.append((a, b))
                used.update([a, b])

        logger.info(
            "Auto-selected interpolation pairs: %s (by max z-distance)",
            selected,
        )
        return selected

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
        """Create a t-SNE scatter with interpolation paths overlaid.

        Runs t-SNE on the combined set (background + interpolation points)
        so that intermediate points are projected into the same space.
        Interpolation points are color-coded by t (gradient from start
        to end) to show the progression through latent space.
        """
        from matplotlib.colors import LinearSegmentedColormap

        from lfm.visualize.style import get_color_map

        z_np = z.cpu().numpy() if isinstance(z, torch.Tensor) else np.asarray(z)

        # Subsample background points for speed
        n_bg = min(len(z_np), _MAX_BG_POINTS)
        rng = np.random.RandomState(self.config.seed)
        bg_idx = rng.choice(len(z_np), size=n_bg, replace=False)
        bg_z = z_np[bg_idx]
        bg_langs = [languages[i] for i in bg_idx]

        # Concatenate background + all interpolation points
        interp_concat = np.concatenate(interp_arrays, axis=0)
        combined = np.concatenate([bg_z, interp_concat], axis=0)

        logger.info(
            "Running t-SNE on %d points (%d bg + %d interp)...",
            combined.shape[0], n_bg, interp_concat.shape[0],
        )
        tsne = TSNE(
            n_components=2,
            perplexity=min(self.config.perplexity, combined.shape[0] - 1),
            random_state=self.config.seed,
            init="pca",
            learning_rate="auto",
        )
        xy = tsne.fit_transform(combined)

        bg_xy = xy[:n_bg]
        interp_xy = xy[n_bg:]

        # Plot
        fig, ax = plt.subplots(figsize=FIGSIZE_WIDE)

        # Background scatter colored by family (more visible)
        family_colors = get_color_map("family")
        for lang_code in sorted(set(bg_langs)):
            mask = np.array([l == lang_code for l in bg_langs])
            info = LANGUAGES.get(lang_code)
            family = info.family if info else "Unknown"
            color = family_colors.get(family, "#cccccc")
            ax.scatter(
                bg_xy[mask, 0], bg_xy[mask, 1],
                s=8, alpha=0.3, c=color, edgecolors="none",
                rasterized=True, label="_bg",
            )

        # Overlay each interpolation path with t-colored markers
        offset = 0
        for i, (arr, (lang_a, lang_b)) in enumerate(
            zip(interp_arrays, pair_labels)
        ):
            n_pts = arr.shape[0]
            path_xy = interp_xy[offset : offset + n_pts]
            offset += n_pts

            pair_color = _PAIR_COLORS[i % len(_PAIR_COLORS)]
            name_a = LANGUAGES[lang_a].name if lang_a in LANGUAGES else lang_a
            name_b = LANGUAGES[lang_b].name if lang_b in LANGUAGES else lang_b
            pair_label = f"{name_a} \u2192 {name_b}"

            # Connected line through all interpolation points
            ax.plot(
                path_xy[:, 0], path_xy[:, 1],
                color=pair_color, linewidth=1.5, alpha=0.6, zorder=3,
            )

            # Color-code points by t (dark at start, light at end)
            ts = np.linspace(0, 1, n_pts)
            scatter = ax.scatter(
                path_xy[:, 0], path_xy[:, 1],
                s=25, c=ts, cmap="coolwarm", edgecolors=pair_color,
                linewidths=0.5, zorder=4, vmin=0, vmax=1,
            )

            # Dummy for legend
            ax.plot([], [], color=pair_color, linewidth=2.5, label=pair_label)

            # Start marker (circle) — offset label away from path
            ax.scatter(
                path_xy[0, 0], path_xy[0, 1],
                s=200, c=pair_color, marker="o", edgecolors="black",
                linewidths=1.5, zorder=5,
            )
            # Smart label placement: offset away from the midpoint
            mid_xy = path_xy[n_pts // 2]
            dx_start = path_xy[0, 0] - mid_xy[0]
            dy_start = path_xy[0, 1] - mid_xy[1]
            norm_s = max(np.sqrt(dx_start**2 + dy_start**2), 1e-6)
            ax.annotate(
                name_a, xy=(path_xy[0, 0], path_xy[0, 1]),
                xytext=(15 * dx_start / norm_s, 15 * dy_start / norm_s),
                textcoords="offset points",
                fontsize=11, fontweight="bold", color=pair_color, zorder=6,
            )

            # End marker (square)
            ax.scatter(
                path_xy[-1, 0], path_xy[-1, 1],
                s=200, c=pair_color, marker="s", edgecolors="black",
                linewidths=1.5, zorder=5,
            )
            dx_end = path_xy[-1, 0] - mid_xy[0]
            dy_end = path_xy[-1, 1] - mid_xy[1]
            norm_e = max(np.sqrt(dx_end**2 + dy_end**2), 1e-6)
            ax.annotate(
                name_b, xy=(path_xy[-1, 0], path_xy[-1, 1]),
                xytext=(15 * dx_end / norm_e, 15 * dy_end / norm_e),
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

            # Precompute language endpoint colors for the margin indicator
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

            for row, (t_val, text) in enumerate(zip(ts, texts)):
                y = row + 1.0
                # Shade alternating rows
                if row % 2 == 0:
                    ax.axhspan(
                        y - 0.4, y + 0.4,
                        facecolor="#f0f0f0", edgecolor="none", zorder=0,
                    )

                # Left-margin color indicator showing language transition
                frac = t_val
                indicator_color = (1 - frac) * c_start + frac * c_end
                ax.axhspan(
                    y - 0.4, y + 0.4,
                    xmin=0.0, xmax=0.02,
                    facecolor=indicator_color, edgecolor="none", zorder=1,
                )

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
                    color="black",
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

        z = data["z"]
        languages = data["languages"]
        lang_means = self._language_means(z, languages)

        pairs = self._parse_pairs(lang_means)
        steps = self.config.steps
        logger.info(
            "Interpolation: %d pair(s), %d steps each", len(pairs), steps
        )

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

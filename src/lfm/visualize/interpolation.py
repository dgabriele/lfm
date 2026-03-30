"""Latent space interpolation between language pairs.

Generates two figures per run:
1. t-SNE scatter with interpolation trajectories overlaid.
2. Decoded IPA text at each interpolation step (text table).
"""

from __future__ import annotations

import logging
from collections import Counter, defaultdict

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.figure import Figure
from sklearn.manifold import TSNE

from lfm.visualize import BaseVisualization
from lfm.visualize.languages import get_label
from lfm.visualize.loader import decode_z
from lfm.visualize.style import (
    FIGSIZE_SINGLE,
    FIGSIZE_TALL,
    FIGSIZE_WIDE,
    apply_style,
    get_color_map,
)

logger = logging.getLogger(__name__)

# Distinct colors for interpolation paths (high contrast, colorblind-safe)
_PATH_COLORS = [
    "#2166ac",  # strong blue
    "#b2182b",  # strong red
    "#1b7837",  # forest green
    "#762a83",  # purple
    "#e08214",  # orange
    "#01665e",  # teal
    "#d6604d",  # salmon
    "#5aae61",  # lime
]

# Background points for the t-SNE projection — large enough to avoid
# the interpolation points dominating the projection.
_MAX_BG_POINTS = 12000

# Minimum samples for a language to be eligible for auto pair selection.
# Languages with fewer points are peripheral outliers.
_MIN_SAMPLES_FOR_PAIR = 1000


class InterpolationVisualization(BaseVisualization):
    """Visualize latent-space interpolation trajectories between language pairs."""

    @property
    def name(self) -> str:
        return "interpolation"

    # ------------------------------------------------------------------
    # Pair selection — fully dynamic, no hardcoded languages
    # ------------------------------------------------------------------

    def _parse_pairs(
        self,
        lang_means: dict[str, np.ndarray] | None = None,
        lang_counts: dict[str, int] | None = None,
    ) -> list[tuple[str, str]]:
        """Parse pairs or auto-select based on maximum z-space distance.

        If ``self.config.pairs`` is set, parse it (``"pol-vie,ara-fin"``).
        Otherwise, dynamically select 2 pairs that span the greatest
        distance in latent space from languages with sufficient samples.
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

        if lang_means is None or len(lang_means) < 2:
            # Return empty — caller will handle gracefully
            logger.warning("Cannot auto-select pairs: fewer than 2 languages")
            return []

        # Filter to well-represented languages (avoid outlier clusters)
        counts = lang_counts or {}
        eligible = {
            code for code, mean in lang_means.items()
            if counts.get(code, 0) >= _MIN_SAMPLES_FOR_PAIR
        }

        # If too few pass the threshold, relax to all available
        if len(eligible) < 4:
            eligible = set(lang_means.keys())

        eligible_codes = sorted(eligible)
        logger.info(
            "Auto-selecting pairs from %d eligible languages (>=%d samples): %s",
            len(eligible_codes),
            _MIN_SAMPLES_FOR_PAIR,
            eligible_codes,
        )

        # Rank all pairs by z-distance
        ranked: list[tuple[float, str, str]] = []
        for i, a in enumerate(eligible_codes):
            for b in eligible_codes[i + 1:]:
                dist = float(np.linalg.norm(lang_means[a] - lang_means[b]))
                ranked.append((dist, a, b))
        ranked.sort(reverse=True)

        # Greedily select 2 pairs with no shared languages
        selected: list[tuple[str, str]] = []
        used: set[str] = set()
        for _, a, b in ranked:
            if a not in used and b not in used:
                selected.append((a, b))
                used.update([a, b])
            if len(selected) >= 2:
                break

        logger.info(
            "Auto-selected interpolation pairs: %s (by max z-distance)",
            selected,
        )
        return selected

    # ------------------------------------------------------------------
    # Per-language mean z
    # ------------------------------------------------------------------

    @staticmethod
    def _language_stats(
        z: torch.Tensor, languages: list[str]
    ) -> tuple[dict[str, np.ndarray], dict[str, int]]:
        """Compute mean z vector and sample count for each language code."""
        z_np = z.cpu().numpy() if isinstance(z, torch.Tensor) else np.asarray(z)

        buckets: dict[str, list[np.ndarray]] = defaultdict(list)
        for vec, lang in zip(z_np, languages):
            buckets[lang].append(vec)

        means = {
            code: np.mean(np.stack(vecs), axis=0)
            for code, vecs in buckets.items()
        }
        counts = {code: len(vecs) for code, vecs in buckets.items()}
        return means, counts

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

        Uses a large background sample (12K+ points) so interpolation
        points don't distort the projection. Square aspect ratio with
        equal axes. Labels placed with offset arrows to avoid overlap.
        """
        z_np = z.cpu().numpy() if isinstance(z, torch.Tensor) else np.asarray(z)

        # Subsample background points
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

        # --- Plot ---
        fig, ax = plt.subplots(figsize=FIGSIZE_SINGLE)

        # Background scatter colored by language family
        unique_bg_codes = sorted(set(bg_langs))
        unique_families = sorted({get_label(c, "family") for c in unique_bg_codes})
        family_colors = get_color_map("family", keys=unique_families)

        for lang_code in unique_bg_codes:
            mask = np.array([lang == lang_code for lang in bg_langs])
            family = get_label(lang_code, "family")
            color = family_colors.get(family, "#cccccc")
            ax.scatter(
                bg_xy[mask, 0], bg_xy[mask, 1],
                s=8, alpha=0.20, c=color, edgecolors="none",
                rasterized=True, label="_bg",
            )

        # Overlay each interpolation path
        offset = 0
        for i, (arr, (lang_a, lang_b)) in enumerate(
            zip(interp_arrays, pair_labels)
        ):
            n_pts = arr.shape[0]
            path_xy = interp_xy[offset: offset + n_pts]
            offset += n_pts

            path_color = _PATH_COLORS[i % len(_PATH_COLORS)]
            name_a = get_label(lang_a, "language")
            name_b = get_label(lang_b, "language")
            pair_str = f"{name_a} \u2192 {name_b}"

            # Connected line
            ax.plot(
                path_xy[:, 0], path_xy[:, 1],
                color=path_color, linewidth=2.0, alpha=0.7, zorder=3,
            )

            # Color-coded points by interpolation t
            ts = np.linspace(0, 1, n_pts)
            ax.scatter(
                path_xy[:, 0], path_xy[:, 1],
                s=30, c=ts, cmap="coolwarm", edgecolors=path_color,
                linewidths=0.6, zorder=4, vmin=0, vmax=1,
            )

            # Legend entry
            ax.plot([], [], color=path_color, linewidth=2.5, label=pair_str)

            # Start marker (circle) with arrow annotation
            ax.scatter(
                path_xy[0, 0], path_xy[0, 1],
                s=200, c=path_color, marker="o", edgecolors="black",
                linewidths=1.5, zorder=5,
            )
            # Offset direction: away from path midpoint
            mid_xy = path_xy[n_pts // 2]
            dx_s = path_xy[0, 0] - mid_xy[0]
            dy_s = path_xy[0, 1] - mid_xy[1]
            norm_s = max(np.sqrt(dx_s**2 + dy_s**2), 1e-6)
            # Annotate with arrow connector for clean label placement
            ax.annotate(
                name_a,
                xy=(path_xy[0, 0], path_xy[0, 1]),
                xytext=(30 * dx_s / norm_s, 30 * dy_s / norm_s),
                textcoords="offset points",
                fontsize=10, fontweight="bold", color=path_color,
                arrowprops=dict(
                    arrowstyle="->", color=path_color,
                    lw=1.2, connectionstyle="arc3,rad=0.15",
                ),
                zorder=6,
                bbox=dict(boxstyle="round,pad=0.2", fc="white", ec=path_color, alpha=0.8),
            )

            # End marker (square) with arrow annotation
            ax.scatter(
                path_xy[-1, 0], path_xy[-1, 1],
                s=200, c=path_color, marker="s", edgecolors="black",
                linewidths=1.5, zorder=5,
            )
            dx_e = path_xy[-1, 0] - mid_xy[0]
            dy_e = path_xy[-1, 1] - mid_xy[1]
            norm_e = max(np.sqrt(dx_e**2 + dy_e**2), 1e-6)
            ax.annotate(
                name_b,
                xy=(path_xy[-1, 0], path_xy[-1, 1]),
                xytext=(30 * dx_e / norm_e, 30 * dy_e / norm_e),
                textcoords="offset points",
                fontsize=10, fontweight="bold", color=path_color,
                arrowprops=dict(
                    arrowstyle="->", color=path_color,
                    lw=1.2, connectionstyle="arc3,rad=0.15",
                ),
                zorder=6,
                bbox=dict(boxstyle="round,pad=0.2", fc="white", ec=path_color, alpha=0.8),
            )

        ax.set_title("Latent Interpolation Trajectories")
        ax.set_xlabel("t-SNE Dimension 1")
        ax.set_ylabel("t-SNE Dimension 2")
        ax.set_aspect("equal")
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

        # Compute figure height
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

            name_a = get_label(lang_a, "language")
            name_b = get_label(lang_b, "language")
            path_color = _PATH_COLORS[idx % len(_PATH_COLORS)]

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

            # Color endpoints for margin indicator
            c_start = np.array(plt.cm.colors.to_rgba(path_color)[:3])
            # Use a complementary lighter shade for end
            c_end = np.clip(c_start * 0.5 + 0.5, 0, 1)

            for row, (t_val, text) in enumerate(zip(ts, texts)):
                y = row + 1.0
                if row % 2 == 0:
                    ax.axhspan(
                        y - 0.4, y + 0.4,
                        facecolor="#f0f0f0", edgecolor="none", zorder=0,
                    )

                # Left-margin color indicator
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
        lang_means, lang_counts = self._language_stats(z, languages)

        pairs = self._parse_pairs(lang_means, lang_counts)
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

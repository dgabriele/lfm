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
        """Parse pairs or auto-select diverse, well-separated pairs.

        If ``self.config.pairs`` is set, parse it (``"pol-vie,ara-fin"``).
        Otherwise, dynamically select 2 pairs that:
        1. Cross language family boundaries (typological diversity)
        2. Span the greatest distance in latent space
        3. Use well-represented languages (sufficient samples)
        4. Don't reuse languages or families between pairs
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

        # Rank cross-family pairs by z-distance (ensures typological diversity)
        ranked: list[tuple[float, str, str]] = []
        for i, a in enumerate(eligible_codes):
            family_a = get_label(a, "family")
            for b in eligible_codes[i + 1:]:
                family_b = get_label(b, "family")
                if family_a == family_b:
                    continue
                dist = float(np.linalg.norm(lang_means[a] - lang_means[b]))
                ranked.append((dist, a, b))
        ranked.sort(reverse=True)

        # Fallback: if no cross-family pairs, use all pairs
        if not ranked:
            for i, a in enumerate(eligible_codes):
                for b in eligible_codes[i + 1:]:
                    dist = float(np.linalg.norm(lang_means[a] - lang_means[b]))
                    ranked.append((dist, a, b))
            ranked.sort(reverse=True)

        # Greedily select 2 pairs with no shared languages or families
        selected: list[tuple[str, str]] = []
        used: set[str] = set()
        used_families: set[str] = set()
        for _, a, b in ranked:
            if a in used or b in used:
                continue
            family_a = get_label(a, "family")
            family_b = get_label(b, "family")
            if family_a in used_families or family_b in used_families:
                continue
            selected.append((a, b))
            used.update([a, b])
            used_families.update([family_a, family_b])
            if len(selected) >= 2:
                break

        # Relax family constraint if needed
        if len(selected) < 2:
            for _, a, b in ranked:
                if a not in used and b not in used:
                    selected.append((a, b))
                    used.update([a, b])
                if len(selected) >= 2:
                    break

        logger.info(
            "Auto-selected interpolation pairs: %s (by max z-distance, cross-family)",
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
        """Spherical interpolation (slerp) from *z_a* to *z_b*.

        Slerp keeps intermediate points on the hypersphere at
        consistent norm, avoiding empty regions of z-space that
        cause t-SNE trajectory collapse when endpoint norms differ.

        Falls back to linear interpolation when vectors are
        near-parallel (dot product ≈ 1).

        Returns:
            Array of shape ``(steps, latent_dim)``.
        """
        norm_a = np.linalg.norm(z_a)
        norm_b = np.linalg.norm(z_b)
        if norm_a < 1e-8 or norm_b < 1e-8:
            # Degenerate — fall back to lerp
            ts = np.linspace(0.0, 1.0, steps)
            return np.stack([(1 - t) * z_a + t * z_b for t in ts])

        u_a = z_a / norm_a
        u_b = z_b / norm_b
        dot = np.clip(np.dot(u_a, u_b), -1.0, 1.0)
        omega = np.arccos(dot)

        if omega < 1e-6:
            # Near-parallel — lerp is fine
            ts = np.linspace(0.0, 1.0, steps)
            return np.stack([(1 - t) * z_a + t * z_b for t in ts])

        ts = np.linspace(0.0, 1.0, steps)
        results = []
        for t in ts:
            # Slerp on unit sphere, then interpolate norms
            s = (np.sin((1 - t) * omega) * u_a + np.sin(t * omega) * u_b) / np.sin(omega)
            r = (1 - t) * norm_a + t * norm_b
            results.append(s * r)
        return np.stack(results)

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

        Uses a large background sample so interpolation points don't
        distort the projection. Axes are clipped to the main data
        distribution (percentile bounds) so outliers don't compress
        the visualization. Labels are placed at fixed offset angles
        to prevent overlap.
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

        # --- Compute axis limits from background data (exclude outliers) ---
        x_lo, x_hi = np.percentile(bg_xy[:, 0], [1, 99])
        y_lo, y_hi = np.percentile(bg_xy[:, 1], [1, 99])
        x_pad = (x_hi - x_lo) * 0.12
        y_pad = (y_hi - y_lo) * 0.12

        # --- Plot ---
        fig, ax = plt.subplots(figsize=FIGSIZE_SINGLE)

        # Background scatter colored by language family (subtle)
        unique_bg_codes = sorted(set(bg_langs))
        unique_families = sorted({get_label(c, "family") for c in unique_bg_codes})
        family_colors = get_color_map("family", keys=unique_families)

        for lang_code in unique_bg_codes:
            mask = np.array([lang == lang_code for lang in bg_langs])
            family = get_label(lang_code, "family")
            color = family_colors.get(family, "#cccccc")
            ax.scatter(
                bg_xy[mask, 0], bg_xy[mask, 1],
                s=6, alpha=0.15, c=color, edgecolors="none",
                rasterized=True, label="_bg",
            )

        # Fixed label offset directions for each endpoint (prevents overlap).
        # Angle in degrees from marker center, cycling through 4 quadrants.
        _LABEL_OFFSETS = [
            (-45, 35),   # pair 0 start: upper-left
            (45, -35),   # pair 0 end: lower-right
            (45, 35),    # pair 1 start: upper-right
            (-45, -35),  # pair 1 end: lower-left
            (-50, 0),    # pair 2 start: left
            (50, 0),     # pair 2 end: right
            (0, 40),     # pair 3 start: top
            (0, -40),    # pair 3 end: bottom
        ]

        # Overlay each interpolation path
        offset = 0
        label_idx = 0
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
                color=path_color, linewidth=2.5, alpha=0.8, zorder=3,
            )

            # Color-coded points by interpolation t
            ts = np.linspace(0, 1, n_pts)
            ax.scatter(
                path_xy[:, 0], path_xy[:, 1],
                s=35, c=ts, cmap="coolwarm", edgecolors=path_color,
                linewidths=0.8, zorder=4, vmin=0, vmax=1,
            )

            # Legend entry
            ax.plot([], [], color=path_color, linewidth=2.5, label=pair_str)

            # Start marker (circle) with arrow annotation
            ax.scatter(
                path_xy[0, 0], path_xy[0, 1],
                s=180, c=path_color, marker="o", edgecolors="black",
                linewidths=1.5, zorder=5,
            )
            ofs_start = _LABEL_OFFSETS[label_idx % len(_LABEL_OFFSETS)]
            label_idx += 1
            ax.annotate(
                name_a,
                xy=(path_xy[0, 0], path_xy[0, 1]),
                xytext=ofs_start,
                textcoords="offset points",
                fontsize=10, fontweight="bold", color=path_color,
                arrowprops=dict(
                    arrowstyle="->", color=path_color,
                    lw=1.2, connectionstyle="arc3,rad=0.15",
                ),
                zorder=6,
                bbox=dict(
                    boxstyle="round,pad=0.3", fc="white",
                    ec=path_color, lw=1.5, alpha=0.95,
                ),
            )

            # End marker (square) with arrow annotation
            ax.scatter(
                path_xy[-1, 0], path_xy[-1, 1],
                s=180, c=path_color, marker="s", edgecolors="black",
                linewidths=1.5, zorder=5,
            )
            ofs_end = _LABEL_OFFSETS[label_idx % len(_LABEL_OFFSETS)]
            label_idx += 1
            ax.annotate(
                name_b,
                xy=(path_xy[-1, 0], path_xy[-1, 1]),
                xytext=ofs_end,
                textcoords="offset points",
                fontsize=10, fontweight="bold", color=path_color,
                arrowprops=dict(
                    arrowstyle="->", color=path_color,
                    lw=1.2, connectionstyle="arc3,rad=0.15",
                ),
                zorder=6,
                bbox=dict(
                    boxstyle="round,pad=0.3", fc="white",
                    ec=path_color, lw=1.5, alpha=0.95,
                ),
            )

        # Clip axes to main data distribution (outliers don't compress plot)
        ax.set_xlim(x_lo - x_pad, x_hi + x_pad)
        ax.set_ylim(y_lo - y_pad, y_hi + y_pad)

        ax.set_title("Latent Interpolation Trajectories")
        ax.set_xlabel("t-SNE Dimension 1")
        ax.set_ylabel("t-SNE Dimension 2")
        ax.set_aspect("equal")
        ax.legend(loc="upper right", frameon=True, framealpha=0.9, fontsize=10)
        fig.tight_layout()
        return fig

    # ------------------------------------------------------------------
    # Figure 2: decoded text table
    # ------------------------------------------------------------------

    def _fig_pca_trajectories(
        self,
        z: torch.Tensor,
        languages: list[str],
        interp_arrays: list[np.ndarray],
        pair_labels: list[tuple[str, str]],
    ) -> Figure:
        """PCA projection with interpolation paths — preserves global geometry.

        Unlike t-SNE, PCA is a linear projection that preserves distances,
        so evenly-spaced interpolation points appear evenly-spaced.
        """
        from sklearn.decomposition import PCA

        z_np = z.cpu().numpy() if isinstance(z, torch.Tensor) else np.asarray(z)

        n_bg = min(len(z_np), _MAX_BG_POINTS)
        rng = np.random.RandomState(self.config.seed)
        bg_idx = rng.choice(len(z_np), size=n_bg, replace=False)
        bg_z = z_np[bg_idx]
        bg_langs = [languages[i] for i in bg_idx]

        interp_concat = np.concatenate(interp_arrays, axis=0)
        combined = np.concatenate([bg_z, interp_concat], axis=0)

        pca = PCA(n_components=2, random_state=self.config.seed)
        xy = pca.fit_transform(combined)

        bg_xy = xy[:n_bg]
        interp_xy = xy[n_bg:]

        x_lo, x_hi = np.percentile(bg_xy[:, 0], [1, 99])
        y_lo, y_hi = np.percentile(bg_xy[:, 1], [1, 99])
        x_pad = (x_hi - x_lo) * 0.12
        y_pad = (y_hi - y_lo) * 0.12

        fig, ax = plt.subplots(figsize=FIGSIZE_SINGLE)

        unique_bg_codes = sorted(set(bg_langs))
        unique_families = sorted({get_label(c, "family") for c in unique_bg_codes})
        family_colors = get_color_map("family", keys=unique_families)

        for code in unique_bg_codes:
            mask = np.array([l == code for l in bg_langs])
            fam = get_label(code, "family")
            ax.scatter(
                bg_xy[mask, 0], bg_xy[mask, 1],
                s=6, alpha=0.15, c=family_colors[fam], edgecolors="none",
            )

        path_colors = ["#1f77b4", "#d62728", "#2ca02c", "#ff7f0e"]
        offset = 0
        for pi, (lang_a, lang_b) in enumerate(pair_labels):
            n_pts = interp_arrays[pi].shape[0]
            pts = interp_xy[offset : offset + n_pts]
            offset += n_pts
            path_color = path_colors[pi % len(path_colors)]

            ax.plot(
                pts[:, 0], pts[:, 1],
                color=path_color, linewidth=2.5, alpha=0.8, zorder=3,
            )
            for k in range(n_pts):
                ax.scatter(
                    pts[k, 0], pts[k, 1], s=30, zorder=4,
                    facecolors="white" if 0 < k < n_pts - 1 else path_color,
                    ec=path_color, lw=1.5, alpha=0.95,
                )

            name_a = get_label(lang_a, "name")
            name_b = get_label(lang_b, "name")
            ax.annotate(
                name_a, pts[0], fontsize=9, fontweight="bold",
                xytext=(8, 8), textcoords="offset points",
                bbox=dict(boxstyle="round,pad=0.2", fc=path_color, alpha=0.2),
            )
            ax.annotate(
                name_b, pts[-1], fontsize=9, fontweight="bold",
                xytext=(8, -12), textcoords="offset points",
                bbox=dict(boxstyle="round,pad=0.2", fc=path_color, alpha=0.2),
            )
            ax.plot([], [], color=path_color, linewidth=2.5,
                    label=f"{name_a} → {name_b}")

        ax.set_xlim(x_lo - x_pad, x_hi + x_pad)
        ax.set_ylim(y_lo - y_pad, y_hi + y_pad)
        var = pca.explained_variance_ratio_
        ax.set_xlabel(f"PC1 ({var[0]*100:.1f}% variance)", fontsize=11)
        ax.set_ylabel(f"PC2 ({var[1]*100:.1f}% variance)", fontsize=11)
        ax.set_title("Latent Interpolation (PCA — preserves global geometry)", fontsize=13)
        ax.legend(loc="upper right", frameon=True, framealpha=0.9, fontsize=10)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        return fig

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

        # Figure 2: PCA trajectories (preserves global geometry)
        fig_pca = self._fig_pca_trajectories(
            z, languages, interp_arrays, valid_pairs
        )

        # Figure 3: decoded text table
        fig_text = self._fig_decoded_table(interp_arrays, valid_pairs, data)

        return [fig_tsne, fig_pca, fig_text]

    def save(
        self, figures: list[Figure], suffixes: list[str] | None = None
    ) -> list:
        """Save with descriptive suffixes."""
        return super().save(
            figures, suffixes=["trajectories", "pca_trajectories", "decoded"],
        )

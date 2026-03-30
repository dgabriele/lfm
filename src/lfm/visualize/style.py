"""Shared matplotlib style constants for publication-quality figures."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

# --------------------------------------------------------------------------
# Color palettes
# --------------------------------------------------------------------------

# 16 distinct colors for individual languages (tab20 subset)
def _build_color_map(keys: list[str]) -> dict[str, str]:
    """Generate maximally distinct colors for a list of keys.

    Uses evenly spaced hues in HSV with varied saturation/value
    to maximize perceptual distinctness regardless of key count.
    """
    import colorsys

    n = max(len(keys), 1)
    colors: dict[str, str] = {}
    for i, k in enumerate(sorted(keys)):
        hue = i / n
        # Alternate saturation and value for adjacent hues
        sat = 0.75 if i % 2 == 0 else 0.55
        val = 0.85 if i % 3 != 2 else 0.65
        r, g, b = colorsys.hsv_to_rgb(hue, sat, val)
        colors[k] = f"#{int(r*255):02x}{int(g*255):02x}{int(b*255):02x}"
    return colors


# Pre-built color maps — extended dynamically by get_color_map()
LANG_COLORS: dict[str, str] = {}
FAMILY_COLORS: dict[str, str] = {}
MORPH_COLORS: dict[str, str] = {}


def get_color_map(by: str = "language", keys: list[str] | None = None) -> dict[str, str]:
    """Return a color map for the given grouping.

    Dynamically generates colors for any set of keys. Caches results
    for consistency across calls.

    Args:
        by: "language", "family", or "type".
        keys: If provided, ensures all keys have colors assigned.
    """
    global LANG_COLORS, FAMILY_COLORS, MORPH_COLORS

    if by == "family":
        target = FAMILY_COLORS
    elif by == "type":
        target = MORPH_COLORS
    else:
        target = LANG_COLORS

    if keys:
        missing = [k for k in keys if k not in target]
        if missing:
            all_keys = sorted(set(list(target.keys()) + keys))
            target.update(_build_color_map(all_keys))

    # If empty, populate from languages.py metadata
    if not target:
        from lfm.visualize.languages import LANGUAGES

        if by == "language":
            all_keys = sorted(LANGUAGES.keys())
        elif by == "family":
            all_keys = sorted({l.family for l in LANGUAGES.values()})
        elif by == "type":
            all_keys = sorted({l.morph_type for l in LANGUAGES.values()})
        else:
            all_keys = []
        target.update(_build_color_map(all_keys, palette))

    return target


# --------------------------------------------------------------------------
# Figure style
# --------------------------------------------------------------------------

FIGSIZE_SINGLE = (8, 6)
FIGSIZE_WIDE = (12, 6)
FIGSIZE_TALL = (8, 10)
FIGSIZE_GRID = (16, 12)

FONT_SIZE_TITLE = 14
FONT_SIZE_LABEL = 12
FONT_SIZE_TICK = 10
FONT_SIZE_LEGEND = 9

SCATTER_SIZE = 3
SCATTER_ALPHA = 0.45


def apply_style() -> None:
    """Apply publication-quality matplotlib style."""
    plt.rcParams.update({
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "axes.grid": True,
        "axes.grid.which": "major",
        "grid.alpha": 0.3,
        "grid.linewidth": 0.5,
        "font.size": FONT_SIZE_TICK,
        "axes.titlesize": FONT_SIZE_TITLE,
        "axes.labelsize": FONT_SIZE_LABEL,
        "xtick.labelsize": FONT_SIZE_TICK,
        "ytick.labelsize": FONT_SIZE_TICK,
        "legend.fontsize": FONT_SIZE_LEGEND,
        "figure.dpi": 150,
        "savefig.dpi": 150,
        "savefig.bbox": "tight",
        "savefig.facecolor": "white",
    })


def language_legend(ax, by: str = "language", **kwargs) -> None:
    """Add a compact language legend to an axes."""
    from lfm.visualize.languages import LANGUAGES, get_label

    cmap = get_color_map(by)
    seen: set[str] = set()
    handles = []

    if by == "language":
        for code in sorted(LANGUAGES.keys()):
            label = get_label(code, by)
            if label not in seen:
                seen.add(label)
                handles.append(
                    plt.Line2D(
                        [0], [0],
                        marker="o",
                        color="w",
                        markerfacecolor=cmap.get(code, "#333"),
                        markersize=6,
                        label=label,
                    )
                )
    else:
        for label, color in sorted(cmap.items()):
            handles.append(
                plt.Line2D(
                    [0], [0],
                    marker="o",
                    color="w",
                    markerfacecolor=color,
                    markersize=6,
                    label=label,
                )
            )

    defaults = dict(
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
        frameon=False,
        ncol=1,
    )
    defaults.update(kwargs)
    ax.legend(handles=handles, **defaults)

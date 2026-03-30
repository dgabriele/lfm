"""Shared matplotlib style constants for publication-quality figures."""

from __future__ import annotations

import colorsys
from functools import lru_cache
from typing import FrozenSet

import matplotlib.pyplot as plt
import numpy as np

# --------------------------------------------------------------------------
# Color generation — fully dynamic, data-driven, zero hardcoded languages
# --------------------------------------------------------------------------

# Session-level caches: mapping (by, frozenset(keys)) -> color dict.
# This ensures consistency across calls within the same session while
# adapting to whatever keys are actually present in the data.
_color_cache: dict[tuple[str, FrozenSet[str]], dict[str, str]] = {}


def _generate_colors(keys: list[str]) -> dict[str, str]:
    """Generate maximally distinct colors for a sorted list of keys.

    Uses evenly spaced hues in HSV with alternating saturation/value
    to maximize perceptual separation regardless of key count.
    Golden-ratio hue spacing avoids clustering when key counts are small.
    """
    n = max(len(keys), 1)
    colors: dict[str, str] = {}
    # Golden angle provides better perceptual spread than linear spacing
    golden_ratio = (1 + 5**0.5) / 2
    for i, k in enumerate(sorted(keys)):
        # Golden-angle spacing wraps around [0,1) without clustering
        hue = (i / golden_ratio) % 1.0
        # Alternate saturation and value for adjacent items
        sat = 0.80 if i % 2 == 0 else 0.55
        val = 0.85 if i % 3 != 2 else 0.65
        r, g, b = colorsys.hsv_to_rgb(hue, sat, val)
        colors[k] = f"#{int(r * 255):02x}{int(g * 255):02x}{int(b * 255):02x}"
    return colors


def get_color_map(by: str = "language", keys: list[str] | None = None) -> dict[str, str]:
    """Return a color map for the given grouping.

    Generates colors dynamically from the provided keys. Caches results
    for consistency across calls within the same session.

    Args:
        by: Grouping type ("language", "family", or "type").
            Used only as a cache namespace.
        keys: The actual data keys to generate colors for.
              MUST be provided — colors are never generated from
              hardcoded language lists.

    Returns:
        Dict mapping each key to a hex color string.

    Raises:
        ValueError: If keys is None or empty.
    """
    if not keys:
        raise ValueError(
            f"get_color_map(by={by!r}) requires explicit keys from the data. "
            "Colors must be generated from actual data, not hardcoded lists."
        )

    cache_key = (by, frozenset(keys))
    if cache_key in _color_cache:
        return _color_cache[cache_key]

    colors = _generate_colors(sorted(set(keys)))
    _color_cache[cache_key] = colors
    return colors


# --------------------------------------------------------------------------
# Figure style
# --------------------------------------------------------------------------

FIGSIZE_SINGLE = (8, 8)  # Square for scatter plots
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


def data_legend(
    ax: plt.Axes,
    keys: list[str],
    colors: dict[str, str],
    labels: dict[str, str] | None = None,
    **kwargs,
) -> None:
    """Add a compact legend showing only the given keys.

    Args:
        ax: Matplotlib axes.
        keys: Data keys to include (only these appear in the legend).
        colors: Color map (key -> hex color).
        labels: Optional display labels (key -> label). If None, keys are used.
        **kwargs: Passed to ax.legend().
    """
    handles = []
    seen: set[str] = set()
    for key in sorted(keys):
        display = labels.get(key, key) if labels else key
        if display in seen:
            continue
        seen.add(display)
        handles.append(
            plt.Line2D(
                [0], [0],
                marker="o",
                color="w",
                markerfacecolor=colors.get(key, "#333333"),
                markersize=6,
                label=display,
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

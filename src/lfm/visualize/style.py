"""Shared matplotlib style constants for publication-quality figures."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

# --------------------------------------------------------------------------
# Color palettes
# --------------------------------------------------------------------------

# 16 distinct colors for individual languages (tab20 subset)
LANG_COLORS: dict[str, str] = {
    "ara": "#1f77b4",
    "ces": "#ff7f0e",
    "deu": "#2ca02c",
    "eng": "#d62728",
    "est": "#9467bd",
    "fin": "#8c564b",
    "hin": "#e377c2",
    "hun": "#7f7f7f",
    "ind": "#bcbd22",
    "kor": "#17becf",
    "pol": "#aec7e8",
    "por": "#ffbb78",
    "rus": "#98df8a",
    "spa": "#ff9896",
    "tur": "#c5b0d5",
    "vie": "#c49c94",
}

# Family colors (fewer groups, bolder)
FAMILY_COLORS: dict[str, str] = {
    "Afro-Asiatic": "#e41a1c",
    "Austroasiatic": "#377eb8",
    "Austronesian": "#4daf4a",
    "Indo-European": "#984ea3",
    "Koreanic": "#ff7f00",
    "Turkic": "#a65628",
    "Uralic": "#f781bf",
}

# Morphological type colors
MORPH_COLORS: dict[str, str] = {
    "agglutinative": "#1b9e77",
    "fusional": "#d95f02",
    "introflexive": "#7570b3",
    "isolating": "#e7298a",
}


def get_color_map(by: str = "language") -> dict[str, str]:
    """Return the color map for the given grouping."""
    if by == "family":
        return FAMILY_COLORS
    if by == "type":
        return MORPH_COLORS
    return LANG_COLORS


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

SCATTER_SIZE = 8
SCATTER_ALPHA = 0.6


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

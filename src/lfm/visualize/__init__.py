"""Visualization modules for the LFM multilingual VAE.

Each visualization module provides a class that inherits from
``BaseVisualization`` and implements ``generate()``.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.figure import Figure

from lfm.visualize.config import VisualizeConfig


class BaseVisualization(ABC):
    """Abstract base for all visualization types."""

    def __init__(self, config: VisualizeConfig) -> None:
        self.config = config

    @property
    @abstractmethod
    def name(self) -> str:
        """Short identifier used in filenames."""

    @abstractmethod
    def generate(self, data: dict) -> list[Figure]:
        """Generate figures from precomputed data.

        Args:
            data: Shared data dict from the loader (z vectors, labels, etc.).

        Returns:
            List of matplotlib Figures.
        """

    def save(self, figures: list[Figure], suffixes: list[str] | None = None) -> list[Path]:
        """Save figures to output directory.

        Args:
            figures: Matplotlib figures to save.
            suffixes: Optional per-figure name suffixes.

        Returns:
            List of saved file paths.
        """
        cfg = self.config
        out_dir = Path(cfg.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        paths: list[Path] = []
        for i, fig in enumerate(figures):
            suffix = f"_{suffixes[i]}" if suffixes and i < len(suffixes) else f"_{i + 1}"
            fname = f"{self.name}{suffix}.{cfg.format}"
            path = out_dir / fname
            fig.savefig(path, dpi=cfg.dpi, bbox_inches="tight", facecolor="white")
            plt.close(fig)
            paths.append(path)

        return paths

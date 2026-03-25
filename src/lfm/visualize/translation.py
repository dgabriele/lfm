"""Translation evaluation visualizations.

Reads saved results from a translator output directory (no model loading).
Produces three figures:
1. Semantic similarity histogram
2. BLEU breakdown bar chart
3. Example translations table
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure

from lfm.visualize.style import (
    FIGSIZE_SINGLE,
    FONT_SIZE_LABEL,
    FONT_SIZE_TITLE,
    apply_style,
)

logger = logging.getLogger(__name__)


class TranslationVisualization:
    """Generate translation evaluation visualizations from saved results.

    Args:
        results_dir: Directory containing ``results.json`` and
            ``translations.jsonl`` from a translator evaluation run.
        output_dir: Where to save figures (default: ``results_dir``).
        fmt: Output format (``png``, ``svg``, ``pdf``).
        dpi: Output resolution.
    """

    def __init__(
        self,
        results_dir: str | Path,
        output_dir: str | Path | None = None,
        fmt: str = "png",
        dpi: int = 150,
    ) -> None:
        self.results_dir = Path(results_dir)
        self.output_dir = Path(output_dir) if output_dir else self.results_dir
        self.fmt = fmt
        self.dpi = dpi

    def generate_all(self) -> list[Path]:
        """Generate all translation visualizations.

        Returns:
            List of saved file paths.
        """
        apply_style()
        paths: list[Path] = []

        results = self._load_results()
        translations = self._load_translations()

        if not results:
            logger.warning("No results.json found in %s", self.results_dir)
            return paths

        # 1. Semantic similarity histogram
        if translations and "cosine_similarity" in translations[0]:
            fig = self._similarity_histogram(translations)
            p = self._save(fig, "similarity_histogram")
            paths.append(p)

        # 2. BLEU breakdown
        fig = self._bleu_breakdown(results)
        p = self._save(fig, "bleu_breakdown")
        paths.append(p)

        # 3. Example translations table
        if translations:
            fig = self._examples_table(translations)
            p = self._save(fig, "translation_examples")
            paths.append(p)

        return paths

    def _similarity_histogram(self, translations: list[dict]) -> Figure:
        """Histogram of per-example cosine similarity."""
        sims = np.array([t["cosine_similarity"] for t in translations])
        mean_sim = sims.mean()
        median_sim = np.median(sims)

        fig, ax = plt.subplots(figsize=FIGSIZE_SINGLE)
        ax.hist(sims, bins=30, color="#4C72B0", alpha=0.8, edgecolor="white")
        ax.axvline(mean_sim, color="#C44E52", linestyle="--", linewidth=2,
                    label=f"Mean: {mean_sim:.3f}")
        ax.axvline(median_sim, color="#DD8452", linestyle=":", linewidth=2,
                    label=f"Median: {median_sim:.3f}")
        ax.set_xlabel("Cosine Similarity (reference vs hypothesis)", fontsize=FONT_SIZE_LABEL)
        ax.set_ylabel("Count", fontsize=FONT_SIZE_LABEL)
        ax.set_title("Semantic Similarity Distribution", fontsize=FONT_SIZE_TITLE)
        ax.legend(fontsize=10)
        fig.tight_layout()
        return fig

    def _bleu_breakdown(self, results: dict) -> Figure:
        """Grouped bar chart of BLEU-1/2/3/4 + geometric mean."""
        labels = ["BLEU-1", "BLEU-2", "BLEU-3", "BLEU-4", "BLEU-4\n(geometric)"]
        keys = ["bleu_1", "bleu_2", "bleu_3", "bleu_4", "bleu_4_geometric"]
        values = [results.get(k, 0.0) for k in keys]

        fig, ax = plt.subplots(figsize=FIGSIZE_SINGLE)
        colors = ["#4C72B0", "#55A868", "#C44E52", "#8172B2", "#CCB974"]
        bars = ax.bar(labels, values, color=colors, edgecolor="white", width=0.6)

        for bar, val in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                f"{val:.3f}", ha="center", va="bottom", fontsize=10,
            )

        ax.set_ylabel("Score", fontsize=FONT_SIZE_LABEL)
        ax.set_title("BLEU Score Breakdown", fontsize=FONT_SIZE_TITLE)
        ax.set_ylim(0, max(values) * 1.2 if max(values) > 0 else 1.0)
        fig.tight_layout()
        return fig

    def _examples_table(self, translations: list[dict], n: int = 10) -> Figure:
        """Table of top examples showing IPA -> reference -> hypothesis."""
        # Sort by cosine similarity (best first) if available
        if "cosine_similarity" in translations[0]:
            sorted_trans = sorted(
                translations, key=lambda t: t["cosine_similarity"], reverse=True
            )
        else:
            sorted_trans = translations
        examples = sorted_trans[:n]

        fig, ax = plt.subplots(figsize=(16, max(4, n * 0.8)))
        ax.axis("off")

        # Build table data
        headers = ["#", "IPA (truncated)", "Reference", "Hypothesis", "Sim"]
        cell_text = []
        cell_colors = []

        for i, ex in enumerate(examples):
            sim = ex.get("cosine_similarity", 0.0)
            # Color-code by similarity
            if sim >= 0.7:
                row_color = "#d4edda"  # green
            elif sim >= 0.4:
                row_color = "#fff3cd"  # yellow
            else:
                row_color = "#f8d7da"  # red

            cell_text.append([
                str(i + 1),
                ex.get("ipa", "")[:50],
                ex.get("reference", "")[:60],
                ex.get("hypothesis", "")[:60],
                f"{sim:.3f}",
            ])
            cell_colors.append([row_color] * 5)

        table = ax.table(
            cellText=cell_text,
            colLabels=headers,
            cellColours=cell_colors,
            loc="center",
            cellLoc="left",
        )
        table.auto_set_font_size(False)
        table.set_fontsize(8)
        table.scale(1.0, 1.5)

        # Style header row
        for j in range(len(headers)):
            table[0, j].set_facecolor("#343a40")
            table[0, j].set_text_props(color="white", fontweight="bold")

        ax.set_title(
            "Top Translation Examples (by similarity)",
            fontsize=FONT_SIZE_TITLE, pad=20,
        )
        fig.tight_layout()
        return fig

    def _load_results(self) -> dict:
        """Load results.json from results_dir."""
        path = self.results_dir / "results.json"
        if not path.exists():
            return {}
        with open(path) as f:
            return json.load(f)

    def _load_translations(self) -> list[dict]:
        """Load translations.jsonl from results_dir."""
        path = self.results_dir / "translations.jsonl"
        if not path.exists():
            return []
        translations = []
        with open(path, encoding="utf-8") as f:
            for line in f:
                translations.append(json.loads(line))
        return translations

    def _save(self, fig: Figure, name: str) -> Path:
        """Save figure and close."""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        path = self.output_dir / f"{name}.{self.fmt}"
        fig.savefig(path, dpi=self.dpi, bbox_inches="tight")
        plt.close(fig)
        logger.info("Saved: %s", path)
        return path

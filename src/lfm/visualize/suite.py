"""Visualization suite — run all visualizations with shared data."""

from __future__ import annotations

import logging
from pathlib import Path

from lfm.visualize.config import VisualizeConfig

logger = logging.getLogger(__name__)


class VisualizationSuite:
    """Orchestrates running all visualization types with shared computation."""

    def __init__(self, config: VisualizeConfig) -> None:
        self.config = config

    def run_all(
        self,
        model_data: dict,
        labeled_data: dict,
        corpus_data: dict,
    ) -> list[Path]:
        """Run all visualizations in dependency order.

        Args:
            model_data: From ``load_checkpoint()``.
            labeled_data: From ``encode_labeled_corpus()`` (has languages).
            corpus_data: From ``encode_corpus()`` (full corpus, may lack labels).

        Returns:
            List of all saved file paths.
        """
        all_paths: list[Path] = []

        # Merge model_data into corpus/labeled dicts for visualizations
        # that need decoder access
        full_labeled = {**labeled_data, **model_data}
        full_corpus = {**corpus_data, **model_data}

        # Phase 1: Labeled visualizations (need per-sentence language codes)
        from lfm.visualize.tsne import TSNEVisualization

        try:
            logger.info("Running: t-SNE")
            viz = TSNEVisualization(self.config)
            figs = viz.generate(labeled_data)
            all_paths.extend(viz.save(figs, ["by_language", "by_family", "by_type"]))
        except Exception:
            logger.exception("t-SNE failed")

        from lfm.visualize.clustering import ClusteringVisualization

        try:
            logger.info("Running: clustering")
            viz = ClusteringVisualization(self.config)
            figs = viz.generate(labeled_data)
            all_paths.extend(viz.save(figs, ["dendrogram", "heatmap"]))
        except Exception:
            logger.exception("clustering failed")

        from lfm.visualize.latent_dims import LatentDimsVisualization

        try:
            logger.info("Running: latent-dims")
            viz = LatentDimsVisualization(self.config)
            figs = viz.generate(labeled_data)
            all_paths.extend(
                viz.save(figs, ["variance", "lang_heatmap", "pca", "f_statistic"])
            )
        except Exception:
            logger.exception("latent-dims failed")

        # Phase 2: Corpus-wide visualizations (don't strictly need labels)
        from lfm.visualize.zipf import ZipfVisualization

        try:
            logger.info("Running: zipf")
            viz = ZipfVisualization(self.config)
            figs = viz.generate(full_corpus)
            all_paths.extend(viz.save(figs, ["rank_frequency", "exponent_comparison"]))
        except Exception:
            logger.exception("zipf failed")

        from lfm.visualize.length_dist import LengthDistVisualization

        try:
            logger.info("Running: length-dist")
            viz = LengthDistVisualization(self.config)
            figs = viz.generate(full_corpus)
            all_paths.extend(viz.save(figs, ["histogram", "by_language", "vs_znorm"]))
        except Exception:
            logger.exception("length-dist failed")

        # Phase 3: Attention (needs decoder)
        from lfm.visualize.attention import AttentionVisualization

        try:
            logger.info("Running: attention")
            viz = AttentionVisualization(self.config)
            figs = viz.generate(full_corpus)
            all_paths.extend(viz.save(figs, ["per_head", "average", "entropy"]))
        except Exception:
            logger.exception("attention failed")

        # Phase 4: Interpolation (needs labels + decoder)
        from lfm.visualize.interpolation import InterpolationVisualization

        try:
            logger.info("Running: interpolation")
            viz = InterpolationVisualization(self.config)
            figs = viz.generate(full_labeled)
            all_paths.extend(viz.save(figs, ["trajectories", "decoded_text"]))
        except Exception:
            logger.exception("interpolation failed")

        # Phase 5: Structural claims (compositionality, smoothness, adaptiveness)
        from lfm.visualize.compositionality import CompositionalityVisualization

        try:
            logger.info("Running: compositionality")
            viz = CompositionalityVisualization(self.config)
            figs = viz.generate(full_corpus)
            all_paths.extend(viz.save(figs, ["heatmap", "scores", "mutual_info"]))
        except Exception:
            logger.exception("compositionality failed")

        from lfm.visualize.smoothness import SmoothnessVisualization

        try:
            logger.info("Running: smoothness")
            viz = SmoothnessVisualization(self.config)
            figs = viz.generate(full_corpus)
            all_paths.extend(
                viz.save(figs, ["lipschitz", "jaccard", "interpolation_continuity"])
            )
        except Exception:
            logger.exception("smoothness failed")

        from lfm.visualize.adaptiveness import AdaptivenessVisualization

        try:
            logger.info("Running: adaptiveness")
            viz = AdaptivenessVisualization(self.config)
            figs = viz.generate(full_corpus)
            all_paths.extend(
                viz.save(figs, ["length_adaptation", "diversity", "complexity_profile"])
            )
        except Exception:
            logger.exception("adaptiveness failed")

        return all_paths

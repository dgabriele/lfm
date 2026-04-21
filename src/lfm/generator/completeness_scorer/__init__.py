"""Completeness scorer — judges whether a token sequence forms a complete thought.

Vocabulary-agnostic: scores structural completeness from word order and
positional patterns, not word identity. Trained on real sentences (positive)
and structurally corrupted versions (negative), including variants with
alien/random content words as positives. This ensures the scorer
generalizes to emergent language where ALL words may be novel.

Usage during VAE training:
    decode z → soft token logits → scorer → completeness score → aux loss
"""

from lfm.generator.completeness_scorer.model import CompletenessScorer
from lfm.generator.completeness_scorer.data import build_scorer_dataset

__all__ = ["CompletenessScorer", "build_scorer_dataset"]

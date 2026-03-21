"""Expressivity metrics.

Metrics for evaluating the expressive capacity of the emergent language,
including morpheme segmentation diversity and effective vocabulary size.
"""

from __future__ import annotations

from torch import Tensor

from lfm.metrics.base import Metric


class UniqueSegmentationsMetric(Metric):
    """Number of distinct morpheme segmentation patterns produced.

    Measures the diversity of structural expression -- more unique
    segmentations indicate the language has more ways to compose
    its morphemes.
    """

    def __init__(self) -> None:
        super().__init__("unique_segmentations")

    def compute(self, outputs: dict[str, Tensor]) -> float:
        """Count unique segmentation patterns in a batch.

        Args:
            outputs: Pipeline output dictionary containing morphology outputs.

        Returns:
            Number of distinct segmentations (as a float for API consistency).
        """
        raise NotImplementedError("UniqueSegmentationsMetric.compute() not yet implemented")


class VocabularySizeMetric(Metric):
    """Effective vocabulary size (unique tokens used).

    Counts the number of unique token indices used across a batch.
    Compared to the total codebook size, this indicates how much of
    the available vocabulary the system actually uses.
    """

    def __init__(self) -> None:
        super().__init__("vocabulary_size")

    def compute(self, outputs: dict[str, Tensor]) -> float:
        """Count unique tokens used in a batch.

        Args:
            outputs: Pipeline output dictionary containing quantization
                outputs with a ``tokens`` key.

        Returns:
            Number of unique tokens (as a float for API consistency).
        """
        raise NotImplementedError("VocabularySizeMetric.compute() not yet implemented")

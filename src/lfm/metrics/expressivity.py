"""Expressivity metrics.

Metrics for evaluating the expressive capacity of the emergent language,
including structural diversity and effective vocabulary size.
"""

from __future__ import annotations

from torch import Tensor

from lfm.metrics.base import Metric


class UniqueStructuresMetric(Metric):
    """Number of distinct structural patterns produced.

    Counts the number of unique parse tree structures induced across
    a batch of inputs.  Higher counts indicate greater structural
    diversity in the emergent language.
    """

    def __init__(self) -> None:
        super().__init__("unique_structures")

    def compute(self, outputs: dict[str, Tensor]) -> float:
        """Count unique structural patterns in a batch.

        Args:
            outputs: Pipeline output dictionary containing syntax outputs.

        Returns:
            Number of distinct structures (as a float for API consistency).
        """
        raise NotImplementedError("UniqueStructuresMetric.compute() not yet implemented")


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

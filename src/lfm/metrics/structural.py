"""Structural metrics.

Metrics for evaluating the morphological structure of emergent language,
including morpheme nesting depth, morpheme variety, agreement consistency,
and ordering regularity.
"""

from __future__ import annotations

from torch import Tensor

from lfm.metrics.base import Metric


class MorphemeNestingDepth(Metric):
    """Average nesting depth of morpheme hierarchy per token.

    Measures how deeply nested the morphological structure is -- tokens
    with more morpheme levels indicate richer morphological composition.
    """

    def __init__(self) -> None:
        super().__init__("morpheme_nesting_depth")

    def compute(self, outputs: dict[str, Tensor]) -> float:
        """Compute average morpheme nesting depth for a batch.

        Args:
            outputs: Pipeline output dictionary containing morphology outputs.

        Returns:
            Scalar average nesting depth.
        """
        raise NotImplementedError("MorphemeNestingDepth.compute() not yet implemented")


class MorphemeVariety(Metric):
    """Diversity of the morpheme inventory.

    Measures the effective vocabulary size of morphemes -- higher variety
    indicates a richer morphological system with more distinct units.
    """

    def __init__(self) -> None:
        super().__init__("morpheme_variety")

    def compute(self, outputs: dict[str, Tensor]) -> float:
        """Compute morpheme variety for a batch.

        Args:
            outputs: Pipeline output dictionary containing morphology outputs.

        Returns:
            Scalar morpheme variety value.
        """
        raise NotImplementedError("MorphemeVariety.compute() not yet implemented")


class AgreementConsistencyMetric(Metric):
    """Measures how consistently morphological features agree at related positions.

    High consistency indicates the language has developed reliable
    agreement patterns (like case or number agreement).
    """

    def __init__(self) -> None:
        super().__init__("agreement_consistency")

    def compute(self, outputs: dict[str, Tensor]) -> float:
        """Compute agreement consistency for a batch.

        Args:
            outputs: Pipeline output dictionary containing syntax and
                morphology outputs.

        Returns:
            Scalar agreement consistency value.
        """
        raise NotImplementedError("AgreementConsistencyMetric.compute() not yet implemented")


class OrderingRegularityMetric(Metric):
    """Measures consistency of token ordering across different inputs.

    High regularity indicates stable word-order patterns have emerged.
    """

    def __init__(self) -> None:
        super().__init__("ordering_regularity")

    def compute(self, outputs: dict[str, Tensor]) -> float:
        """Compute ordering regularity for a batch.

        Args:
            outputs: Pipeline output dictionary containing syntax outputs.

        Returns:
            Scalar ordering regularity value.
        """
        raise NotImplementedError("OrderingRegularityMetric.compute() not yet implemented")

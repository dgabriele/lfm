"""Structural metrics.

Metrics for evaluating the structural properties of induced parse trees,
including depth, branching factor, and consistency.
"""

from __future__ import annotations

from torch import Tensor

from lfm.metrics.base import Metric


class TreeDepthMetric(Metric):
    """Average depth of induced parse trees.

    Computes the mean depth of parse trees induced by the syntax module
    across a batch.  Deeper trees indicate more hierarchical structure.
    """

    def __init__(self) -> None:
        super().__init__("tree_depth")

    def compute(self, outputs: dict[str, Tensor]) -> float:
        """Compute average tree depth for a batch.

        Args:
            outputs: Pipeline output dictionary containing syntax outputs
                with a ``depth`` key.

        Returns:
            Scalar average tree depth.
        """
        raise NotImplementedError("TreeDepthMetric.compute() not yet implemented")


class BranchingFactorMetric(Metric):
    """Average branching factor of induced parse trees.

    Computes the mean branching factor (average number of children per
    internal node) of the induced parse trees.
    """

    def __init__(self) -> None:
        super().__init__("branching_factor")

    def compute(self, outputs: dict[str, Tensor]) -> float:
        """Compute average branching factor for a batch.

        Args:
            outputs: Pipeline output dictionary containing syntax outputs.

        Returns:
            Scalar average branching factor.
        """
        raise NotImplementedError("BranchingFactorMetric.compute() not yet implemented")


class StructuralConsistencyMetric(Metric):
    """Consistency of structural patterns across different inputs.

    Measures how consistently the model applies the same structural
    patterns to inputs that share relevant features, indicating whether
    the induced grammar is systematic rather than arbitrary.
    """

    def __init__(self) -> None:
        super().__init__("structural_consistency")

    def compute(self, outputs: dict[str, Tensor]) -> float:
        """Compute structural consistency for a batch.

        Args:
            outputs: Pipeline output dictionary containing syntax outputs.

        Returns:
            Scalar structural consistency value.
        """
        raise NotImplementedError("StructuralConsistencyMetric.compute() not yet implemented")

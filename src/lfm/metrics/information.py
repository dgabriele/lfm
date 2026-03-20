"""Information-theoretic metrics.

Metrics for evaluating codebook utilization and mutual information
between agent states and messages.
"""

from __future__ import annotations

from torch import Tensor

from lfm.metrics.base import Metric


class CodebookUtilizationMetric(Metric):
    """Fraction of codebook entries actively used.

    Computes the proportion of codebook entries that are selected at
    least once in the current batch.  Low utilization indicates codebook
    collapse, where the quantizer ignores most available codes.
    """

    def __init__(self) -> None:
        super().__init__("codebook_utilization")

    def compute(self, outputs: dict[str, Tensor]) -> float:
        """Compute codebook utilization for a batch.

        Args:
            outputs: Pipeline output dictionary containing quantization
                outputs with a ``tokens`` key.

        Returns:
            Scalar utilization fraction in [0, 1].
        """
        raise NotImplementedError("CodebookUtilizationMetric.compute() not yet implemented")


class MutualInformationMetric(Metric):
    """Mutual information between agent states and messages.

    Estimates the mutual information I(S; M) between agent internal
    states S and the messages M produced by the LFM pipeline.  Higher
    mutual information indicates that messages carry more information
    about the agent's internal state.
    """

    def __init__(self) -> None:
        super().__init__("mutual_information")

    def compute(self, outputs: dict[str, Tensor]) -> float:
        """Estimate mutual information for a batch.

        Args:
            outputs: Pipeline output dictionary containing agent states
                and message representations.

        Returns:
            Scalar mutual information estimate.
        """
        raise NotImplementedError("MutualInformationMetric.compute() not yet implemented")

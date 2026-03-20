"""Compositionality metrics.

Metrics for measuring the compositional structure of emergent language,
including topographic similarity, positional disentanglement, and
bag-of-symbols disentanglement.
"""

from __future__ import annotations

from torch import Tensor

from lfm.metrics.base import Metric


class TopographicSimilarity(Metric):
    """Measures correlation between meaning space and message space distances.

    Computes the Spearman correlation between pairwise distances in the
    agent's meaning space (input representations) and pairwise distances
    in the message space (output token sequences).  A high topographic
    similarity indicates that similar meanings are expressed with similar
    messages, a hallmark of compositional language.
    """

    def __init__(self) -> None:
        super().__init__("topographic_similarity")

    def compute(self, outputs: dict[str, Tensor]) -> float:
        """Compute topographic similarity for a batch.

        Args:
            outputs: Pipeline output dictionary containing agent states
                and message representations.

        Returns:
            Scalar topographic similarity value.
        """
        raise NotImplementedError("TopographicSimilarity.compute() not yet implemented")


class PositionalDisentanglement(Metric):
    """PosDis: measures positional disentanglement of emergent language.

    Evaluates whether each position in the message sequence encodes
    information about a single, distinct dimension of the input meaning
    space.  High positional disentanglement indicates that the emergent
    language uses a systematic, position-dependent encoding.
    """

    def __init__(self) -> None:
        super().__init__("positional_disentanglement")

    def compute(self, outputs: dict[str, Tensor]) -> float:
        """Compute positional disentanglement for a batch.

        Args:
            outputs: Pipeline output dictionary.

        Returns:
            Scalar positional disentanglement value.
        """
        raise NotImplementedError("PositionalDisentanglement.compute() not yet implemented")


class BagOfSymbolsDisentanglement(Metric):
    """BosDis: bag-of-symbols disentanglement metric.

    Similar to positional disentanglement but ignores position, measuring
    whether the set of symbols used (regardless of order) systematically
    encodes distinct meaning dimensions.
    """

    def __init__(self) -> None:
        super().__init__("bag_of_symbols_disentanglement")

    def compute(self, outputs: dict[str, Tensor]) -> float:
        """Compute bag-of-symbols disentanglement for a batch.

        Args:
            outputs: Pipeline output dictionary.

        Returns:
            Scalar bag-of-symbols disentanglement value.
        """
        raise NotImplementedError("BagOfSymbolsDisentanglement.compute() not yet implemented")

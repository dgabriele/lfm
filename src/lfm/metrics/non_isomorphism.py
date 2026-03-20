"""Non-isomorphism metrics.

Metrics for measuring how the emergent language diverges from human
language structure, using Representational Similarity Analysis (RSA).
"""

from __future__ import annotations

from torch import Tensor

from lfm.metrics.base import Metric


class RSADivergenceMetric(Metric):
    """Representational Similarity Analysis divergence from human language.

    Computes the divergence between the representational geometry of
    the emergent language and that of a reference human language model.
    This measures how structurally different the emergent language is
    from human language, which is expected to be non-zero since LFM
    does not aim to replicate human language structure.
    """

    def __init__(self) -> None:
        super().__init__("rsa_divergence")

    def compute(self, outputs: dict[str, Tensor]) -> float:
        """Compute RSA divergence for a batch.

        Args:
            outputs: Pipeline output dictionary containing message
                representations and optional reference embeddings.

        Returns:
            Scalar RSA divergence value.
        """
        raise NotImplementedError("RSADivergenceMetric.compute() not yet implemented")

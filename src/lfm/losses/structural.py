"""Morphological structure losses.

Loss functions that reward well-formed morphological structure, consistent
agreement patterns, regular ordering, and distinct case marking across
grammatical roles.
"""

from __future__ import annotations

from torch import Tensor

from lfm._registry import register
from lfm.core.loss import LFMLoss


@register("loss", "agreement_consistency")
class AgreementConsistencyLoss(LFMLoss):
    """Penalizes inconsistency in morphological agreement between related positions.

    If the syntax module identifies positions that should agree (high
    agreement_scores), their grammatical features should be similar.
    Conversely, positions with low agreement scores should have distinct
    features.

    Reads from outputs:
        syntax.agreement_scores: (batch, seq_len, seq_len)
        morphology.grammatical_features: (batch, seq_len, num_features)

    Args:
        config: Optional loss configuration.
        weight: Multiplicative weight for this loss term.
    """

    def __init__(self, config: object = None, weight: float = 1.0) -> None:
        super().__init__(config, weight)

    def forward(
        self,
        outputs: dict[str, Tensor],
        targets: dict[str, Tensor] | None = None,
    ) -> Tensor:
        """Compute agreement consistency loss.

        Args:
            outputs: Combined pipeline output dictionary.
            targets: Optional ground-truth tensors (unused).

        Returns:
            Scalar loss tensor.
        """
        raise NotImplementedError("AgreementConsistencyLoss.forward() not yet implemented")


@register("loss", "morphological_well_formedness")
class MorphologicalWellFormednessLoss(LFMLoss):
    """Rewards well-formed morpheme sequences and valid feature combinations.

    Encourages morpheme segmentations that produce consistent, reusable
    patterns.  Penalizes degenerate segmentations (all one morpheme, or
    all unique morphemes).

    Reads from outputs:
        morphology.segments: (batch, seq_len, max_morphemes)
        morphology.segment_mask: (batch, seq_len, max_morphemes)
        morphology.segment_log_probs: (batch, seq_len)
        morphology.grammatical_features: (batch, seq_len, num_features)

    Args:
        config: Optional loss configuration.
        weight: Multiplicative weight for this loss term.
    """

    def __init__(self, config: object = None, weight: float = 1.0) -> None:
        super().__init__(config, weight)

    def forward(
        self,
        outputs: dict[str, Tensor],
        targets: dict[str, Tensor] | None = None,
    ) -> Tensor:
        """Compute morphological well-formedness loss.

        Args:
            outputs: Combined pipeline output dictionary.
            targets: Optional ground-truth tensors (unused).

        Returns:
            Scalar loss tensor.
        """
        raise NotImplementedError("MorphologicalWellFormednessLoss.forward() not yet implemented")


@register("loss", "ordering_regularity")
class OrderingRegularityLoss(LFMLoss):
    """Rewards consistent information-theoretic ordering of tokens.

    Encourages the agent to develop stable ordering strategies -- e.g.,
    placing high-information tokens before low-information ones, or
    consistently ordering agent-before-patient.

    Reads from outputs:
        syntax.ordering_scores: (batch, seq_len)

    Args:
        config: Optional loss configuration.
        weight: Multiplicative weight for this loss term.
    """

    def __init__(self, config: object = None, weight: float = 1.0) -> None:
        super().__init__(config, weight)

    def forward(
        self,
        outputs: dict[str, Tensor],
        targets: dict[str, Tensor] | None = None,
    ) -> Tensor:
        """Compute ordering regularity loss.

        Args:
            outputs: Combined pipeline output dictionary.
            targets: Optional ground-truth tensors (unused).

        Returns:
            Scalar loss tensor.
        """
        raise NotImplementedError("OrderingRegularityLoss.forward() not yet implemented")


@register("loss", "case_marking_pressure")
class CaseMarkingPressureLoss(LFMLoss):
    """Encourages distinct morphological marking of different grammatical roles.

    Pushes the morphology to develop distinct feature patterns for tokens
    serving different structural functions (like case marking in
    agglutinative languages).  Without this pressure, all tokens may
    develop identical features.

    Reads from outputs:
        morphology.grammatical_features: (batch, seq_len, num_features)

    Args:
        config: Optional loss configuration.
        weight: Multiplicative weight for this loss term.
    """

    def __init__(self, config: object = None, weight: float = 1.0) -> None:
        super().__init__(config, weight)

    def forward(
        self,
        outputs: dict[str, Tensor],
        targets: dict[str, Tensor] | None = None,
    ) -> Tensor:
        """Compute case marking pressure loss.

        Args:
            outputs: Combined pipeline output dictionary.
            targets: Optional ground-truth tensors (unused).

        Returns:
            Scalar loss tensor.
        """
        raise NotImplementedError("CaseMarkingPressureLoss.forward() not yet implemented")

"""Abstract base class for syntax modules.

All syntax implementations must subclass ``SyntaxModule`` and implement the
required abstract interface.  The syntax module learns soft agreement constraints
between morphological features and information-theoretic ordering pressures.
Phrase structure emerges from these constraints rather than being prescribed.
"""

from __future__ import annotations

from abc import abstractmethod
from typing import ClassVar

from torch import Tensor

from lfm._types import GrammaticalFeatures, Mask, TokenEmbeddings
from lfm.core.module import LFMModule


class SyntaxModule(LFMModule):
    """Base class for structural agreement and ordering modules.

    Unlike traditional syntax modules that impose explicit tree structure (PCFG),
    this module learns soft agreement constraints between morphological features
    and information-theoretic ordering pressures. Phrase structure emerges from
    these constraints rather than being prescribed.
    """

    output_prefix: ClassVar[str] = "syntax"

    @abstractmethod
    def forward(
        self,
        embeddings: TokenEmbeddings,
        mask: Mask,
        grammatical_features: GrammaticalFeatures | None = None,
    ) -> dict[str, Tensor]:
        """Compute structural agreement and ordering scores.

        Args:
            embeddings: Token embeddings, shape (batch, seq_len, dim).
            mask: Boolean padding mask, shape (batch, seq_len).
            grammatical_features: Optional morphological feature vectors from
                upstream morphology module, shape (batch, seq_len, num_features).

        Returns:
            Dictionary with the following keys:

            - ``agreement_scores`` -- pairwise agreement between positions,
              shape ``(batch, seq_len, seq_len)``. High values indicate
              positions whose morphological features are consistent.
            - ``ordering_scores`` -- per-position information content scores,
              shape ``(batch, seq_len)``. Used by ordering losses to encourage
              consistent information-theoretic structure.
            - ``structural_features`` -- refined embeddings incorporating
              structural information, shape ``(batch, seq_len, dim)``.
        """
        ...

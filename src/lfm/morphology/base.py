"""Abstract base class for morphology modules.

All morphology implementations must subclass ``MorphologyModule`` and implement
the required abstract interface.  The morphology module discovers and applies
sub-token structure, segmenting tokens into morpheme-like units and recomposing
them into enriched representations.
"""

from __future__ import annotations

from abc import abstractmethod
from typing import ClassVar

from torch import Tensor

from lfm._types import Mask, TokenEmbeddings, TokenIds
from lfm.core.module import LFMModule


class MorphologyModule(LFMModule):
    """Abstract base for morphology modules.

    A morphology module segments discrete tokens into morpheme-like sub-units,
    produces segment masks, and composes morpheme representations back into
    token-level embeddings.  It also provides log-probabilities over segmentation
    decisions for use in training.

    Subclasses must implement:
        - ``forward``: full morphological analysis with segmentation and
          recomposition.
        - ``segment``: pure segmentation without recomposition.
    """

    output_prefix: ClassVar[str] = "morphology"

    @abstractmethod
    def forward(self, tokens: TokenIds, embeddings: TokenEmbeddings) -> dict[str, Tensor]:
        """Run morphological analysis on a batch of token sequences.

        Args:
            tokens: Integer token indices of shape ``(batch, seq_len)``.
            embeddings: Dense token embeddings of shape
                ``(batch, seq_len, dim)``.

        Returns:
            Dictionary with the following keys:

            - ``segments`` — morpheme indices, shape
              ``(batch, seq_len, max_morphemes)``.
            - ``segment_mask`` — boolean mask indicating valid morpheme
              positions, shape ``(batch, seq_len, max_morphemes)``.
            - ``composed`` — recomposed token embeddings, shape
              ``(batch, seq_len, dim)``.
            - ``segment_log_probs`` — log-probabilities of segmentation
              decisions, shape ``(batch, seq_len)``.
            - ``grammatical_features`` — learned latent grammatical
              categories per token (emergent case, number, tense, aspect,
              etc.), shape ``(batch, seq_len, num_features)``.
        """
        ...

    @abstractmethod
    def segment(self, tokens: TokenIds) -> tuple[Tensor, Mask]:
        """Segment tokens into morpheme units without recomposition.

        Args:
            tokens: Integer token indices of shape ``(batch, seq_len)``.

        Returns:
            A tuple of:
                - Morpheme segment indices of shape
                  ``(batch, seq_len, max_morphemes)``.
                - Boolean mask of shape ``(batch, seq_len, max_morphemes)``
                  indicating valid morpheme positions.
        """
        ...

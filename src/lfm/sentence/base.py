"""Abstract base class for sentence modules.

All sentence-level implementations must subclass ``SentenceModule`` and
implement the required abstract interface.  The sentence module classifies
sequences by sentence type (e.g. statement, question) and detects sentence
boundaries within token sequences.
"""

from __future__ import annotations

from abc import abstractmethod
from typing import ClassVar

from torch import Tensor

from lfm._types import Mask, TokenEmbeddings
from lfm.core.module import LFMModule


class SentenceModule(LFMModule):
    """Abstract base for sentence-level modules.

    A sentence module takes token embeddings and a padding mask and produces
    sentence-type classifications, boundary indicators, and type-level
    embeddings that capture sentence-level properties.

    Subclasses must implement:
        - ``forward``: full sentence-level analysis producing type
          classifications, boundary indicators, and type embeddings.
    """

    output_prefix: ClassVar[str] = "sentence"

    @abstractmethod
    def forward(self, embeddings: TokenEmbeddings, mask: Mask) -> dict[str, Tensor]:
        """Run sentence-level analysis on a batch of token sequences.

        Args:
            embeddings: Dense token embeddings of shape
                ``(batch, seq_len, dim)``.
            mask: Boolean padding mask of shape ``(batch, seq_len)``, where
                ``True`` indicates a valid (non-padding) position.

        Returns:
            Dictionary with the following keys:

            - ``sentence_type`` — predicted sentence type logits or
              probabilities, shape ``(batch, num_sentence_types)``.
            - ``boundaries`` — boundary indicator scores at each position,
              shape ``(batch, seq_len)``.
            - ``type_embedding`` — dense embedding capturing the sentence
              type, shape ``(batch, dim)``.
        """
        ...

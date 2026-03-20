"""Sentence type classification head.

Classifies token sequences by sentence type (e.g. statement, question,
imperative, exclamatory) and produces type-specific embeddings that
encode sentence-level properties.
"""

from __future__ import annotations

from torch import Tensor, nn

from lfm._registry import register
from lfm._types import Mask, TokenEmbeddings
from lfm.sentence.base import SentenceModule
from lfm.sentence.config import SentenceConfig


@register("sentence", "type_head")
class SentenceTypeHead(SentenceModule):
    """Classifies sentence type and produces type embeddings.

    Pools token-level embeddings into a sentence representation, classifies
    the sentence type, and produces a type-specific embedding that can be
    used by downstream components to condition generation or evaluation
    on the sentence type.

    Args:
        config: Sentence configuration specifying the number of sentence
            types and boundary detection parameters.
    """

    def __init__(self, config: SentenceConfig) -> None:
        super().__init__(config)

        self.num_sentence_types = config.num_sentence_types

        # Placeholder classification head — input dim will be determined
        # by upstream embedding size at build time.
        self.classifier: nn.Module | None = None

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, embeddings: TokenEmbeddings, mask: Mask) -> dict[str, Tensor]:
        """Classify sentence type and produce type embeddings.

        Args:
            embeddings: Dense token embeddings of shape
                ``(batch, seq_len, dim)``.
            mask: Boolean padding mask of shape ``(batch, seq_len)``.

        Returns:
            Dictionary with sentence type logits, boundary indicators,
            and type embeddings.
        """
        raise NotImplementedError("SentenceTypeHead.forward() not yet implemented")

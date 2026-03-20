"""Sentence boundary detection module.

Detects sentence boundaries in token sequences, producing per-position
boundary scores that indicate where one sentence ends and the next begins.
"""

from __future__ import annotations

from torch import Tensor, nn

from lfm._registry import register
from lfm._types import Mask, TokenEmbeddings
from lfm.sentence.base import SentenceModule
from lfm.sentence.config import SentenceConfig


@register("sentence", "boundary_detector")
class BoundaryDetector(SentenceModule):
    """Detects sentence boundaries in token sequences.

    Produces a boundary score at each token position indicating the
    likelihood that a sentence boundary occurs at that point.  The
    boundary threshold from the config controls the binarisation of
    these scores into hard boundary decisions.

    Args:
        config: Sentence configuration specifying boundary threshold
            and sentence type parameters.
    """

    def __init__(self, config: SentenceConfig) -> None:
        super().__init__(config)

        self.boundary_threshold = config.boundary_threshold
        self.num_sentence_types = config.num_sentence_types

        # Placeholder boundary scoring head — input dim determined
        # by upstream embedding size at build time.
        self.boundary_head: nn.Module | None = None

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, embeddings: TokenEmbeddings, mask: Mask) -> dict[str, Tensor]:
        """Detect sentence boundaries and classify sentence types.

        Args:
            embeddings: Dense token embeddings of shape
                ``(batch, seq_len, dim)``.
            mask: Boolean padding mask of shape ``(batch, seq_len)``.

        Returns:
            Dictionary with sentence type logits, boundary indicators,
            and type embeddings.
        """
        raise NotImplementedError("BoundaryDetector.forward() not yet implemented")

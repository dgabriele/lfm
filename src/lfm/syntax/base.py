"""Abstract base class for syntax modules.

All syntax implementations must subclass ``SyntaxModule`` and implement the
required abstract interface.  The syntax module induces hierarchical
constituency structure over token sequences and produces structural masks
that can constrain downstream attention patterns.
"""

from __future__ import annotations

from abc import abstractmethod
from typing import ClassVar

import torch
from torch import Tensor

from lfm._types import Mask, TokenEmbeddings
from lfm.core.module import LFMModule


class SyntaxModule(LFMModule):
    """Abstract base for syntax modules.

    A syntax module takes token embeddings and a padding mask and produces
    parse-related outputs: tree log-probabilities, structural attention masks,
    constituent representations, and parse depth information.

    Subclasses must implement:
        - ``forward``: full syntactic analysis producing tree probabilities,
          attention masks, constituents, and depth information.

    The ``constrain_attention`` method is provided with a default additive
    masking implementation and may be overridden for alternative strategies.
    """

    output_prefix: ClassVar[str] = "syntax"

    @abstractmethod
    def forward(self, embeddings: TokenEmbeddings, mask: Mask) -> dict[str, Tensor]:
        """Run syntactic analysis on a batch of token sequences.

        Args:
            embeddings: Dense token embeddings of shape
                ``(batch, seq_len, dim)``.
            mask: Boolean padding mask of shape ``(batch, seq_len)``, where
                ``True`` indicates a valid (non-padding) position.

        Returns:
            Dictionary with the following keys:

            - ``tree_log_probs`` — log-probability of the induced parse tree,
              shape ``(batch,)``.
            - ``attention_mask`` — structural mask for constraining attention,
              shape ``(batch, seq_len, seq_len)`` or broadcastable.
            - ``constituents`` — constituent representations, shape
              ``(batch, num_constituents, dim)``.
            - ``depth`` — parse depth at each token position, shape
              ``(batch, seq_len)``.
        """
        ...

    def constrain_attention(
        self, attention_logits: Tensor, syntax_output: dict[str, Tensor]
    ) -> Tensor:
        """Apply structural mask to attention logits.

        Uses additive masking by default: positions where the syntax mask is
        zero receive ``-inf``, preventing attention to structurally
        disconnected positions.

        Args:
            attention_logits: Raw attention logits of shape
                ``(batch, heads, seq_len, seq_len)`` or
                ``(batch, seq_len, seq_len)``.
            syntax_output: Output dictionary from ``forward()``, expected
                to contain an ``"attention_mask"`` key.

        Returns:
            Masked attention logits with the same shape as the input.
            If no ``"attention_mask"`` is present, the logits are returned
            unchanged.
        """
        mask = syntax_output.get("attention_mask")
        if mask is None:
            return attention_logits
        return attention_logits + torch.where(mask > 0, 0.0, float("-inf"))

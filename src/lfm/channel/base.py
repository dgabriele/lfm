"""Abstract base class for communication channel modules.

All channel implementations must subclass ``Channel`` and implement the
required abstract interface.  The channel module sits at the end of the
sender's pipeline, converting structured logits into discrete or
differentiable messages, and provides a decode path for the receiver.
"""

from __future__ import annotations

from abc import abstractmethod
from typing import ClassVar

from torch import Tensor

from lfm._types import Logits, Mask, TokenEmbeddings
from lfm.core.module import LFMModule


class Channel(LFMModule):
    """Abstract base for communication channel modules.

    A channel module takes logits over a message vocabulary and a padding
    mask and produces a discrete or differentiable message along with any
    channel-specific loss (e.g. entropy regularization).  It also provides
    a ``decode`` method for converting received messages back to embeddings
    on the receiver side.

    Subclasses must implement:
        - ``forward``: encode logits into a message with optional channel loss.
        - ``decode``: decode a received message back to token embeddings.
    """

    output_prefix: ClassVar[str] = "channel"

    @abstractmethod
    def forward(self, logits: Logits, mask: Mask) -> dict[str, Tensor]:
        """Encode logits into a channel message.

        Args:
            logits: Un-normalized log-probabilities over the message
                vocabulary, shape ``(batch, seq_len, vocab_size)``.
            mask: Boolean padding mask of shape ``(batch, seq_len)``, where
                ``True`` indicates a valid (non-padding) position.

        Returns:
            Dictionary with the following keys:

            - ``message`` — the encoded message, shape
              ``(batch, seq_len, vocab_size)`` for soft messages or
              ``(batch, seq_len)`` for hard discrete messages.
            - ``channel_loss`` — scalar channel loss (e.g. entropy penalty).
        """
        ...

    @abstractmethod
    def decode(self, message: Tensor) -> TokenEmbeddings:
        """Decode a received message back to token embeddings.

        This is the receiver-side operation that maps an incoming message
        (discrete indices or soft distributions) back into a dense
        embedding space for downstream processing.

        Args:
            message: The received message tensor.

        Returns:
            Token embeddings of shape ``(batch, seq_len, dim)``.
        """
        ...

"""Straight-through estimator channel.

Implements a differentiable discrete channel using the straight-through
estimator: the forward pass uses hard argmax for discretisation, while
the backward pass passes gradients through as if the operation were the
identity.
"""

from __future__ import annotations

from torch import Tensor, nn

from lfm._registry import register
from lfm._types import Logits, Mask, TokenEmbeddings
from lfm.channel.base import Channel
from lfm.channel.config import ChannelConfig


@register("channel", "straight_through")
class StraightThroughChannel(Channel):
    """Differentiable discrete channel using straight-through estimator.

    During the forward pass, messages are discretised via argmax.  During
    the backward pass, gradients are passed through the discretisation
    step unchanged (straight-through estimation).  This provides a simple
    mechanism for discrete communication with unbiased gradient estimates.

    Args:
        config: Channel configuration specifying vocabulary size and
            noise parameters.
    """

    def __init__(self, config: ChannelConfig) -> None:
        super().__init__(config)

        self.vocab_size = config.vocab_size
        self.noise_std = config.noise_std

        # Embedding for decoding received messages back to dense vectors
        if config.vocab_size is not None:
            self.embedding = nn.Embedding(config.vocab_size, config.vocab_size)
        else:
            self.embedding = None

    # ------------------------------------------------------------------
    # Forward / Decode
    # ------------------------------------------------------------------

    def forward(self, logits: Logits, mask: Mask) -> dict[str, Tensor]:
        """Encode logits into a discrete message via straight-through.

        Args:
            logits: Un-normalized log-probabilities of shape
                ``(batch, seq_len, vocab_size)``.
            mask: Boolean padding mask of shape ``(batch, seq_len)``.

        Returns:
            Dictionary with ``message`` and ``channel_loss`` keys.
        """
        raise NotImplementedError("StraightThroughChannel.forward() not yet implemented")

    def decode(self, message: Tensor) -> TokenEmbeddings:
        """Decode a received discrete message back to token embeddings.

        Args:
            message: The received message tensor.

        Returns:
            Token embeddings of shape ``(batch, seq_len, dim)``.
        """
        raise NotImplementedError("StraightThroughChannel.decode() not yet implemented")

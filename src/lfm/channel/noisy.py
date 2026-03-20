"""Noisy channel module.

Implements an information-bottleneck channel with configurable additive
Gaussian noise.  The noise level controls the amount of information that
can be transmitted through the channel, encouraging robust and compressed
representations.
"""

from __future__ import annotations

from torch import Tensor, nn

from lfm._registry import register
from lfm._types import Logits, Mask, TokenEmbeddings
from lfm.channel.base import Channel
from lfm.channel.config import ChannelConfig


@register("channel", "noisy")
class NoisyChannel(Channel):
    """Information-bottleneck channel with configurable noise.

    Adds Gaussian noise to the message representation during training,
    creating an information bottleneck that forces the sender to encode
    only the most important information.  The noise standard deviation
    is controlled by ``config.noise_std``.

    Args:
        config: Channel configuration specifying noise level, vocabulary
            size, and temperature parameters.
    """

    def __init__(self, config: ChannelConfig) -> None:
        super().__init__(config)

        self.vocab_size = config.vocab_size
        self.noise_std = config.noise_std

        # Embedding for decoding received messages
        if config.vocab_size is not None:
            self.embedding = nn.Embedding(config.vocab_size, config.vocab_size)
        else:
            self.embedding = None

    # ------------------------------------------------------------------
    # Forward / Decode
    # ------------------------------------------------------------------

    def forward(self, logits: Logits, mask: Mask) -> dict[str, Tensor]:
        """Encode logits through a noisy channel.

        Args:
            logits: Un-normalized log-probabilities of shape
                ``(batch, seq_len, vocab_size)``.
            mask: Boolean padding mask of shape ``(batch, seq_len)``.

        Returns:
            Dictionary with ``message`` and ``channel_loss`` keys.
        """
        raise NotImplementedError("NoisyChannel.forward() not yet implemented")

    def decode(self, message: Tensor) -> TokenEmbeddings:
        """Decode a noisy received message back to token embeddings.

        Args:
            message: The received message tensor.

        Returns:
            Token embeddings of shape ``(batch, seq_len, dim)``.
        """
        raise NotImplementedError("NoisyChannel.decode() not yet implemented")

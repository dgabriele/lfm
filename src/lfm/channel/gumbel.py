"""Gumbel-Softmax channel.

Implements a differentiable discrete communication channel using the
Gumbel-Softmax relaxation.  During training, messages are soft
(continuous relaxations of one-hot vectors); during evaluation, hard
argmax is used with straight-through gradients.
"""

from __future__ import annotations

from torch import Tensor, nn

from lfm._registry import register
from lfm._types import Logits, Mask, TokenEmbeddings
from lfm.channel.base import Channel
from lfm.channel.config import ChannelConfig


@register("channel", "gumbel")
class GumbelSoftmaxChannel(Channel):
    """Gumbel-Softmax channel for differentiable discrete communication.

    Uses the Gumbel-Softmax trick to produce approximately discrete
    messages that are fully differentiable.  The temperature parameter
    is annealed during training from ``temperature`` down to
    ``temperature_min`` over ``temperature_anneal_steps`` steps.

    Args:
        config: Channel configuration specifying temperature schedule,
            vocabulary size, and noise parameters.
    """

    def __init__(self, config: ChannelConfig) -> None:
        super().__init__(config)

        self.vocab_size = config.vocab_size
        self.temperature = config.temperature
        self.temperature_min = config.temperature_min
        self.temperature_anneal_steps = config.temperature_anneal_steps

        # Current temperature (mutable — updated during training)
        self._current_temperature = config.temperature

        # Embedding for decoding received messages
        if config.vocab_size is not None:
            self.embedding = nn.Embedding(config.vocab_size, config.vocab_size)
        else:
            self.embedding = None

    # ------------------------------------------------------------------
    # Forward / Decode
    # ------------------------------------------------------------------

    def forward(self, logits: Logits, mask: Mask) -> dict[str, Tensor]:
        """Encode logits into a Gumbel-Softmax message.

        Args:
            logits: Un-normalized log-probabilities of shape
                ``(batch, seq_len, vocab_size)``.
            mask: Boolean padding mask of shape ``(batch, seq_len)``.

        Returns:
            Dictionary with ``message`` and ``channel_loss`` keys.
        """
        raise NotImplementedError("GumbelSoftmaxChannel.forward() not yet implemented")

    def decode(self, message: Tensor) -> TokenEmbeddings:
        """Decode a received Gumbel-Softmax message back to embeddings.

        Args:
            message: The received message tensor.

        Returns:
            Token embeddings of shape ``(batch, seq_len, dim)``.
        """
        raise NotImplementedError("GumbelSoftmaxChannel.decode() not yet implemented")

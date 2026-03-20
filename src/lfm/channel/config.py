"""Configuration for channel modules.

Defines the ``ChannelConfig`` used to parameterize communication channel
components that handle message encoding and decoding between agents.
"""

from __future__ import annotations

from lfm.config.base import LFMBaseConfig


class ChannelConfig(LFMBaseConfig):
    """Configuration for a communication channel module.

    Attributes:
        name: Registry name of the channel implementation to use.
        temperature: Initial temperature for the Gumbel-Softmax or
            similar relaxation.
        temperature_min: Minimum temperature after annealing.
        temperature_anneal_steps: Number of training steps over which to
            anneal the temperature from ``temperature`` to
            ``temperature_min``.
        noise_std: Standard deviation of additive Gaussian noise injected
            into the channel (0.0 for a noiseless channel).
        vocab_size: Size of the discrete message vocabulary.  If ``None``,
            this is inferred from the upstream quantizer's codebook size.
    """

    name: str = "gumbel"
    temperature: float = 1.0
    temperature_min: float = 0.1
    temperature_anneal_steps: int = 10000
    noise_std: float = 0.0
    vocab_size: int | None = None

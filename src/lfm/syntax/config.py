"""Configuration for syntax modules.

Defines the ``SyntaxConfig`` used to parameterize structural agreement and
ordering modules that learn soft morphological constraints.
"""

from __future__ import annotations

from lfm.config.base import LFMBaseConfig


class SyntaxConfig(LFMBaseConfig):
    """Configuration for structural agreement and ordering modules.

    Attributes:
        name: Registry name of the syntax implementation to use.
        num_agreement_heads: Number of multi-head attention heads for
            agreement computation. Each head specializes in a different
            type of agreement (e.g., case, number).
        ordering_temperature: Temperature parameter for ordering score
            softmax. Lower values produce sharper ordering preferences.
        latent_dim: Dimensionality of the latent structural representations.
    """

    name: str = "agreement"
    num_agreement_heads: int = 4
    ordering_temperature: float = 1.0
    latent_dim: int = 64

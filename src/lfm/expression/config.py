"""Configuration for expression generation."""

from __future__ import annotations

from lfm.config.base import LFMBaseConfig


class ExpressionConfig(LFMBaseConfig):
    """Configuration for ExpressionGenerator and ExpressionEncoder.

    Args:
        input_dim: Dimension of the input anchor embedding.
        latent_dim: Dimension of the leaf z vectors.
        hidden_dim: Decoder hidden dimension.
        output_dim: ExpressionEncoder output message dimension.
        max_depth: Maximum tree depth (root = depth 0).
        min_depth: Minimum forced depth before leaf decisions are allowed.
        max_tokens_per_leaf: Maximum tokens per leaf phrase before z-switch.
        transition_on_eos: Whether to switch z when decoder emits EOS.
    """

    input_dim: int = 384
    latent_dim: int = 384
    hidden_dim: int = 512
    output_dim: int = 384
    max_depth: int = 3
    min_depth: int = 1
    max_tokens_per_leaf: int = 96
    transition_on_eos: bool = True

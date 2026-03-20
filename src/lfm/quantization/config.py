"""Configuration for quantization modules.

Defines the ``QuantizationConfig`` used to parameterize all quantizer
implementations (e.g. VQ-VAE, FSQ, residual quantization).
"""

from __future__ import annotations

from lfm.config.base import LFMBaseConfig


class QuantizationConfig(LFMBaseConfig):
    """Configuration for a quantization module.

    Attributes:
        name: Registry name of the quantizer implementation to use.
        codebook_size: Number of entries in each codebook.
        codebook_dim: Dimensionality of each codebook vector.
        num_codebooks: Number of independent codebooks (for residual /
            product quantization).
        commitment_weight: Scalar weight for the commitment loss term that
            encourages encoder outputs to stay close to codebook entries.
        decay: Exponential moving average decay rate for codebook updates.
        input_dim: Dimensionality of the incoming agent state vectors.
        seq_len: Number of discrete tokens to produce from each agent state.
    """

    name: str = "vqvae"
    codebook_size: int = 512
    codebook_dim: int = 64
    num_codebooks: int = 1
    commitment_weight: float = 0.25
    decay: float = 0.99
    input_dim: int = 256
    seq_len: int = 16

"""Finite Scalar Quantization (FSQ) implementation.

FSQ quantizes each dimension of the latent vector independently to one of a
fixed number of levels, avoiding codebook collapse entirely.  The total
codebook size is the product of the per-dimension levels.
"""

from __future__ import annotations

from math import prod

from torch import Tensor, nn

from lfm._registry import register
from lfm._types import TokenEmbeddings, TokenIds
from lfm.quantization.base import Quantizer
from lfm.quantization.config import QuantizationConfig


@register("quantizer", "fsq")
class FSQQuantizer(Quantizer):
    """Finite Scalar Quantization -- per-dimension rounding, no codebook collapse.

    Each latent dimension is independently rounded to one of a fixed number of
    levels.  The implicit codebook size is the product of all per-dimension
    levels, giving precise control over the number of representable codes
    without any risk of index collapse.

    The ``levels`` parameter is taken from the config's ``codebook_dim`` field
    to determine the number of quantized dimensions, with each dimension
    quantized to ``codebook_size`` levels by default.  For more precise
    control, subclass and override.

    Args:
        config: Quantization configuration.  ``codebook_dim`` controls the
            number of quantized dimensions.
    """

    def __init__(self, config: QuantizationConfig) -> None:
        super().__init__(config)

        self._num_dims = config.codebook_dim
        self._levels_per_dim = config.codebook_size
        self._seq_len = config.seq_len

        # Total implicit codebook size is levels_per_dim ** num_dims, but
        # we expose the per-dim level count via codebook_size for API compat.
        self._implicit_codebook_size = prod([self._levels_per_dim] * self._num_dims)

        # Project agent state into quantizable dimensions
        self.input_proj = nn.Linear(config.input_dim, config.seq_len * config.codebook_dim)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def codebook_size(self) -> int:
        """Return the implicit codebook size (product of per-dim levels)."""
        return self._implicit_codebook_size

    @property
    def codebook_dim(self) -> int:
        """Return the number of quantized dimensions."""
        return self._num_dims

    # ------------------------------------------------------------------
    # Lookup
    # ------------------------------------------------------------------

    def lookup(self, indices: TokenIds) -> TokenEmbeddings:
        """Look up embeddings for FSQ token indices.

        Args:
            indices: Integer tensor of shape ``(batch, seq_len)``.

        Returns:
            Token embeddings of shape ``(batch, seq_len, codebook_dim)``.
        """
        raise NotImplementedError("FSQQuantizer.lookup() not yet implemented")

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, x: Tensor) -> dict[str, Tensor]:
        """Quantize a batch of continuous agent state vectors via FSQ.

        Args:
            x: Agent state tensor of shape ``(batch, input_dim)``.

        Returns:
            Dictionary with keys ``tokens``, ``embeddings``, and
            ``commitment_loss``.
        """
        raise NotImplementedError("FSQQuantizer.forward() not yet implemented")

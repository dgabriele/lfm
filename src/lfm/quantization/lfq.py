"""Lookup-Free Quantization (LFQ) implementation.

LFQ replaces codebook lookup with per-dimension binary thresholding.
Each latent dimension is mapped to {-1, +1}, eliminating the codebook
table entirely while retaining a large implicit vocabulary.
"""

from __future__ import annotations

from torch import Tensor, nn

from lfm._registry import register
from lfm._types import TokenEmbeddings, TokenIds
from lfm.quantization.base import Quantizer
from lfm.quantization.config import QuantizationConfig


@register("quantizer", "lfq")
class LFQQuantizer(Quantizer):
    """Lookup-Free Quantization -- binary thresholding, no codebook table.

    Each dimension of the latent representation is binarised to {-1, +1}
    using a sign function (with straight-through gradient estimation).
    The implicit codebook size is ``2 ** codebook_dim``, giving exponential
    representational capacity with linear parameter cost.

    Args:
        config: Quantization configuration.  ``codebook_dim`` controls the
            number of binary dimensions (and thus implicit codebook size
            ``2 ** codebook_dim``).
    """

    def __init__(self, config: QuantizationConfig) -> None:
        super().__init__(config)

        self._codebook_dim = config.codebook_dim
        self._codebook_size = 2**config.codebook_dim
        self._seq_len = config.seq_len

        # Project agent state into binary-quantisable dimensions
        self.input_proj = nn.Linear(config.input_dim, config.seq_len * config.codebook_dim)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def codebook_size(self) -> int:
        """Return the implicit codebook size (2 ** codebook_dim)."""
        return self._codebook_size

    @property
    def codebook_dim(self) -> int:
        """Return the number of binary dimensions."""
        return self._codebook_dim

    # ------------------------------------------------------------------
    # Lookup
    # ------------------------------------------------------------------

    def lookup(self, indices: TokenIds) -> TokenEmbeddings:
        """Convert integer indices to binary code vectors.

        Args:
            indices: Integer tensor of shape ``(batch, seq_len)``.

        Returns:
            Binary code vectors of shape ``(batch, seq_len, codebook_dim)``
            with values in {-1, +1}.
        """
        raise NotImplementedError("LFQQuantizer.lookup() not yet implemented")

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, x: Tensor) -> dict[str, Tensor]:
        """Quantize a batch of continuous agent state vectors via LFQ.

        Args:
            x: Agent state tensor of shape ``(batch, input_dim)``.

        Returns:
            Dictionary with keys ``tokens``, ``embeddings``, and
            ``commitment_loss``.
        """
        raise NotImplementedError("LFQQuantizer.forward() not yet implemented")

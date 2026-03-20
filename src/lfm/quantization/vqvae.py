"""VQ-VAE quantizer implementation.

Vector Quantization Variational Autoencoder with exponential moving average
(EMA) codebook updates.  Converts continuous agent state vectors into discrete
tokens by nearest-neighbour lookup in a learned codebook.
"""

from __future__ import annotations

from torch import Tensor, nn

from lfm._registry import register
from lfm._types import TokenEmbeddings, TokenIds
from lfm.quantization.base import Quantizer
from lfm.quantization.config import QuantizationConfig


@register("quantizer", "vqvae")
class VQVAEQuantizer(Quantizer):
    """Vector Quantization VAE with EMA codebook updates.

    Maps continuous agent states to a sequence of discrete codebook indices
    via nearest-neighbour lookup.  The codebook is updated using an
    exponential moving average of encoder outputs, which avoids the need
    for a separate codebook gradient and improves training stability.

    Args:
        config: Quantization configuration specifying codebook size,
            dimensionality, input projection, and EMA parameters.
    """

    def __init__(self, config: QuantizationConfig) -> None:
        super().__init__(config)

        self._codebook_size = config.codebook_size
        self._codebook_dim = config.codebook_dim
        self._commitment_weight = config.commitment_weight
        self._decay = config.decay
        self._seq_len = config.seq_len

        # Project agent state to (seq_len * codebook_dim) then reshape
        self.input_proj = nn.Linear(config.input_dim, config.seq_len * config.codebook_dim)

        # Learnable codebook embeddings
        self.codebook = nn.Embedding(config.codebook_size, config.codebook_dim)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def codebook_size(self) -> int:
        """Return the number of entries in the codebook."""
        return self._codebook_size

    @property
    def codebook_dim(self) -> int:
        """Return the dimensionality of each codebook vector."""
        return self._codebook_dim

    # ------------------------------------------------------------------
    # Lookup (trivial — implemented)
    # ------------------------------------------------------------------

    def lookup(self, indices: TokenIds) -> TokenEmbeddings:
        """Look up codebook embeddings for the given token indices.

        Args:
            indices: Integer tensor of shape ``(batch, seq_len)``.

        Returns:
            Token embeddings of shape ``(batch, seq_len, codebook_dim)``.
        """
        return self.codebook(indices)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, x: Tensor) -> dict[str, Tensor]:
        """Quantize a batch of continuous agent state vectors.

        Args:
            x: Agent state tensor of shape ``(batch, input_dim)``.

        Returns:
            Dictionary with keys ``tokens``, ``embeddings``, and
            ``commitment_loss``.
        """
        raise NotImplementedError("VQVAEQuantizer.forward() not yet implemented")

"""Abstract base class for quantization modules.

All quantizer implementations must subclass ``Quantizer`` and implement the
required abstract interface.  The quantizer sits at the beginning of the LFM
pipeline, converting continuous agent state vectors into discrete token
representations suitable for downstream linguistic processing.
"""

from __future__ import annotations

from abc import abstractmethod
from typing import ClassVar

from torch import Tensor

from lfm._types import AgentState, TokenEmbeddings, TokenIds
from lfm.core.module import LFMModule


class Quantizer(LFMModule):
    """Abstract base for quantization modules.

    A quantizer maps continuous agent state vectors to discrete token
    sequences and their corresponding dense embeddings.  It also provides
    the reverse lookup from token indices to embeddings.

    Subclasses must implement:
        - ``forward``: full quantization pass producing tokens, embeddings,
          and commitment loss.
        - ``lookup``: codebook lookup from integer indices to embeddings.
        - ``codebook_size``: number of entries in the codebook.
        - ``codebook_dim``: dimensionality of each codebook vector.
    """

    output_prefix: ClassVar[str] = "quantization"

    @abstractmethod
    def forward(self, x: AgentState) -> dict[str, Tensor]:
        """Quantize a batch of continuous agent state vectors.

        Args:
            x: Agent state tensor of shape ``(batch, input_dim)``.

        Returns:
            Dictionary with the following keys:

            - ``tokens`` — integer token indices, shape ``(batch, seq_len)``.
            - ``embeddings`` — dense embeddings for the selected codes,
              shape ``(batch, seq_len, codebook_dim)``.
            - ``commitment_loss`` — scalar commitment loss.
        """
        ...

    @abstractmethod
    def lookup(self, indices: TokenIds) -> TokenEmbeddings:
        """Look up codebook embeddings for the given token indices.

        Args:
            indices: Integer tensor of shape ``(batch, seq_len)``.

        Returns:
            Token embeddings of shape ``(batch, seq_len, codebook_dim)``.
        """
        ...

    @property
    @abstractmethod
    def codebook_size(self) -> int:
        """Return the number of entries in the codebook."""
        ...

    @property
    @abstractmethod
    def codebook_dim(self) -> int:
        """Return the dimensionality of each codebook vector."""
        ...

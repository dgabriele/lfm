"""Abstract base class for generator modules.

All generator implementations must subclass ``GeneratorModule`` and implement
the required abstract interface.  A generator maps input embeddings to
variable-length subword token sequences via a learned generative model,
providing an alternative to the stage-by-stage linguistic pipeline.
"""

from __future__ import annotations

from abc import abstractmethod
from typing import ClassVar

from torch import Tensor

from lfm._types import Mask, TokenEmbeddings
from lfm.core.module import LFMModule


class GeneratorModule(LFMModule):
    """Base class for generator modules.

    A generator takes input embeddings (from quantizer, pretokenized, or raw
    agent state) and produces variable-length subword token sequences via a
    generative model.  Unlike the stage-by-stage pipeline, generators produce
    complete linguistic surface forms in one shot.

    Subclasses must implement:
        - ``forward``: generate subword tokens from input embeddings.
        - ``decode_to_text``: convert generated token ids to strings.
    """

    output_prefix: ClassVar[str] = "generator"

    @abstractmethod
    def forward(
        self,
        embeddings: TokenEmbeddings,
        mask: Mask,
    ) -> dict[str, Tensor]:
        """Generate subword token sequences from input embeddings.

        Args:
            embeddings: Dense input embeddings of shape
                ``(batch, seq_len, dim)``.
            mask: Boolean mask of shape ``(batch, seq_len)``.
                ``True`` marks valid positions.

        Returns:
            Dictionary with the following keys:

            - ``tokens`` — generated subword token indices, shape
              ``(batch, output_len)``, dtype ``int64``.
            - ``token_probs`` — differentiable Gumbel-Softmax distributions,
              shape ``(batch, output_len, vocab_size)``.
            - ``embeddings`` — decoder hidden states, shape
              ``(batch, output_len, decoder_hidden_dim)``.
            - ``lengths`` — actual sequence lengths before padding, shape
              ``(batch,)``.
            - ``mask`` — boolean output mask, shape ``(batch, output_len)``.
            - ``mu`` — VAE posterior mean, shape ``(batch, latent_dim)``.
            - ``logvar`` — VAE posterior log-variance, shape
              ``(batch, latent_dim)``.
        """
        ...

    @abstractmethod
    def decode_to_text(self, token_ids: Tensor) -> list[str]:
        """Decode integer token ids to human-readable strings.

        Args:
            token_ids: Integer tensor of shape ``(batch, seq_len)``.

        Returns:
            List of decoded strings, one per batch element.
        """
        ...

"""Subword tokenizer wrapper for the generator module.

Provides a thin wrapper around ``sentencepiece`` for subword tokenization,
with BOS/EOS token management.  ``sentencepiece`` is imported lazily so the
rest of the generator module works without it installed.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from torch import Tensor

if TYPE_CHECKING:
    import sentencepiece as spm


class SubwordTokenizer:
    """Wrapper around sentencepiece for subword tokenization.

    BOS and EOS token ids are allocated past the sentencepiece vocabulary
    (``vocab_size`` and ``vocab_size + 1`` respectively) so they never
    collide with content tokens.

    Args:
        spm_model_path: Path to a trained ``.model`` file.
        vocab_size: Expected sentencepiece vocabulary size (validated
            against the loaded model).

    Raises:
        ImportError: If ``sentencepiece`` is not installed.
        ValueError: If the loaded model's vocabulary size doesn't match.
    """

    def __init__(self, spm_model_path: str, vocab_size: int) -> None:
        try:
            import sentencepiece as _spm
        except ImportError as e:
            raise ImportError(
                "sentencepiece is required for the generator module. "
                "Install it with: pip install sentencepiece>=0.2"
            ) from e

        self._sp: spm.SentencePieceProcessor = _spm.SentencePieceProcessor(
            model_file=spm_model_path
        )
        if self._sp.vocab_size() != vocab_size:
            raise ValueError(
                f"Sentencepiece model vocab_size={self._sp.vocab_size()} "
                f"does not match expected vocab_size={vocab_size}"
            )

    @property
    def vocab_size(self) -> int:
        """Number of content tokens (excludes BOS/EOS)."""
        return self._sp.vocab_size()

    @property
    def bos_id(self) -> int:
        """BOS token index (one past the sentencepiece vocab)."""
        return self._sp.vocab_size()

    @property
    def eos_id(self) -> int:
        """EOS token index (two past the sentencepiece vocab)."""
        return self._sp.vocab_size() + 1

    def encode(self, text: str) -> list[int]:
        """Encode text to a list of subword token ids.

        Args:
            text: Input text string.

        Returns:
            List of integer token ids (content tokens only, no BOS/EOS).
        """
        return self._sp.encode(text, out_type=int)

    def decode(self, ids: list[int]) -> str:
        """Decode token ids to a text string.

        BOS and EOS tokens are filtered out before decoding.

        Args:
            ids: List of integer token ids.

        Returns:
            Decoded text string.
        """
        filtered = [i for i in ids if i < self._sp.vocab_size()]
        return self._sp.decode(filtered)

    def batch_encode(
        self,
        texts: list[str],
        max_len: int,
        device: torch.device | str = "cpu",
    ) -> tuple[Tensor, Tensor]:
        """Encode and pad a batch of texts.

        Each text is tokenized and padded (or truncated) to ``max_len``.

        Args:
            texts: List of input text strings.
            max_len: Target sequence length.
            device: Device for output tensors.

        Returns:
            Tuple of ``(token_ids, lengths)`` where ``token_ids`` is
            ``(batch, max_len)`` and ``lengths`` is ``(batch,)``.
        """
        batch_ids: list[list[int]] = []
        lengths: list[int] = []

        for text in texts:
            ids = self._sp.encode(text, out_type=int)
            length = min(len(ids), max_len)
            padded = ids[:length] + [0] * (max_len - length)
            batch_ids.append(padded)
            lengths.append(length)

        return (
            torch.tensor(batch_ids, dtype=torch.long, device=device),
            torch.tensor(lengths, dtype=torch.long, device=device),
        )

    def batch_decode(self, token_ids: Tensor) -> list[str]:
        """Decode a batch of token id tensors to strings.

        Args:
            token_ids: Integer tensor of shape ``(batch, seq_len)``.

        Returns:
            List of decoded strings, one per batch element.
        """
        return [self.decode(row.tolist()) for row in token_ids]

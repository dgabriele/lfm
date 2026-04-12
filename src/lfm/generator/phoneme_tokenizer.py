"""Phoneme-level tokenizer for the Qwen-BPE-stable alphabet.

Each phoneme is a short ASCII string (2-3 chars) that Qwen's BPE
tokenizes deterministically as a single token when space-prefixed.
A Neuroglot "word" is one or more phonemes concatenated; "phrases" are
words separated by spaces; sequences are space-joined.

The tokenizer's internal vocabulary consists of the 50 phonemes plus
reserved BOS/EOS/PAD ids.  Encoding maps phoneme-index sequences to
integer id tensors (used by the VAE decoder); `decode_to_text` joins
phonemes with hyphens to produce surface Neuroglot strings.
"""

from __future__ import annotations

import json
from pathlib import Path

import torch
from torch import Tensor


class PhonemeTokenizer:
    """Maps between phoneme-index sequences and Neuroglot surface text.

    The alphabet is loaded from a JSON file produced by
    ``scripts/design_phoneme_alphabet_multi.py``.  Phoneme ids are the
    positional indices into the ``phonemes`` list; reserved ids
    ``bos_id`` and ``eos_id`` sit at the end of the id space and are
    populated by the VAE decoder itself (not by this tokenizer).

    Args:
        alphabet_path: Path to the phoneme-alphabet JSON artifact.
        word_boundary: Character used to join phonemes within a word
            when decoding to surface text.  Default ``"-"``.
    """

    def __init__(
        self,
        alphabet_path: str | Path,
        word_boundary: str = "",
    ) -> None:
        with open(alphabet_path, encoding="utf-8") as f:
            alphabet = json.load(f)
        self.phonemes: list[str] = alphabet["phonemes"]
        self.num_phonemes: int = len(self.phonemes)
        # Reserve one id past the phoneme range for the word-boundary marker.
        # vocab_size includes this reserved id, so decoder embedding/output
        # layers are sized correctly.
        self.word_boundary_id: int = self.num_phonemes  # id=50 for 50-phoneme alphabet
        self.vocab_size: int = self.num_phonemes + 1
        # Within-word joiner for display (empty = concatenated words).
        # Between words we always emit a literal space, regardless of this.
        self.word_boundary = word_boundary

        self._phoneme_to_id: dict[str, int] = {
            p: i for i, p in enumerate(self.phonemes)
        }
        self.alphabet_metadata: dict = {
            k: v for k, v in alphabet.items() if k != "phonemes"
        }

    def phoneme_to_id(self, phoneme: str) -> int:
        return self._phoneme_to_id[phoneme]

    def id_to_phoneme(self, pid: int) -> str:
        return self.phonemes[pid]

    def encode(self, phonemes: list[str]) -> list[int]:
        """Map a list of phoneme strings to integer ids."""
        return [self._phoneme_to_id[p] for p in phonemes]

    def decode_ids(self, ids: list[int]) -> list[str]:
        """Map integer ids to phoneme strings (ignoring out-of-vocab)."""
        out: list[str] = []
        for i in ids:
            if 0 <= i < self.vocab_size:
                out.append(self.phonemes[i])
        return out

    def batch_decode(
        self,
        token_ids: Tensor,
        word_size: int | None = None,
    ) -> list[str]:
        """Decode a batch of integer token-id tensors to Neuroglot strings.

        Word boundaries are native: wherever the VAE emits
        ``self.word_boundary_id`` (``50`` for the 50-phoneme alphabet),
        we emit a literal space.  Within-word phonemes are joined with
        ``self.word_boundary`` (empty by default → concatenated).

        Args:
            token_ids: Integer tensor of shape ``(batch, seq_len)``.
                Out-of-vocab ids (BOS/EOS/PAD outside [0, vocab_size))
                are skipped.
            word_size: Legacy parameter.  When the sequence contains
                word-boundary tokens this is ignored; it is only used as
                a fallback display grouping when no boundary tokens are
                present (e.g. when rendering a VAE trained without the
                boundary marker).  ``None`` (default) means "don't
                fallback-group; flat stream".

        Returns:
            List of Neuroglot strings, one per batch element.
        """
        if isinstance(token_ids, Tensor):
            token_ids = token_ids.detach().cpu().tolist()
        out: list[str] = []
        for row in token_ids:
            # Filter out BOS/EOS/PAD (ids outside valid vocab range).
            valid = [i for i in row if 0 <= i < self.vocab_size]
            if not valid:
                out.append("")
                continue

            has_boundaries = any(i == self.word_boundary_id for i in valid)
            if has_boundaries:
                # Native path: split on word-boundary id into words of phonemes.
                words: list[str] = []
                cur: list[str] = []
                for i in valid:
                    if i == self.word_boundary_id:
                        if cur:
                            words.append(self.word_boundary.join(cur))
                            cur = []
                    else:
                        cur.append(self.phonemes[i])
                if cur:
                    words.append(self.word_boundary.join(cur))
                out.append(" ".join(words))
            else:
                # Fallback: flat phoneme stream, no boundary info.
                phs = [self.phonemes[i] for i in valid if i != self.word_boundary_id]
                if word_size is None:
                    out.append(self.word_boundary.join(phs))
                else:
                    words = [
                        self.word_boundary.join(phs[i:i + word_size])
                        for i in range(0, len(phs), word_size)
                    ]
                    out.append(" ".join(words))
        return out

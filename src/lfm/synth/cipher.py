"""Deterministic English-word -> alien-syllable cipher.

Each English word maps to a fixed 1-3 syllable alien string based on a
SHA-256 hash of the lowercased word.  The mapping is stable across runs
as long as the vocabulary is the same (same seed / vocab_size).

Output format for the tokenizer uses space-separated syllables so that
each syllable is one token in the WordLevel alien tokenizer.  Punctuation
is emitted as a separate space-separated token.
"""

from __future__ import annotations

import hashlib
import re

from lfm.synth.vocab import AlienVocab


_WORD_RE = re.compile(r"[A-Za-zÀ-ɏ]+|[^\w\s]|\s+", re.UNICODE)


class WordCipher:
    """Deterministic English -> alien cipher.

    Args:
        vocab: The alien vocabulary whose syllable list is used.
    """

    def __init__(self, vocab: AlienVocab) -> None:
        self._sylls = vocab.syllables
        self._n = len(self._sylls)

    # ---- public API ----

    def word_syllables(self, word: str) -> list[str]:
        """Return alien syllables for a single English word."""
        key = word.lower()
        h = int(hashlib.sha256(key.encode()).hexdigest(), 16)
        n = 1 if len(key) <= 2 else (2 if len(key) <= 5 else 3)
        return [self._sylls[(h >> (i * 20)) % self._n] for i in range(n)]

    def encode_sentence(self, sentence: str, capitalize: bool = True) -> str:
        """Encode sentence to a space-separated alien syllable string.

        Punctuation tokens appear as individual space-separated items.
        Whitespace is collapsed to single spaces.

        Args:
            sentence: Source English text.
            capitalize: If True, capitalize the first syllable of each
                source-capitalized word (display mode).  Pass False to
                produce lowercase-only output suitable for the WordLevel
                tokenizer.
        """
        parts: list[str] = []
        for tok in _WORD_RE.findall(sentence):
            if tok.isspace():
                continue
            if not re.match(r"[A-Za-zÀ-ɏ]", tok):
                parts.append(tok.strip())
                continue
            sylls = self.word_syllables(tok)
            if capitalize and tok[0].isupper():
                sylls = [sylls[0][0].upper() + sylls[0][1:]] + sylls[1:]
            parts.extend(sylls)
        return " ".join(parts)

    def encode_for_tokenizer(self, sentence: str) -> str:
        """Encode to lowercase alien syllables for the WordLevel tokenizer."""
        return self.encode_sentence(sentence, capitalize=False)

    def encode_batch(self, sentences: list[str]) -> list[str]:
        return [self.encode_for_tokenizer(s) for s in sentences]

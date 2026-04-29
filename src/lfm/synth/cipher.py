"""Deterministic English-word -> alien-syllable cipher with optional
semantic-similarity-aware first-syllable selection.

Default mode: each English word maps to a fixed 1-3 syllable alien string
based on a SHA-256 hash of the lowercased word.  The mapping is stable
across runs as long as the vocabulary is the same (same seed / vocab_size).

Semantic mode (when ``cluster_map`` is provided): the FIRST syllable of
each word is drawn from a cluster-specific subset of the syllable vocab,
so that semantically similar English words (assigned to the same k-means
cluster on Qwen input embeddings) produce alien words that share
statistical first-syllable structure.  Subsequent syllables remain
hash-derived to preserve bijectivity within each cluster.  Out-of-vocab
words fall back to pure-hash mode.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

from lfm.synth.vocab import AlienVocab


_WORD_RE = re.compile(r"[A-Za-zÀ-ɏ]+|[^\w\s]|\s+", re.UNICODE)


class WordCipher:
    """Deterministic English -> alien cipher.

    Args:
        vocab: The alien vocabulary whose syllable list is used.
        cluster_map: Optional dict {english_word: cluster_id} produced by
            scripts/build_semantic_cipher_clusters.py. When supplied, the
            first alien syllable is restricted to the cluster's subset of
            the syllable vocab, giving semantically similar words a shared
            statistical signature.
        n_clusters: Number of distinct cluster ids in ``cluster_map``;
            determines the size of each cluster's syllable subset.
    """

    def __init__(
        self,
        vocab: AlienVocab,
        cluster_map: dict[str, int] | None = None,
        n_clusters: int | None = None,
    ) -> None:
        self._sylls = vocab.syllables
        self._n = len(self._sylls)
        self._cluster_map = cluster_map or {}
        self._n_clusters = n_clusters or 0
        # Each cluster owns a contiguous subset of size ``cluster_pool``.
        # Subsets overlap when n_clusters * pool > vocab — that's fine; the
        # signature is still cluster-stable, just less exclusive.
        self._cluster_pool = max(1, self._n // max(self._n_clusters, 1)) if self._n_clusters else 0

    @classmethod
    def from_dirs(cls, vocab: AlienVocab, output_dir: Path) -> "WordCipher":
        """Construct WordCipher, loading cluster_map from output_dir if present."""
        cm_path = Path(output_dir) / "word_clusters.json"
        if cm_path.exists():
            data = json.loads(cm_path.read_text())
            return cls(vocab, cluster_map=data["word_to_cluster"], n_clusters=data["n_clusters"])
        return cls(vocab)

    # ---- public API ----

    def word_syllables(self, word: str) -> list[str]:
        """Return alien syllables for a single English word."""
        key = word.lower()
        h = int(hashlib.sha256(key.encode()).hexdigest(), 16)
        n = 1 if len(key) <= 2 else (2 if len(key) <= 5 else 3)
        cluster_id = self._cluster_map.get(key, -1) if self._n_clusters else -1
        if cluster_id >= 0:
            # First syllable is constrained to cluster's pool.
            base = (cluster_id * self._cluster_pool) % self._n
            first_idx = (base + (h & 0xFFFFFFFF) % self._cluster_pool) % self._n
            sylls = [self._sylls[first_idx]]
            sylls += [self._sylls[(h >> (i * 20)) % self._n] for i in range(1, n)]
            return sylls
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
            parts.append("".join(sylls))
        return " ".join(parts)

    def encode_for_tokenizer(self, sentence: str) -> str:
        """Encode to lowercase alien syllables for the WordLevel tokenizer."""
        return self.encode_sentence(sentence, capitalize=False)

    def encode_batch(self, sentences: list[str]) -> list[str]:
        return [self.encode_for_tokenizer(s) for s in sentences]

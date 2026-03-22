"""Phonetically-weighted distance for IPA token sequences.

Computes edit distances between IPA subword sequences where substitution
costs are weighted by articulatory feature distance via PanPhon.  Two
tokens differing by one voicing feature (p/b) cost much less than two
differing by place+manner+voicing (p/ʒ).

This provides a linguistically grounded distance metric for the
topological regularization loss — the VAE latent space topology should
mirror the natural metric space of human phonology.
"""

from __future__ import annotations

import logging

import torch
from torch import Tensor

logger = logging.getLogger(__name__)


class PhoneticDistanceCache:
    """Cached phonetic feature distances between IPA characters.

    Builds a lookup table mapping IPA character pairs to articulatory
    feature distances via PanPhon on first use.  Subsequent lookups are
    O(1).  Unknown characters get a default high distance.

    IPA suprasegmentals (ː length mark, ˈˌ stress, ̃ nasalization, etc.)
    are not phones and lack articulatory features.  They get a small
    fixed distance (``suprasegmental_dist``) rather than the full default.

    The cache operates on individual IPA characters (not subwords),
    providing the substitution cost matrix for weighted edit distance.
    """

    # Characters that are IPA suprasegmentals / modifiers, not segments
    _SUPRASEGMENTALS = set("ːˈˌ̃̆̈ˑ")

    def __init__(
        self,
        default_dist: float = 12.0,
        suprasegmental_dist: float = 2.0,
    ) -> None:
        self._default_dist = default_dist
        self._suprasegmental_dist = suprasegmental_dist
        self._cache: dict[tuple[str, str], float] = {}
        self._feat_cache: dict[str, list[int] | None] = {}
        self._ft: object | None = None

    def _ensure_panphon(self) -> None:
        """Lazily load PanPhon."""
        if self._ft is not None:
            return
        try:
            import panphon

            self._ft = panphon.FeatureTable()
        except ImportError:
            logger.warning(
                "PanPhon not available; using default distances for topo loss"
            )

    def _get_features(self, char: str) -> list[int] | None:
        """Get articulatory features for an IPA character."""
        if char not in self._feat_cache:
            self._ensure_panphon()
            if self._ft is None:
                self._feat_cache[char] = None
            else:
                fts = self._ft.fts(char)  # type: ignore[union-attr]
                # PanPhon returns a Segment for valid IPA phones, or an
                # empty dict for non-phoneme characters (ː, punctuation, etc.)
                if fts is not None and hasattr(fts, "numeric"):
                    self._feat_cache[char] = fts.numeric()
                else:
                    self._feat_cache[char] = None
        return self._feat_cache[char]

    def char_distance(self, a: str, b: str) -> float:
        """Articulatory feature distance between two IPA characters.

        Returns sum of absolute feature differences (Manhattan distance
        on the articulatory feature space).  Suprasegmental modifiers
        (ː, stress marks, etc.) get a small fixed cost.  Unknown
        characters get ``default_dist``.
        """
        if a == b:
            return 0.0
        key = (min(a, b), max(a, b))
        if key not in self._cache:
            a_is_supra = a in self._SUPRASEGMENTALS
            b_is_supra = b in self._SUPRASEGMENTALS
            if a_is_supra or b_is_supra:
                # Suprasegmental vs anything: small fixed cost
                self._cache[key] = self._suprasegmental_dist
            else:
                fa = self._get_features(a)
                fb = self._get_features(b)
                if fa is None or fb is None:
                    self._cache[key] = self._default_dist
                else:
                    self._cache[key] = float(
                        sum(abs(x - y) for x, y in zip(fa, fb))
                    )
        return self._cache[key]


def weighted_phonetic_hamming(
    tokens_a: Tensor,
    tokens_b: Tensor,
    lengths_a: Tensor,
    lengths_b: Tensor,
    sp: object,
    dist_cache: PhoneticDistanceCache,
    max_compare_len: int = 32,
) -> Tensor:
    """Phonetically-weighted Hamming distance between token sequence pairs.

    Decodes token IDs to IPA text via sentencepiece, then computes
    character-level weighted Hamming distance using PanPhon articulatory
    features.  For sequences of different lengths, the shorter is padded
    with a high-cost mismatch.

    This is computed on CPU with no gradient requirement (used as a
    fixed target in the topo loss).

    Args:
        tokens_a: ``(n_pairs, seq_len)`` token IDs.
        tokens_b: ``(n_pairs, seq_len)`` token IDs.
        lengths_a: ``(n_pairs,)`` actual lengths.
        lengths_b: ``(n_pairs,)`` actual lengths.
        sp: Sentencepiece processor for decoding tokens to IPA text.
        dist_cache: PanPhon distance cache.
        max_compare_len: Maximum IPA characters to compare per pair.

    Returns:
        ``(n_pairs,)`` float tensor of phonetic distances.
    """
    n = tokens_a.size(0)
    distances = torch.zeros(n)

    for i in range(n):
        # Decode to IPA strings
        ids_a = tokens_a[i, : lengths_a[i]].cpu().tolist()
        ids_b = tokens_b[i, : lengths_b[i]].cpu().tolist()
        vocab_size = sp.vocab_size()  # type: ignore[union-attr]
        ids_a = [x for x in ids_a if x < vocab_size]
        ids_b = [x for x in ids_b if x < vocab_size]
        text_a = sp.decode(ids_a)  # type: ignore[union-attr]
        text_b = sp.decode(ids_b)  # type: ignore[union-attr]

        # Character-level weighted Hamming
        chars_a = list(text_a[:max_compare_len])
        chars_b = list(text_b[:max_compare_len])

        # Pad to same length
        max_len = max(len(chars_a), len(chars_b))
        total_dist = 0.0
        for j in range(max_len):
            ca = chars_a[j] if j < len(chars_a) else ""
            cb = chars_b[j] if j < len(chars_b) else ""
            if ca == "" or cb == "":
                total_dist += dist_cache._default_dist  # deletion/insertion
            elif ca != cb:
                total_dist += dist_cache.char_distance(ca, cb)

        # Normalize by length
        distances[i] = total_dist / max(max_len, 1)

    return distances

"""Robust, citable linguistic metrics for periodic checkpoint diagnostics.

Operates on lists of decoded text strings (post-respell). Cheap to
compute on a digest's n_recon=8 samples each checkpoint.

Implementations chosen for simplicity and stability under small N:
  - chrF: character-n-gram F-score (Popović 2015), order-6 macro-averaged
    F1 with beta=2 emphasizing recall. Robust on morphology-rich / IPA-ish
    text where word-level BLEU is brittle.
  - distinct-N: |unique n-grams| / |total n-grams| across all sample
    texts. Captures lexical diversity at the corpus level.
  - mean_syllables, mean_chars: simple structural complexity stats.
"""

from __future__ import annotations

from collections import Counter

# Vowels in the respelled IPA-ish corpus (matches romanize.respell).
_VOWELS = set("aeiouäàèéìíîòóùúëïöüÿɑɛɪɔʊə")


def _char_ngrams(s: str, n: int) -> Counter:
    s = s.replace(" ", "")
    if len(s) < n:
        return Counter()
    return Counter(s[i : i + n] for i in range(len(s) - n + 1))


def chrf(rec: str, ref: str, max_n: int = 6, beta: float = 2.0) -> float:
    """Character n-gram F-score, macro-averaged across orders 1..max_n."""
    if not rec or not ref:
        return 0.0
    f_scores = []
    for n in range(1, max_n + 1):
        rec_ng = _char_ngrams(rec, n)
        ref_ng = _char_ngrams(ref, n)
        rec_total = sum(rec_ng.values())
        ref_total = sum(ref_ng.values())
        match = sum((rec_ng & ref_ng).values())
        if rec_total == 0 or ref_total == 0:
            continue
        precision = match / rec_total
        recall = match / ref_total
        if precision + recall == 0:
            f = 0.0
        else:
            f = (1 + beta ** 2) * precision * recall / (beta ** 2 * precision + recall)
        f_scores.append(f)
    return sum(f_scores) / max(len(f_scores), 1)


def mean_chrf(recs: list[str], refs: list[str]) -> float:
    if not recs:
        return 0.0
    pairs = list(zip(recs, refs))
    return sum(chrf(r, g) for r, g in pairs) / len(pairs)


def distinct_n(texts: list[str], n: int = 1) -> float:
    """Unique-n-gram fraction across the corpus of generated texts."""
    total = 0
    seen = set()
    for t in texts:
        words = t.split()
        if len(words) < n:
            continue
        for i in range(len(words) - n + 1):
            ng = tuple(words[i : i + n])
            seen.add(ng)
            total += 1
    return len(seen) / max(total, 1)


def syllable_count(word: str) -> int:
    """Vowel-group count; rough syllable approximation."""
    if not word:
        return 0
    n = 0
    in_vowel = False
    for ch in word.lower():
        if ch in _VOWELS:
            if not in_vowel:
                n += 1
                in_vowel = True
        else:
            in_vowel = False
    return max(n, 1)


def mean_syllables(texts: list[str]) -> float:
    words = [w for t in texts for w in t.split()]
    if not words:
        return 0.0
    return sum(syllable_count(w) for w in words) / len(words)


def mean_chars(texts: list[str]) -> float:
    words = [w for t in texts for w in t.split()]
    if not words:
        return 0.0
    return sum(len(w) for w in words) / len(words)

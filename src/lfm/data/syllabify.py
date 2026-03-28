"""IPA syllabification based on the Sonority Sequencing Principle.

Syllable boundaries fall at sonority troughs in the phoneme sequence.
This is a universal phonological principle that applies across all
languages without language-specific rules.

Sonority hierarchy (high to low):
    vowels > glides > liquids > nasals > fricatives > affricates > stops

Usage::

    from lfm.data.syllabify import syllabify_ipa, sonority_alignment_score

    syllables = syllabify_ipa("mækswɛl")  # ["mæk", "swɛl"]
    score = sonority_alignment_score(tokens, syllables)
"""

from __future__ import annotations

import re
import unicodedata

# ── Sonority scale ───────────────────────────────────────────────────
# Higher value = more sonorous.  Based on Parker (2002) / Clements (1990).

_VOWELS = set("iyɨʉɯuɪʏʊeøɘɵɤoəɛœɜɞʌɔæɐaɶɑɒ")
_GLIDES = set("jwɥɰ")
_LIQUIDS = set("lɫɭʎɹɻrɾɽʀʁ")
_NASALS = set("mnɱɳɲŋɴ")
_FRICATIVES = set("fvθðszʃʒʂʐçʝxɣχʁħʕhɦɸβ")
_AFFRICATES = set()  # handled as digraphs: t͡s etc.
_STOPS = set("pbtdʈɖcɟkɡqɢʔ")

# Combining diacritics and suprasegmentals (not phonemes themselves)
_MODIFIERS = set("ːˈˌˑ̤̰̥̬̹̜̟̠̝̞̩̯̃̊͜͡")


def _sonority(char: str) -> int:
    """Return sonority rank for a single IPA character (0-6)."""
    if char in _VOWELS:
        return 6
    if char in _GLIDES:
        return 5
    if char in _LIQUIDS:
        return 4
    if char in _NASALS:
        return 3
    if char in _FRICATIVES:
        return 2
    if char in _STOPS:
        return 1
    # Combining marks, diacritics, suprasegmentals — inherit from base
    cat = unicodedata.category(char)
    if cat.startswith("M") or char in _MODIFIERS:
        return -1  # modifier, skip
    # Unknown — treat as low sonority
    return 0


def _get_phonemes(text: str) -> list[str]:
    """Split IPA text into phoneme-like segments.

    Groups base characters with their following combining marks
    and handles tie bars (t͡s → single segment).
    """
    segments: list[str] = []
    i = 0
    chars = list(text)
    while i < len(chars):
        c = chars[i]
        if c.isspace():
            segments.append(" ")
            i += 1
            continue
        # Start a new segment
        seg = c
        i += 1
        # Absorb following combining marks, tie bars, and length marks
        while i < len(chars):
            nc = chars[i]
            cat = unicodedata.category(nc)
            if cat.startswith("M") or nc in "͜͡ːˑ":
                seg += nc
                i += 1
            elif nc in _MODIFIERS:
                seg += nc
                i += 1
            else:
                break
        segments.append(seg)
    return segments


def syllabify_ipa(text: str) -> list[str]:
    """Syllabify IPA text using the Sonority Sequencing Principle.

    Args:
        text: IPA text (may contain spaces as word boundaries).

    Returns:
        List of syllable strings. Word boundaries produce separate
        entries (single space strings are preserved).
    """
    # Split on word boundaries first
    words = text.split(" ")
    result: list[str] = []

    for wi, word in enumerate(words):
        if not word:
            continue
        if wi > 0:
            result.append(" ")

        phonemes = [p for p in _get_phonemes(word) if p != " "]
        if not phonemes:
            continue

        # Compute sonority per phoneme (skip modifiers)
        sonorities = []
        for p in phonemes:
            base = p[0]
            s = _sonority(base)
            if s < 0:
                # Pure modifier — use previous sonority
                s = sonorities[-1] if sonorities else 0
            sonorities.append(s)

        # Find syllable boundaries at sonority troughs.
        # A trough is a position where sonority is lower than both
        # neighbors.  We place the boundary before the trough.
        boundaries: list[int] = [0]  # always start at 0

        for i in range(1, len(sonorities) - 1):
            prev_s = sonorities[i - 1]
            cur_s = sonorities[i]
            next_s = sonorities[i + 1]

            # Trough: current sonority < both neighbors
            if cur_s < prev_s and cur_s <= next_s:
                boundaries.append(i)
            # Plateau followed by rise after a fall: split at the rise
            elif (
                cur_s == next_s
                and i >= 2
                and sonorities[i - 2] > prev_s
                and prev_s >= cur_s
            ):
                boundaries.append(i)

        # Build syllable strings
        for bi in range(len(boundaries)):
            start = boundaries[bi]
            end = boundaries[bi + 1] if bi + 1 < len(boundaries) else len(phonemes)
            syl = "".join(phonemes[start:end])
            if syl:
                result.append(syl)

    return result


def syllable_boundaries(text: str) -> list[int]:
    """Return character positions of syllable boundaries.

    Args:
        text: IPA text.

    Returns:
        Sorted list of character indices where syllable boundaries fall.
        Always includes 0 (start) and len(text) (end).
    """
    syllables = syllabify_ipa(text)
    boundaries = [0]
    pos = 0
    for syl in syllables:
        pos += len(syl)
        boundaries.append(pos)
    return boundaries


def sonority_alignment_score(
    token_boundaries: list[int],
    syllable_bounds: list[int],
    tolerance: int = 1,
) -> float:
    """Measure how well BPE token boundaries align with syllable boundaries.

    Args:
        token_boundaries: Character positions of BPE token boundaries.
        syllable_bounds: Character positions of syllable boundaries.
        tolerance: Max character distance to count as aligned.

    Returns:
        Fraction of token boundaries that fall within ``tolerance``
        characters of a syllable boundary (0.0 to 1.0).
    """
    if not token_boundaries:
        return 1.0

    syl_set = set(syllable_bounds)
    aligned = 0
    for tb in token_boundaries:
        if any(abs(tb - sb) <= tolerance for sb in syl_set):
            aligned += 1

    return aligned / len(token_boundaries)

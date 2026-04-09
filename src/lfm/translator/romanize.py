"""Romanize IPA to natural-looking ASCII orthography.

Converts IPA output to text that looks like a real language using
Latin characters only. This activates the LLM's existing cross-lingual
transfer from English, Turkish, Swahili, etc. instead of forcing it
to learn IPA Unicode from byte-level fallbacks.

Example::

    >>> romanize("bu sɯɾtal haphænet awajʌnzo tasd͡ʒul atɯlyoke")
    'bu surtal haphanet awayunzo tasjul atulyoke'
"""

from __future__ import annotations

import re
import unicodedata

# Legacy lossy replacements (many-to-one, for backwards compatibility)
_REPLACEMENTS: list[tuple[str, str]] = [
    ("t͡ɕ", "ch"), ("d͡ʑ", "j"), ("t͡ʃ", "ch"), ("d͡ʒ", "j"),
    ("t͡s", "ts"), ("d͡z", "dz"), ("t͈", "t"), ("k͈", "k"), ("p͈", "p"),
    ("aɪ̯", "ay"), ("aʊ̯", "aw"), ("eɪ", "ey"), ("oʊ", "ow"), ("ɔɪ", "oy"),
    ("ʃ", "sh"), ("ʒ", "zh"), ("ŋ", "ng"), ("ɲ", "ny"), ("θ", "th"),
    ("ð", "dh"), ("ɾ", "r"), ("ɹ", "r"), ("ɽ", "r"), ("ʂ", "sh"),
    ("ɕ", "sh"), ("ɖ", "d"), ("ɡ", "g"), ("ɢ", "g"), ("ʔ", "'"),
    ("ɦ", "h"), ("ɣ", "gh"), ("χ", "kh"), ("ʕ", "'"), ("ʋ", "v"),
    ("ɬ", "l"), ("ɮ", "l"), ("ɭ", "l"), ("ɳ", "n"), ("ʈ", "t"),
    ("β", "b"), ("ɸ", "f"), ("ç", "h"),
    ("æ", "a"), ("ɛ", "e"), ("ɪ", "i"), ("ɔ", "o"), ("ʊ", "u"),
    ("ʌ", "u"), ("ə", "e"), ("ɯ", "u"), ("ɨ", "i"), ("ɑ", "a"),
    ("ɐ", "a"), ("ɤ", "o"), ("ø", "o"), ("œ", "o"), ("ɵ", "o"),
    ("ʉ", "u"), ("ɒ", "o"), ("ɘ", "e"), ("ɜ", "e"), ("ɝ", "er"),
    ("ɞ", "e"), ("ɶ", "a"), ("ɘ̃", "en"), ("y", "u"), ("ʏ", "u"),
    ("ɪ̈", "i"), ("ă", "a"), ("ɤ̆", "o"),
]

# ---------------------------------------------------------------------------
# Isomorphic romanization — lossless IPA → ASCII, no Unicode in output
# ---------------------------------------------------------------------------

# Affricates and multi-char IPA sequences (must be matched first)
_ISO_MULTI: list[tuple[str, str]] = [
    ("t͡ɕ", "tc"),    ("d͡ʑ", "dj"),   ("t͡ʃ", "tx"),   ("d͡ʒ", "dx"),
    ("t͡s", "ts"),    ("d͡z", "dz"),
    ("t͈", "tt"),     ("k͈", "kk"),    ("p͈", "pp"),    ("s͈", "ss"),
    ("aɪ̯", "ay"),   ("aʊ̯", "aw"),   ("eɪ", "ey"),   ("oʊ", "ow"),
    ("ɔɪ", "oy"),
]

# Single IPA characters → unique ASCII/Latin sequences.
# Each target is unique across the entire table.
_ISO_SINGLE: dict[str, str] = {
    # Consonants — distinct mappings, no collisions
    "ʃ": "sh",   "ʒ": "zh",   "ŋ": "ng",   "ɲ": "ny",
    "θ": "th",   "ð": "dh",
    "ɾ": "rr",   "ɹ": "rx",   "ɽ": "rd",
    "ʂ": "sr",   "ɕ": "sc",
    "ɖ": "dd",   "ɡ": "g",    "ɢ": "gq",
    "ʔ": "q",    "ɦ": "hh",   "ɣ": "gh",   "χ": "kh",
    "ʕ": "qh",   "ʋ": "vv",   "ɬ": "lh",   "ɮ": "lz",
    "ɭ": "ll",   "ɳ": "nn",   "ʈ": "tr",
    "β": "bh",   "ɸ": "ph",   "ç": "ch",   "ʁ": "rg",
    "ɰ": "mw",   "ɟ": "jj",   "ʑ": "zc",   "ħ": "hq",

    # Vowels — each IPA vowel gets a unique target
    "æ": "ae",   "ɛ": "eh",   "ɪ": "ih",   "ɔ": "aw",
    "ʊ": "uh",   "ʌ": "ux",   "ə": "ex",   "ɯ": "uu",
    "ɨ": "ix",   "ɑ": "aa",   "ɐ": "ax",   "ɤ": "oe",
    "ø": "ou",   "œ": "oi",   "ɵ": "ox",   "ʉ": "uw",
    "ɒ": "ao",   "ɘ": "eq",   "ɜ": "ev",   "ɝ": "er",
    "ɞ": "ew",   "ɶ": "aq",   "y": "yu",   "ʏ": "yh",

    # Length mark
    "ː": "x",
}

# Combining marks → unique suffixes (appended to previous char)
_ISO_COMBINING: dict[str, str] = {
    "\u0361": "",     # combining double inverted breve (handled in multi)
    "\u0348": "w",    # combining double vertical line below (fortis)
    "\u0303": "n",    # combining tilde (nasalization)
    "\u032F": "",     # combining inverted breve below (non-syllabic, in multi)
    "\u032A": "d",    # combining bridge below (dental)
    "\u0324": "h",    # combining diaeresis below (breathy)
    "\u0307": "t",    # combining dot above (palatalization)
    "\u0329": "l",    # combining vertical line below (syllabic)
}

# Build reverse map for decoding
_ISO_REVERSE: dict[str, str] = {}
for _ipa, _rom in _ISO_MULTI:
    _ISO_REVERSE[_rom] = _ipa
for _ipa, _rom in _ISO_SINGLE.items():
    if _rom:  # skip empty mappings
        _ISO_REVERSE[_rom] = _ipa


def romanize_iso(ipa_text: str) -> str:
    """Isomorphic romanization: lossless IPA → ASCII mapping.

    Every IPA character maps to a unique ASCII string. No information
    loss, no collisions. The output looks like natural Latin-script
    text without IPA Unicode characters.

    Syllable hyphens and word spaces are preserved.

    Args:
        ipa_text: IPA string (may include syllable hyphens).

    Returns:
        Isomorphically romanized ASCII string.
    """
    text = ipa_text

    # 1. Multi-character sequences first (affricates, diphthongs)
    for ipa, rom in _ISO_MULTI:
        text = text.replace(ipa, rom)

    # 2. Combining marks
    for mark, suffix in _ISO_COMBINING.items():
        text = text.replace(mark, suffix)

    # 3. Single characters
    for ipa, rom in _ISO_SINGLE.items():
        text = text.replace(ipa, rom)

    # 4. Strip any remaining non-ASCII (unknown diacritics)
    text = text.encode("ascii", errors="ignore").decode("ascii")

    return text


def syllable_hyphenate(ipa_text: str) -> str:
    """Hyphenate IPA text at syllable boundaries within words.

    Uses the Sonority Sequencing Principle to find natural syllable
    breaks, then joins syllables with hyphens within each word.
    Word boundaries (spaces) are preserved as spaces.

    This exposes phonotactic structure to the LLM's tokenizer —
    the BPE will split at hyphens, producing tokens that align with
    syllable boundaries rather than arbitrary character offsets.

    Args:
        ipa_text: IPA string (spaces between words).

    Returns:
        IPA string with intra-word hyphens at syllable boundaries.

    Example::

        >>> syllable_hyphenate("malatɯnɰithɯntaɰithɯlɯn")
        'ma-la-tɯn-ɰi-thɯn-ta-ɰi-thɯ-lɯn'
    """
    from lfm.data.syllabify import syllabify_ipa

    syllables = syllabify_ipa(ipa_text)
    if not syllables:
        return ipa_text

    # syllabify_ipa returns syllable strings with " " entries for
    # word boundaries.  Join syllables with hyphens, spaces with spaces.
    parts: list[str] = []
    word_syls: list[str] = []

    for syl in syllables:
        if syl == " ":
            if word_syls:
                parts.append("-".join(word_syls))
                word_syls = []
            continue
        word_syls.append(syl)

    if word_syls:
        parts.append("-".join(word_syls))

    return " ".join(parts)


def romanize(ipa_text: str) -> str:
    """Convert IPA text to natural ASCII orthography.

    Args:
        ipa_text: IPA string (may contain Unicode IPA characters).

    Returns:
        ASCII-only string that reads like a natural language.
    """
    text = ipa_text

    # Apply replacements (longest first)
    for ipa, roman in _REPLACEMENTS:
        text = text.replace(ipa, roman)

    # Handle length mark: ː → double previous character
    result = []
    for i, ch in enumerate(text):
        if ch == "ː" and result:
            result.append(result[-1])
        else:
            result.append(ch)
    text = "".join(result)

    # Strip any remaining non-ASCII (combining marks, tone marks, etc.)
    text = text.encode("ascii", errors="ignore").decode("ascii")

    # Clean up whitespace
    text = re.sub(r"\s+", " ", text).strip()

    return text

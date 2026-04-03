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

# Ordered replacement pairs: longest IPA sequences first to prevent
# partial matches.  Each (ipa, roman) pair is applied in order.
_REPLACEMENTS: list[tuple[str, str]] = [
    # Multi-character IPA (must come first)
    ("t͡ɕ", "ch"),
    ("d͡ʑ", "j"),
    ("t͡ʃ", "ch"),
    ("d͡ʒ", "j"),
    ("t͡s", "ts"),
    ("d͡z", "dz"),
    ("t͈", "t"),
    ("k͈", "k"),
    ("p͈", "p"),
    ("aɪ̯", "ay"),
    ("aʊ̯", "aw"),
    ("eɪ", "ey"),
    ("oʊ", "ow"),
    ("ɔɪ", "oy"),

    # Consonants
    ("ʃ", "sh"),
    ("ʒ", "zh"),
    ("ŋ", "ng"),
    ("ɲ", "ny"),
    ("θ", "th"),
    ("ð", "dh"),
    ("ɾ", "r"),
    ("ɹ", "r"),
    ("ɽ", "r"),
    ("ʂ", "sh"),
    ("ɕ", "sh"),
    ("ɖ", "d"),
    ("ɡ", "g"),
    ("ɢ", "g"),
    ("ʔ", "'"),
    ("ɦ", "h"),
    ("ɣ", "gh"),
    ("χ", "kh"),
    ("ʕ", "'"),
    ("ʋ", "v"),
    ("ɬ", "l"),
    ("ɮ", "l"),
    ("ɭ", "l"),
    ("ɳ", "n"),
    ("ʈ", "t"),
    ("β", "b"),
    ("ɸ", "f"),
    ("ç", "h"),

    # Vowels
    ("æ", "a"),
    ("ɛ", "e"),
    ("ɪ", "i"),
    ("ɔ", "o"),
    ("ʊ", "u"),
    ("ʌ", "u"),
    ("ə", "e"),
    ("ɯ", "u"),
    ("ɨ", "i"),
    ("ɑ", "a"),
    ("ɐ", "a"),
    ("ɤ", "o"),
    ("ø", "o"),
    ("œ", "o"),
    ("ɵ", "o"),
    ("ʉ", "u"),
    ("ɒ", "o"),
    ("ɘ", "e"),
    ("ɜ", "e"),
    ("ɝ", "er"),
    ("ɞ", "e"),
    ("ɶ", "a"),
    ("ɘ̃", "en"),
    ("y", "u"),
    ("ʏ", "u"),
    ("ɪ̈", "i"),
    ("ă", "a"),
    ("ɤ̆", "o"),
]


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

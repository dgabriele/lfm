"""English contraction expansion.

Expands contractions to full forms while preserving possessives.
Used in corpus preprocessing to ensure clean word-level tokenization.
"""

from __future__ import annotations

import re

# Ordered by specificity — longer patterns first to avoid partial matches.
# Each tuple: (pattern, replacement)
CONTRACTION_MAP: list[tuple[re.Pattern, str]] = [
    # Negations (most common, unambiguous)
    (re.compile(r"\bcan'?t\b", re.I), "can not"),
    (re.compile(r"\bwon'?t\b", re.I), "will not"),
    (re.compile(r"\bshan'?t\b", re.I), "shall not"),
    (re.compile(r"\bain'?t\b", re.I), "is not"),
    (re.compile(r"\b(\w+)n'?t\b", re.I), r"\1 not"),  # don't→do not, doesn't→does not, etc.
    # Would / had
    (re.compile(r"\b(i|you|he|she|it|we|they|who|that|there)'?d\b", re.I), r"\1 would"),
    # Will / shall
    (re.compile(r"\b(i|you|he|she|it|we|they|who|that|there)'?ll\b", re.I), r"\1 will"),
    # Have
    (re.compile(r"\b(i|you|we|they|who|that|there)'?ve\b", re.I), r"\1 have"),
    # Are
    (re.compile(r"\b(you|we|they|who|that|there)'?re\b", re.I), r"\1 are"),
    # Is / has (ambiguous — default to "is" as more common)
    (re.compile(r"\b(he|she|it|that|there|who|what|where|when|how)'?s\b", re.I), r"\1 is"),
    # I'm
    (re.compile(r"\bi'?m\b", re.I), "i am"),
    # Let's
    (re.compile(r"\blet'?s\b", re.I), "let us"),
]

# Possessive pattern — 's preceded by a noun-like word that isn't
# a known pronoun contraction. We KEEP these.
_PRONOUN_CONTRACTIONS = {
    "he", "she", "it", "that", "there", "who", "what",
    "where", "when", "how", "let",
}


def expand_contractions(text: str) -> str:
    """Expand English contractions while preserving possessives.

    Examples:
        >>> expand_contractions("I won't go")
        "I will not go"
        >>> expand_contractions("the dog's bowl")
        "the dog's bowl"
        >>> expand_contractions("she's running")
        "she is running"
    """
    for pattern, replacement in CONTRACTION_MAP:
        text = pattern.sub(replacement, text)
    return text

#!/usr/bin/env python
"""Post-parse cleaner for v13 wrapped constituents.

Input: newline-delimited wrapped constituents (``<TAG> ... </TAG>``)
from the Stanza fan-out parse.  Output: same format, with:

  1. Gutenberg / license / transcriber boilerplate lines dropped.
  2. Archaic apostrophe-d forms (``crown 'd``, ``o'er``, etc.) dropped.
  3. URLs / email addresses dropped.
  4. Exact-duplicate lines deduplicated.
  5. Lines whose inner content would be too short post-strip dropped.

Streaming I/O — runs in a few minutes on 50M lines.

Usage::

    poetry run python scripts/clean_constituents.py \\
        --input  data/datasets/english-constituents-v13/constituents.txt \\
        --output data/datasets/english-constituents-v13/constituents_clean.txt
"""

from __future__ import annotations

import argparse
import logging
import re
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger(__name__)


# ── Filters ────────────────────────────────────────────────────────────
_TAG_RE = re.compile(r"^<(\w+)>\s+(.*?)\s+</\1>$")

# Gutenberg / license boilerplate — any match on the inner content
# drops the whole constituent.
_BOILERPLATE_PATTERNS = [
    r"\bproject gutenberg\b",
    r"\bat no cost and with almost no restrictions\b",
    r"\bgive it away or re\s*-?\s*use it\b",
    r"\btranscriber'?s? note\b",
    r"\betext prepared by\b",
    r"\b(?:e\s*text|e\s*book|ebook)\s+(?:prepared|distributed|provided)\b",
    r"\bpage numbers enclosed\b",
    r"\benclosed by curly\b",
    r"\bthis (?:e[-\s]?book|etext|ebook) is for the use of\b",
    r"\bonline distributed proofreading\b",
    r"\bpublic domain\b.*\b(work|text|book)\b",
    r"\beditor'?s? note\b",
    r"\b(?:copyright|licen[sc]e).*\b(?:terms|agreement|information)\b",
    r"\bwww\.\w+|https?://|@\w+\.(?:com|org|net|edu|io)\b",
    r"\banyone anywhere at no cost\b",
    r"\bincluded in the (?:ebook|etext)\b",
    r"\bredistribute this file\b",
    r"\bstart of (?:the|this) project\b",
    r"\bend of (?:the|this) project\b",
]
_RE_BOILERPLATE = re.compile("|".join(_BOILERPLATE_PATTERNS), re.IGNORECASE)

# Archaic apostrophe-d: "crown 'd", "o'er", "e'en", "'em", "'tis", "'twas".
# If any of these appear as standalone tokens, the line is poetic/archaic
# and not representative of modern English prose — drop.
_RE_ARCHAIC = re.compile(
    r"(?:(?<=\s)|^)(?:'d|'em|'er|o'er|e'er|'neath|'tis|'twas|'ere|'tween)"
    r"(?=\s|$)",
    re.IGNORECASE,
)

# URL / email residue that slipped through normalization.
_RE_URL_EMAIL = re.compile(
    r"(?:https?://|www\.|[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,})",
)

# Common English prepositions — used to validate PP constituents
# (Stanza sometimes mislabels numeric sequences or fragments as PP).
_PREPOSITIONS = frozenset({
    "of", "in", "on", "at", "by", "to", "from", "with", "without",
    "for", "about", "over", "under", "above", "below", "through",
    "between", "among", "during", "before", "after", "since", "until",
    "as", "like", "against", "toward", "towards", "within", "into",
    "onto", "upon", "across", "around", "behind", "beyond", "beside",
    "near", "past", "off", "out", "via", "per", "along", "amid",
    "despite", "throughout", "unlike", "regarding", "concerning",
    "including", "excluding",
})

# Multi-spaces or stray digits-only tokens would have been filtered
# upstream but check inner content is non-trivial after trim.
_MIN_INNER_CHARS = 10
_MIN_PHRASE_TOKENS = 2   # sub-constituent must have at least 2 words
_MIN_ALPHA_FRACTION = 0.5  # <50% alphabetic tokens → drop (digit-heavy)


def _clean(line: str) -> tuple[str | None, str]:
    """Return ``(cleaned_line, reason)`` where ``cleaned_line`` is
    either the kept line or ``None`` if dropped.  ``reason`` is an
    identifier for bookkeeping.
    """
    m = _TAG_RE.match(line)
    if m is None:
        return None, "malformed"
    label, inner = m.group(1), m.group(2)
    inner_lc = inner.lower()

    if _RE_BOILERPLATE.search(inner_lc):
        return None, "boilerplate"
    if _RE_ARCHAIC.search(inner):
        return None, "archaic"
    if _RE_URL_EMAIL.search(inner):
        return None, "url_email"
    if len(inner) < _MIN_INNER_CHARS:
        return None, "too_short"
    if not (inner[0].isalnum() and inner[-1].isalnum()):
        return None, "edge_punct"

    # Structural validity checks on the parsed content itself.
    tokens = inner.split()
    if len(tokens) < _MIN_PHRASE_TOKENS:
        return None, "too_few_tokens"
    # A PP must start with a real English preposition, not a digit or
    # random word — Stanza mislabels numeric sequences as PP.
    if label == "PP" and tokens[0].lower() not in _PREPOSITIONS:
        return None, "pp_no_prep"
    # Drop digit-dominated constituents (sports scores, stats).
    n_alpha = sum(1 for t in tokens if any(c.isalpha() for c in t))
    if n_alpha / len(tokens) < _MIN_ALPHA_FRACTION:
        return None, "digit_heavy"
    return line, "kept"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--no-dedupe", action="store_true",
                    help="Skip exact-duplicate removal")
    ap.add_argument("--report-every", type=int, default=1_000_000)
    args = ap.parse_args()

    reasons: dict[str, int] = {}
    seen: set[int] = set()  # hash(line) for dedupe
    n_in = 0
    n_out = 0

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.input.open() as f_in, args.output.open("w") as f_out:
        for line in f_in:
            n_in += 1
            line = line.rstrip("\n")
            cleaned, reason = _clean(line)
            if cleaned is None:
                reasons[reason] = reasons.get(reason, 0) + 1
                continue
            if not args.no_dedupe:
                h = hash(cleaned)
                if h in seen:
                    reasons["duplicate"] = reasons.get("duplicate", 0) + 1
                    continue
                seen.add(h)
            f_out.write(cleaned + "\n")
            n_out += 1
            if n_in % args.report_every == 0:
                logger.info(
                    "%d in → %d out (%.1f%% kept, %s)",
                    n_in, n_out, 100 * n_out / n_in,
                    ", ".join(f"{r}={c}" for r, c in sorted(reasons.items())),
                )

    logger.info(
        "done: %d in → %d out (%.1f%% kept)",
        n_in, n_out, 100 * n_out / max(n_in, 1),
    )
    logger.info("drops by reason:")
    for r, c in sorted(reasons.items(), key=lambda kv: -kv[1]):
        logger.info("  %-12s %10d  (%.2f%%)", r, c, 100 * c / max(n_in, 1))


if __name__ == "__main__":
    main()

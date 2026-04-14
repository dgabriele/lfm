"""Clean incidental junk from the English constituents corpus.

Reads ``data/datasets/english-diverse-constituents/constituents.txt``,
applies a principled normalization pass, and writes the cleaned result
back in place (with a .bak backup of the original).

Normalizations (in order):
  1. Strip HTML tags (``<i>``, ``</p>``, etc.) — preserves inner text.
  2. Decode common HTML entities (``&amp;`` → ``&``, ``&nbsp;`` → space).
  3. Remove citation-style bracket noise (``[1]``, ``[citation needed]``,
     ``[a]``, etc.) — square-bracketed content that doesn't look like
     normal prose.
  4. Collapse dash variants: ``–`` (en-dash), ``—`` (em-dash), ``−``
     (minus) → ``-``.
  5. Collapse curly quotes: ``’`` → ``'``, ``“``/``”`` → ``"``,
     ``‘`` → ``'``.
  6. Strip wikitext ``@-@`` / ``@.@`` / ``@,@`` artefacts (shouldn't
     survive the original build, but belt and suspenders).
  7. Drop control characters other than space/tab (line ending will be
     re-added on write).
  8. Drop stray underscores, backslashes, curly braces (Markdown /
     LaTeX / code leftovers — rare in real prose).
  9. Collapse multi-whitespace → single space; strip.
  10. Drop the `literal "unknown"` placeholder token (parser failure
      marker — a few thousand of these exist).
  11. Drop any line that becomes empty or < MIN_LEN chars post-cleaning.

This is also backported into ``build_english_constituents.py`` so future
generations produce clean output directly.
"""

from __future__ import annotations

import argparse
import html
import logging
import re
import shutil
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger(__name__)

MIN_LEN = 10

_RE_HTML_TAG = re.compile(r"<[^>]+>")
_RE_BRACKET_NOISE = re.compile(r"\[[^\[\]]*\]")
_RE_WIKI_AT = re.compile(r"@([-.,])@")
_RE_CONTROL = re.compile(r"[\x00-\x08\x0b-\x1f\x7f]")
_RE_MULTISPACE = re.compile(r"\s+")
_RE_STRAY = re.compile(r"[_\\{}]")
_RE_UNKNOWN_TOK = re.compile(r"\bunknown\b", re.IGNORECASE)

# Anything outside basic Latin + Latin-1 supplement + Latin Extended-A/B
# (covers ASCII, diacritics, accented names like "Café", "Dvořák", "Senjō")
# but drops CJK, Cyrillic, Greek, Arabic, Hebrew, Thai, Devanagari, IPA
# stress/length/schwa, dingbats, etc.  This is a line-level filter: we
# drop the whole line rather than strip inline, since stripping leaves
# malformed residue like "( , lit . )".
_RE_NON_LATIN = re.compile(r"[^\x00-\x7F\u00A0-\u024F]")

_CURLY_QUOTES = {
    "\u2018": "'", "\u2019": "'",
    "\u201c": '"', "\u201d": '"',
    "\u02bc": "'",
}
_DASHES = {
    "\u2013": "-", "\u2014": "-", "\u2212": "-",
}
_REPLACEMENTS = {**_CURLY_QUOTES, **_DASHES}


def clean_line(line: str) -> str:
    s = line.rstrip("\n\r")
    s = _RE_HTML_TAG.sub("", s)
    s = html.unescape(s)
    s = _RE_BRACKET_NOISE.sub("", s)
    s = _RE_WIKI_AT.sub(r"\1", s)
    for src, tgt in _REPLACEMENTS.items():
        if src in s:
            s = s.replace(src, tgt)
    s = _RE_CONTROL.sub(" ", s)
    s = _RE_STRAY.sub(" ", s)
    # Drop "unknown" filler sentences entirely (parser hallucinations).
    # We don't want the LLM learning this token; safer to drop the whole
    # line than leave it silently in the corpus.
    if _RE_UNKNOWN_TOK.search(s):
        return ""
    # Drop lines containing non-Latin scripts (CJK, Cyrillic, Greek, etc.)
    # or stray IPA symbols — orthographic English only.
    if _RE_NON_LATIN.search(s):
        return ""
    s = _RE_MULTISPACE.sub(" ", s).strip()
    return s


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--path", type=Path,
        default=Path("data/datasets/english-diverse-constituents/constituents.txt"),
    )
    ap.add_argument("--inplace", action="store_true",
                    help="Replace the file in place (backup saved as .bak)")
    ap.add_argument("--out", type=Path, default=None,
                    help="If --inplace is NOT set, write cleaned output here")
    args = ap.parse_args()

    if args.inplace:
        out_path = args.path
        bak = args.path.with_suffix(args.path.suffix + ".bak")
        if not bak.exists():
            logger.info(f"backing up {args.path} → {bak}")
            shutil.copy2(args.path, bak)
    else:
        if args.out is None:
            raise SystemExit("Pass either --inplace or --out <path>")
        out_path = args.out

    logger.info(f"cleaning {args.path} → {out_path}")
    n_in = 0
    n_out = 0
    n_dropped_short = 0
    n_dropped_unknown = 0
    n_dropped_non_latin = 0
    tmp_out = out_path.with_suffix(out_path.suffix + ".tmp")
    with args.path.open() as f_in, tmp_out.open("w") as f_out:
        for line in f_in:
            n_in += 1
            cleaned = clean_line(line)
            if not cleaned:
                if _RE_UNKNOWN_TOK.search(line):
                    n_dropped_unknown += 1
                elif _RE_NON_LATIN.search(line):
                    n_dropped_non_latin += 1
                else:
                    n_dropped_short += 1
                continue
            if len(cleaned) < MIN_LEN:
                n_dropped_short += 1
                continue
            f_out.write(cleaned + "\n")
            n_out += 1
            if n_in % 1_000_000 == 0:
                logger.info(f"  {n_in:,} in  {n_out:,} out  "
                            f"{n_dropped_short:,} short  "
                            f"{n_dropped_unknown:,} unk  "
                            f"{n_dropped_non_latin:,} non-latin")
    shutil.move(tmp_out, out_path)
    logger.info(
        f"done. in={n_in:,}  out={n_out:,}  "
        f"dropped short={n_dropped_short:,}  "
        f"unknown={n_dropped_unknown:,}  "
        f"non-latin={n_dropped_non_latin:,}",
    )


if __name__ == "__main__":
    main()

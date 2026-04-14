#!/usr/bin/env python
"""Build a v8-scale English constituent dataset from diverse Wikipedia text.

Source: ``data/translator/english_paragraphs.txt`` — wikitext-style
Wikipedia article paragraphs (history, science, sports, music, etc.,
~750K paragraphs, ~500MB).  Wikipedia gives broad topical diversity even
though it's a single source-type (encyclopedia).

Pipeline:
  1. Sentence-segment paragraphs with NLTK punkt → flat sentence list.
  2. Sanitize: drop short / digit-heavy / wikitext-noise sentences.
  3. Constituency-parse each sentence in parallel; flatten constituent tree.
  4. Write constituents one-per-line to a text file (downstream transcoder
     converts to v9.5 token ids).

This stops short of v9.5 transcoding so we can iterate on the constituent
text without re-parsing.
"""

from __future__ import annotations

import argparse
import logging
import re
import time
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger(__name__)

SOURCE = Path("data/translator/english_paragraphs.txt")
OUT_DIR = Path("data/datasets/english-diverse-constituents")
OUT_SENTENCES = OUT_DIR / "sentences.txt"
OUT_CONSTITUENTS = OUT_DIR / "constituents.txt"

MIN_LEN = 25
MAX_LEN = 400
MIN_ALPHA_FRAC = 0.75


# Wikitext markup leftovers ("@-@", "@.@", "@,@") — replace with the
# bare punctuation they wrap.
_WIKI_AT = re.compile(r"@([\-.,])@")
_MULTI_SPACE = re.compile(r"\s+")
_HTML_TAG = re.compile(r"<[^>]+>")
_BRACKET_NOISE = re.compile(r"\[[^\[\]]*\]")
_CONTROL = re.compile(r"[\x00-\x08\x0b-\x1f\x7f]")
_STRAY = re.compile(r"[_\\{}]")
_UNKNOWN_TOK = re.compile(r"\bunknown\b", re.IGNORECASE)
_NON_LATIN = re.compile(r"[^\x00-\x7F\u00A0-\u024F]")
_CHAR_MAP = {
    "\u2013": "-", "\u2014": "-", "\u2212": "-",
    "\u2018": "'", "\u2019": "'", "\u02bc": "'",
    "\u201c": '"', "\u201d": '"',
}


def clean_wikitext(s: str) -> str:
    """Apply a full normalization pass.  Returns '' if the line should be
    dropped entirely (e.g. contains the parser's 'unknown' placeholder).
    """
    import html
    s = _HTML_TAG.sub("", s)
    s = html.unescape(s)
    s = _BRACKET_NOISE.sub("", s)
    s = _WIKI_AT.sub(r"\1", s)
    for src, tgt in _CHAR_MAP.items():
        if src in s:
            s = s.replace(src, tgt)
    s = _CONTROL.sub(" ", s)
    s = _STRAY.sub(" ", s)
    if _UNKNOWN_TOK.search(s):
        return ""
    if _NON_LATIN.search(s):
        return ""
    s = _MULTI_SPACE.sub(" ", s)
    return s.strip()


def is_acceptable(s: str) -> bool:
    if not (MIN_LEN <= len(s) <= MAX_LEN):
        return False
    alpha = sum(c.isalpha() or c.isspace() for c in s)
    if alpha / len(s) < MIN_ALPHA_FRAC:
        return False
    return True


def step1_sentences(args: argparse.Namespace) -> list[str]:
    """Sentence-segment + sanitize the paragraph corpus."""
    import nltk
    try:
        nltk.data.find("tokenizers/punkt")
    except LookupError:
        nltk.download("punkt", quiet=True)
    try:
        nltk.data.find("tokenizers/punkt_tab")
    except LookupError:
        nltk.download("punkt_tab", quiet=True)
    from nltk.tokenize import sent_tokenize

    logger.info(f"reading {SOURCE}")
    sents: list[str] = []
    n_para = 0
    with SOURCE.open() as f:
        for i, line in enumerate(f):
            if args.max_paragraphs and n_para >= args.max_paragraphs:
                break
            line = line.strip()
            if not line:
                continue
            n_para += 1
            cleaned = clean_wikitext(line)
            for s in sent_tokenize(cleaned):
                s = s.strip()
                if is_acceptable(s):
                    sents.append(s)
            if n_para % 50_000 == 0:
                logger.info(f"  scanned {n_para:,} paragraphs → {len(sents):,} sentences")
    logger.info(f"{n_para:,} paragraphs → {len(sents):,} sentences (after sanitize)")
    OUT_SENTENCES.parent.mkdir(parents=True, exist_ok=True)
    OUT_SENTENCES.write_text("\n".join(sents))
    logger.info(f"wrote {OUT_SENTENCES}")
    return sents


def step2_constituents(sents: list[str], args: argparse.Namespace) -> None:
    """Parse and write constituents (one per line) — chunked for parallelism.

    The library's ``extract_constituents_parallel`` groups by language, so
    single-language input yields a single worker.  We bypass that here by
    chunking the English sentences into N sub-batches and running the
    per-language worker in parallel processes (one Stanza CPU pipeline per
    chunk).  CPU pipelines avoid the GPU-memory ceiling that limits
    GPU mode to ~2 workers.
    """
    if args.max_sentences:
        sents = sents[: args.max_sentences]

    from lfm.data.constituents import extract_constituents_parallel

    n = len(sents)
    samples = [("eng", s) for s in sents]
    logger.info(f"running constituency extraction on {n:,} sentences (single GPU worker)")
    t0 = time.time()
    results = extract_constituents_parallel(samples, min_length=10)
    elapsed = time.time() - t0
    logger.info(
        f"  parsed in {elapsed/60:.1f} min → {len(results):,} constituents "
        f"({n / max(elapsed, 1e-9):.1f} sents/sec)",
    )

    # Drop the original full sentences (parent_seq=-1, label='S') if you only
    # want sub-constituents — but for VAE training we typically keep both.
    n_full = sum(1 for r in results if r[3] == -1)
    n_subs = len(results) - n_full
    logger.info(f"  full sentences: {n_full:,}, sub-constituents: {n_subs:,}")

    with OUT_CONSTITUENTS.open("w") as f:
        for lang, text, label, parent in results:
            f.write(f"{text}\n")
    logger.info(f"wrote {OUT_CONSTITUENTS}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-paragraphs", type=int, default=None,
                    help="Cap source paragraphs (dev only)")
    ap.add_argument("--max-sentences", type=int, default=None,
                    help="Cap parsed sentences (dev only)")
    ap.add_argument("--num-workers", type=int, default=None,
                    help="Constituency parser workers (default: 90% of CPU)")
    ap.add_argument("--skip-sentences", action="store_true",
                    help="Reuse existing sentences.txt; skip step 1")
    args = ap.parse_args()

    if args.skip_sentences and OUT_SENTENCES.exists():
        logger.info(f"reusing existing {OUT_SENTENCES}")
        sents = OUT_SENTENCES.read_text().splitlines()
        logger.info(f"  {len(sents):,} sentences loaded")
    else:
        sents = step1_sentences(args)
    step2_constituents(sents, args)


if __name__ == "__main__":
    main()

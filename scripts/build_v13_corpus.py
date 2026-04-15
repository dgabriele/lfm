#!/usr/bin/env python
"""Build the v13 constituent corpus.

v13 vs v12:
  * Multi-source: Wikipedia paragraphs + Project Gutenberg non-fiction
    (heuristically filtered for prose) + arXiv abstracts.
  * Stricter per-sentence filters: no quote-containing sentences, no
    dialogue-verb starts, parenthetical content extracted to separate
    samples.
  * Contractions expanded (don't → do not, won't → will not, etc.).
  * Possessive 's merged onto preceding word.
  * Lowercased everything.
  * Phrase-type special tokens wrap each constituent: ``<NP> ... </NP>``,
    ``<VP>``, ``<PP>``, ``<S>``, ``<SBAR>``, ``<CP>`` etc.
  * Standalone punctuation tokens stripped from constituent content.
  * SPM 10K vocab with phrase-type tags declared as
    ``user_defined_symbols``.

Output::
    data/datasets/english-constituents-v13/
        samples.h5    (ipa field = wrapped+lowercased constituent)
        spm.model / .vocab
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import re
import sys
from pathlib import Path

import h5py
import numpy as np
import sentencepiece as spm

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────────────────────
OUT_DIR = Path("data/datasets/english-constituents-v13")
SPM_MODEL_PREFIX = OUT_DIR / "spm"
OUT_H5 = OUT_DIR / "samples.h5"
OUT_SENTENCES = OUT_DIR / "sentences.txt"

SPM_VOCAB_SIZE = 10_000
SPM_TRAIN_LINES = 750_000

PHRASE_TAGS = [
    "<S>", "</S>", "<NP>", "</NP>", "<VP>", "</VP>",
    "<PP>", "</PP>", "<CP>", "</CP>", "<SBAR>", "</SBAR>",
    "<ADJP>", "</ADJP>", "<ADVP>", "</ADVP>",
]

MIN_LEN_CHARS = 20
MAX_LEN_CHARS = 400
MIN_ALPHA_FRAC = 0.75

# ──────────────────────────────────────────────────────────────────────
# Normalization
# ──────────────────────────────────────────────────────────────────────
_RE_HTML_TAG = re.compile(r"<[^>]+>")
_RE_BRACKET_NOISE = re.compile(r"\[[^\[\]]*\]")
_RE_PAREN = re.compile(r"\(([^()]{1,200})\)")
_RE_EMDASH_ASIDE = re.compile(r"\s+—\s+([^—]{5,100})\s+—\s+")
_RE_WIKI_AT = re.compile(r"@([-.,])@")
_RE_CONTROL = re.compile(r"[\x00-\x08\x0b-\x1f\x7f]")
_RE_MULTISPACE = re.compile(r"\s+")
_RE_STRAY = re.compile(r"[_\\{}]")
_RE_UNKNOWN = re.compile(r"\bunknown\b", re.IGNORECASE)
# Allow basic Latin + Latin-1 + Latin Extended-A/B; drop everything else
_RE_NON_LATIN = re.compile(r"[^\x00-\x7F\u00A0-\u024F]")
_RE_GUTENBERG_START = re.compile(
    r"\*\*\*\s*START OF (?:THE|THIS) PROJECT GUTENBERG[^\n]*\*\*\*", re.IGNORECASE,
)
_RE_GUTENBERG_END = re.compile(
    r"\*\*\*\s*END OF (?:THE|THIS) PROJECT GUTENBERG[^\n]*\*\*\*", re.IGNORECASE,
)

_CHAR_MAP = {
    "\u2013": "-", "\u2014": "-", "\u2212": "-",
    "\u2018": "'", "\u2019": "'", "\u02bc": "'",
    "\u201c": '"', "\u201d": '"',
}


def _normalize(text: str) -> str:
    text = _RE_HTML_TAG.sub("", text)
    import html
    text = html.unescape(text)
    text = _RE_BRACKET_NOISE.sub("", text)
    text = _RE_WIKI_AT.sub(r"\1", text)
    for src, tgt in _CHAR_MAP.items():
        if src in text:
            text = text.replace(src, tgt)
    text = _RE_CONTROL.sub(" ", text)
    text = _RE_STRAY.sub(" ", text)
    text = _RE_MULTISPACE.sub(" ", text).strip()
    return text


# ──────────────────────────────────────────────────────────────────────
# Contraction expansion
# ──────────────────────────────────────────────────────────────────────
_FIXED_EXPANSIONS = {
    "ain't": "is not", "aren't": "are not",
    "isn't": "is not", "wasn't": "was not", "weren't": "were not",
    "haven't": "have not", "hasn't": "has not", "hadn't": "had not",
    "don't": "do not", "doesn't": "does not", "didn't": "did not",
    "can't": "cannot", "cannot": "cannot",
    "couldn't": "could not", "shouldn't": "should not",
    "wouldn't": "would not", "mightn't": "might not",
    "mustn't": "must not", "needn't": "need not",
    "won't": "will not", "shan't": "shall not",
    "shouldnt": "should not",  # occasional rogue missing apostrophe
    "it's": "it is", "that's": "that is", "there's": "there is",
    "here's": "here is", "what's": "what is", "who's": "who is",
    "he's": "he is", "she's": "she is",
    "where's": "where is", "when's": "when is", "how's": "how is",
    "i'm": "i am", "you're": "you are", "we're": "we are", "they're": "they are",
    "i've": "i have", "you've": "you have", "we've": "we have", "they've": "they have",
    "i'll": "i will", "you'll": "you will", "he'll": "he will",
    "she'll": "she will", "it'll": "it will",
    "we'll": "we will", "they'll": "they will",
    "i'd": "i would", "you'd": "you would", "he'd": "he would",
    "she'd": "she would", "it'd": "it would",
    "we'd": "we would", "they'd": "they would",
    "let's": "let us",
    "'tis": "it is", "'twas": "it was",
    "o'clock": "o'clock",  # keep
}
_CONTRACTION_RE = re.compile(
    r"\b(" + "|".join(re.escape(k) for k in _FIXED_EXPANSIONS) + r")\b",
    re.IGNORECASE,
)


def expand_contractions(text: str) -> str:
    """Expand common English contractions on lowercased text."""
    return _CONTRACTION_RE.sub(
        lambda m: _FIXED_EXPANSIONS[m.group(0).lower()], text,
    )


# ──────────────────────────────────────────────────────────────────────
# Sentence-level filters
# ──────────────────────────────────────────────────────────────────────
_DIALOGUE_VERBS = {
    "said", "says", "asked", "asks", "replied", "replies",
    "shouted", "whispered", "murmured", "exclaimed",
    "cried", "muttered", "answered", "responded",
    "added", "continued",
}


def _is_acceptable_sentence(s: str) -> bool:
    if not (MIN_LEN_CHARS <= len(s) <= MAX_LEN_CHARS):
        return False
    # Must end with sentence-terminal punctuation (we strip it later)
    if not s.rstrip().endswith((".", "?", "!")):
        return False
    # Quote pairs — dialogue indicator
    if s.count('"') >= 2:
        return False
    # Direct-speech verb near start — narrative dialogue
    head_words = [w.strip(".,;:'\"").lower() for w in s.split()[:5]]
    if any(w in _DIALOGUE_VERBS for w in head_words):
        return False
    # Alpha fraction
    if not s:
        return False
    alpha = sum(c.isalpha() or c.isspace() for c in s)
    if alpha / len(s) < MIN_ALPHA_FRAC:
        return False
    # Non-Latin
    if _RE_NON_LATIN.search(s):
        return False
    if _RE_UNKNOWN.search(s):
        return False
    return True


# ──────────────────────────────────────────────────────────────────────
# Parenthetical extraction
# ──────────────────────────────────────────────────────────────────────
def extract_parentheticals(sentence: str) -> tuple[str, list[str]]:
    """Return (host_with_parens_stripped, [extracted_inner_strings])."""
    inners: list[str] = []

    def _grab(m: re.Match) -> str:
        inner = m.group(1).strip()
        if 20 <= len(inner) <= 200:
            inners.append(inner)
        return " "

    host = _RE_PAREN.sub(_grab, sentence)
    host = _RE_EMDASH_ASIDE.sub(_grab, host)
    host = _RE_MULTISPACE.sub(" ", host).strip()
    return host, inners


# ──────────────────────────────────────────────────────────────────────
# Gutenberg cleanup
# ──────────────────────────────────────────────────────────────────────
def strip_gutenberg_boilerplate(text: str) -> str:
    """Keep only content between Gutenberg's START/END markers."""
    m_start = _RE_GUTENBERG_START.search(text)
    m_end = _RE_GUTENBERG_END.search(text)
    if m_start:
        text = text[m_start.end():]
    if m_end:
        text = text[: m_end.start()] if m_start else text
    return text


def _is_likely_nonfiction(text: str) -> bool:
    """Heuristic: filter out fiction by dialogue density + first-person use.

    Fiction typically has many quoted utterances and heavy I/me usage.
    Non-fiction prose (philosophy, science, biography summary) is third-
    person-dominant with few quotation marks.
    """
    if len(text) < 2000:
        return False
    per_1k = 1000.0 / len(text)
    # High rate of double-quote → dialogue-heavy
    n_quote = text.count('"')
    if n_quote * per_1k > 4:
        return False
    # Count " I " (capitalized pronoun) — an overly-first-person book is
    # likely a memoir or letter collection with dialogue.  Allow some.
    n_i = len(re.findall(r"\bI\b", text))
    if n_i * per_1k > 12:
        return False
    return True


# ──────────────────────────────────────────────────────────────────────
# Source loaders
# ──────────────────────────────────────────────────────────────────────
def load_wikipedia_paragraphs(path: Path, limit: int | None) -> list[str]:
    logger.info("loading wikipedia from %s", path)
    lines: list[str] = []
    with path.open() as f:
        for i, line in enumerate(f):
            if limit and i >= limit:
                break
            s = line.strip()
            if s:
                lines.append(s)
    logger.info("  wikipedia: %d paragraphs", len(lines))
    return lines


def load_gutenberg_nonfiction(
    path: Path, limit: int | None,
) -> list[str]:
    """Iterate the cached jsonl and keep likely-non-fiction books."""
    logger.info("loading gutenberg from %s", path)
    paragraphs: list[str] = []
    n_books_read = 0
    n_books_kept = 0
    with path.open() as f:
        for line in f:
            if limit and len(paragraphs) >= limit:
                break
            try:
                rec = json.loads(line)
            except Exception:
                continue
            text = rec.get("text", "")
            if not text:
                continue
            text = strip_gutenberg_boilerplate(text)
            n_books_read += 1
            if not _is_likely_nonfiction(text):
                continue
            n_books_kept += 1
            # Split into paragraphs on blank lines
            for para in re.split(r"\n\s*\n", text):
                para = _RE_MULTISPACE.sub(" ", para).strip()
                if len(para) >= 200:  # reasonable prose paragraph
                    paragraphs.append(para)
    logger.info(
        "  gutenberg: %d books scanned, %d non-fiction kept, %d paragraphs",
        n_books_read, n_books_kept, len(paragraphs),
    )
    return paragraphs


def load_arxiv_abstracts(limit: int | None) -> list[str]:
    """Stream arXiv abstracts from the HF `ccdv/arxiv-summarization` dataset.

    Short abstracts are good dense prose samples.
    """
    logger.info("loading arxiv abstracts from HF (streaming)")
    try:
        from datasets import load_dataset
    except ImportError:
        logger.warning("datasets not installed, skipping arxiv")
        return []
    ds = load_dataset(
        "ccdv/arxiv-summarization", split="train", streaming=True,
    )
    out: list[str] = []
    for i, ex in enumerate(ds):
        if limit and i >= limit:
            break
        abstract = ex.get("abstract") or ex.get("summary") or ""
        if isinstance(abstract, list):
            abstract = " ".join(abstract)
        abstract = _RE_MULTISPACE.sub(" ", abstract).strip()
        if len(abstract) >= 200:
            out.append(abstract)
    logger.info("  arxiv: %d abstracts", len(out))
    return out


# ──────────────────────────────────────────────────────────────────────
# Main sentence pipeline
# ──────────────────────────────────────────────────────────────────────
def paragraphs_to_sentences(
    paragraphs: list[str], source_tag: str,
) -> list[str]:
    """Sentence-segment, normalize, filter, extract parentheticals, expand
    contractions, lowercase.  Returns a list of candidate sentences
    (still with terminal punctuation for downstream parser).
    """
    import nltk
    try:
        nltk.data.find("tokenizers/punkt_tab")
    except LookupError:
        nltk.download("punkt_tab", quiet=True)
    from nltk.tokenize import sent_tokenize

    sents: list[str] = []
    for i, para in enumerate(paragraphs):
        text = _normalize(para)
        for s in sent_tokenize(text):
            host, inners = extract_parentheticals(s)
            # Process host
            for cand in [host, *inners]:
                cand = cand.strip()
                # Ensure terminal punctuation (inners may lack it)
                if not cand.endswith((".", "?", "!")):
                    cand = cand + "."
                cand = expand_contractions(cand.lower())
                # Re-filter after lowercase+expansion
                if _is_acceptable_sentence(cand):
                    sents.append(cand)
        if (i + 1) % 100_000 == 0:
            logger.info(
                "  %s: %d paragraphs scanned → %d sentences",
                source_tag, i + 1, len(sents),
            )
    logger.info("  %s final: %d sentences", source_tag, len(sents))
    return sents


# ──────────────────────────────────────────────────────────────────────
# Constituent wrapping
# ──────────────────────────────────────────────────────────────────────
_ALNUM_RE = re.compile(r"\w")


def _tokenize_keep_words(text: str) -> list[str]:
    """Split on whitespace, drop tokens that have no alphanumerics, and
    merge standalone possessive ``'s`` / ``s'`` back onto the previous
    token so constituents never begin or end with an apostrophe chunk.
    """
    raw = text.split()
    out: list[str] = []
    for tok in raw:
        if tok in ("'s", "s'", "'"):
            if out:
                out[-1] = out[-1] + tok
            continue
        # Must contain at least one letter or digit
        if not _ALNUM_RE.search(tok):
            continue
        out.append(tok)
    return out


def _clean_constituent_content(text: str) -> str | None:
    toks = _tokenize_keep_words(text)
    if not toks:
        return None
    content = " ".join(toks)
    # First and last character must be alphanumeric after cleaning
    if not (content[0].isalnum() and content[-1].isalnum()):
        return None
    if len(content) < 3:
        return None
    return content


_TAG_MAP = {
    "S": ("<S>", "</S>"),
    "NP": ("<NP>", "</NP>"),
    "VP": ("<VP>", "</VP>"),
    "PP": ("<PP>", "</PP>"),
    "SBAR": ("<SBAR>", "</SBAR>"),
    "CP": ("<CP>", "</CP>"),
    "ADJP": ("<ADJP>", "</ADJP>"),
    "ADVP": ("<ADVP>", "</ADVP>"),
}


def wrap_constituent(text: str, label: str) -> str | None:
    """Wrap ``text`` with ``<LABEL> ... </LABEL>`` if supported.
    Return None if unsupported label, or cleaned content fails."""
    tag_open, tag_close = _TAG_MAP.get(label, (None, None))
    if tag_open is None:
        return None
    cleaned = _clean_constituent_content(text)
    if cleaned is None:
        return None
    return f"{tag_open} {cleaned} {tag_close}"


# ──────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--wikipedia", type=Path,
                    default=Path("data/translator/english_paragraphs.txt"))
    ap.add_argument("--gutenberg", type=Path,
                    default=Path("data/qwen_targets_cache/gutenberg-en.jsonl"))
    ap.add_argument("--wiki-limit", type=int, default=None)
    ap.add_argument("--gutenberg-limit", type=int, default=200_000,
                    help="Cap gutenberg paragraphs (not books).")
    ap.add_argument("--arxiv-limit", type=int, default=300_000,
                    help="Cap arxiv abstracts.")
    ap.add_argument("--skip-arxiv", action="store_true")
    ap.add_argument("--skip-gutenberg", action="store_true")
    ap.add_argument("--skip-sentences", action="store_true",
                    help="Reuse existing sentences.txt")
    ap.add_argument("--skip-parse", action="store_true",
                    help="Reuse existing constituents.txt")
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # ── 1. Gather + segment sentences from all sources ───────────────
    if args.skip_sentences and OUT_SENTENCES.exists():
        logger.info("reusing existing sentences from %s", OUT_SENTENCES)
        sentences = OUT_SENTENCES.read_text().splitlines()
        logger.info("  %d sentences loaded", len(sentences))
    else:
        all_sentences: list[str] = []
        # Wikipedia
        wiki_paras = load_wikipedia_paragraphs(args.wikipedia, args.wiki_limit)
        all_sentences.extend(paragraphs_to_sentences(wiki_paras, "wiki"))
        del wiki_paras
        # Gutenberg
        if not args.skip_gutenberg and args.gutenberg.exists():
            gut_paras = load_gutenberg_nonfiction(args.gutenberg, args.gutenberg_limit)
            all_sentences.extend(paragraphs_to_sentences(gut_paras, "gutenberg"))
            del gut_paras
        # arXiv
        if not args.skip_arxiv:
            arx_paras = load_arxiv_abstracts(args.arxiv_limit)
            all_sentences.extend(paragraphs_to_sentences(arx_paras, "arxiv"))
            del arx_paras
        logger.info("total sentences: %d", len(all_sentences))
        OUT_SENTENCES.write_text("\n".join(all_sentences))
        sentences = all_sentences

    # ── 2. Constituency parse (parallel Stanza) ──────────────────────
    out_constituents = OUT_DIR / "constituents.txt"
    if args.skip_parse and out_constituents.exists():
        logger.info("reusing existing constituents from %s", out_constituents)
        wrapped = out_constituents.read_text().splitlines()
    else:
        import time
        from lfm.data.constituents import extract_constituents_parallel

        samples = [("eng", s) for s in sentences]
        t0 = time.time()
        logger.info("constituency extraction on %d sentences", len(samples))
        results = extract_constituents_parallel(samples, min_length=10)
        logger.info(
            "  parsed in %.1f min → %d raw constituents",
            (time.time() - t0) / 60, len(results),
        )
        # ── 3. Wrap + clean each constituent ─────────────────────────
        wrapped: list[str] = []
        skipped = 0
        for lang, text, label, _parent in results:
            w = wrap_constituent(text, label)
            if w is None:
                skipped += 1
                continue
            wrapped.append(w)
        logger.info(
            "  wrapped %d, skipped %d (label unsupported or content rejected)",
            len(wrapped), skipped,
        )
        out_constituents.write_text("\n".join(wrapped))

    # ── 4. Train SPM with phrase-tag special tokens ──────────────────
    tmp_train = OUT_DIR / "text_for_spm.txt"
    rng = random.Random(42)
    spm_train = (
        rng.sample(wrapped, SPM_TRAIN_LINES)
        if len(wrapped) > SPM_TRAIN_LINES else wrapped
    )
    logger.info("writing SPM training text (%d lines)", len(spm_train))
    tmp_train.write_text("\n".join(spm_train))

    logger.info("training SentencePiece (vocab=%d, user symbols=%d)",
                SPM_VOCAB_SIZE, len(PHRASE_TAGS))
    spm.SentencePieceTrainer.Train(
        input=str(tmp_train),
        model_prefix=str(SPM_MODEL_PREFIX),
        vocab_size=SPM_VOCAB_SIZE,
        model_type="bpe",
        character_coverage=0.9995,
        pad_id=-1,
        unk_id=0,
        bos_id=-1,
        eos_id=-1,
        normalization_rule_name="identity",
        shuffle_input_sentence=True,
        input_sentence_size=len(spm_train),
        user_defined_symbols=",".join(PHRASE_TAGS),
    )
    tmp_train.unlink()
    logger.info("wrote %s.model + .vocab", SPM_MODEL_PREFIX)

    # ── 5. Emit HDF5 in the v7/v12-compatible schema ─────────────────
    logger.info("writing %s", OUT_H5)
    str_dt = h5py.string_dtype(encoding="utf-8")
    n = len(wrapped)
    text_lengths = np.asarray([len(s) for s in wrapped], dtype=np.int32)
    logger.info(
        "constituent length stats: min=%d max=%d mean=%.1f median=%.0f p99=%.0f",
        text_lengths.min(), text_lengths.max(), text_lengths.mean(),
        np.median(text_lengths), np.percentile(text_lengths, 99),
    )
    with h5py.File(OUT_H5, "w") as h:
        g = h.create_group("samples")
        g.create_dataset("ipa", shape=(n,), dtype=str_dt, data=wrapped)
        g.create_dataset("ipa_length", shape=(n,), dtype=np.int32, data=text_lengths)
        g.create_dataset("language", shape=(n,), dtype=str_dt, data=["eng"] * n)
        g.create_dataset("raw", shape=(n,), dtype=str_dt, data=wrapped)
        g.create_dataset("seq", shape=(n,), dtype=np.int64, data=np.arange(n, dtype=np.int64))
        g.create_dataset("source", shape=(n,), dtype=str_dt, data=["constituent"] * n)
        g.create_dataset("source_file", shape=(n,), dtype=str_dt, data=["v13"] * n)
    logger.info("done.")


if __name__ == "__main__":
    main()

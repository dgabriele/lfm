"""Build a multi-sentence Phase 1 training corpus.

Pipeline:
  1. Load source paragraphs (wikitext-103 by default; --add-news to mix in
     CC-News for register diversity).
  2. Filter: skip empty lines, headers, list items; keep paragraphs with
     >= ``--min-sentences`` and within ``--min-len/--max-len`` whitespace
     tokens (proxies cipher length).
  3. Run NER + paragraph-scoped normalisation (scripts/normalize_paragraph.py).
  4. Write JSONL: ``{"text": "<normalised paragraph>"}`` one per line.

Usage:
  poetry run python scripts/build_multisent_corpus.py \\
      --target-paragraphs 1000000 \\
      --out data/multisent_corpus/passages_normalized.jsonl
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from pathlib import Path

import spacy

sys.path.insert(0, str(Path(__file__).resolve().parent))
from normalize_paragraph import normalize_paragraph  # noqa: E402

# Coarse sentence-end regex used for cheap filtering (NOT the actual
# sentence segmentation — spaCy handles that during NER).
_SENT_END_RE = re.compile(r"[.!?](?:\s|$)")
_HEADER_RE = re.compile(r"^\s*=+[^=]+=+\s*$")        # wikitext "= Title =" lines
_BULLET_RE = re.compile(r"^\s*[-*•]\s")


def _looks_like_paragraph(text: str, min_sentences: int,
                         min_len: int, max_len: int) -> bool:
    """Cheap pre-filter: paragraph-shaped multi-sentence prose only."""
    s = text.strip()
    if not s:
        return False
    if _HEADER_RE.match(s) or _BULLET_RE.match(s):
        return False
    n_words = len(s.split())
    if n_words < min_len or n_words > max_len:
        return False
    n_sents = len(_SENT_END_RE.findall(s))
    if n_sents < min_sentences:
        return False
    # Reject paragraphs that are mostly numbers / formatting (low alpha ratio)
    alpha = sum(1 for c in s if c.isalpha())
    if alpha / max(len(s), 1) < 0.5:
        return False
    # Reject paragraphs with significant non-ASCII content (transliterations,
    # foreign-script embeds — would tokenise as OOV downstream)
    non_ascii = sum(1 for c in s if ord(c) > 127)
    if non_ascii / max(len(s), 1) > 0.05:
        return False
    return True


def _iter_wikitext(min_sentences: int, min_len: int, max_len: int):
    """Yield wikitext-103 paragraphs that pass the cheap pre-filter."""
    from datasets import load_dataset
    ds = load_dataset("wikitext", "wikitext-103-raw-v1", split="train")
    for row in ds:
        text = row["text"]
        if _looks_like_paragraph(text, min_sentences, min_len, max_len):
            yield text.strip()


def _iter_cc_news(min_sentences: int, min_len: int, max_len: int):
    """Yield CC-News article paragraphs (split each article on \\n\\n)."""
    from datasets import load_dataset
    ds = load_dataset("cc_news", split="train", streaming=True)
    for row in ds:
        text = row.get("text", "")
        if not text:
            continue
        for para in text.split("\n\n"):
            if _looks_like_paragraph(para, min_sentences, min_len, max_len):
                yield para.strip()


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--out", required=True)
    p.add_argument("--target-paragraphs", type=int, default=1_000_000)
    p.add_argument("--min-sentences", type=int, default=3)
    p.add_argument("--min-len", type=int, default=40,
                   help="min whitespace-tokens per paragraph (rough cipher-len proxy)")
    p.add_argument("--max-len", type=int, default=400,
                   help="max whitespace-tokens per paragraph")
    p.add_argument("--add-news", action="store_true",
                   help="mix in CC-News (streaming) for register diversity")
    p.add_argument("--news-fraction", type=float, default=0.2)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--log-every", type=int, default=10_000)
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    log = logging.getLogger(__name__)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    log.info("loading spaCy en_core_web_sm ...")
    nlp = spacy.load("en_core_web_sm", disable=["lemmatizer", "tagger", "attribute_ruler"])
    nlp.max_length = 5_000_000  # we filter <500 words anyway

    target_news = int(args.target_paragraphs * args.news_fraction) if args.add_news else 0
    target_wiki = args.target_paragraphs - target_news
    log.info("targets: wiki=%d news=%d  (total %d)",
             target_wiki, target_news, args.target_paragraphs)

    def _process_stream(stream, target, kind, n_written, fh):
        """Single-threaded spaCy.pipe — no multiprocessing deadlocks."""
        n_local = 0
        # Pull N items from generator into a list, run nlp.pipe (single proc),
        # write results, repeat. Manual chunking avoids spaCy's generator-
        # multiprocessing deadlock at end-of-input.
        chunk: list[str] = []
        chunk_size = max(args.batch_size * 4, 256)

        def _flush(c):
            nonlocal n_local, n_written
            if not c:
                return
            for doc in nlp.pipe(c, batch_size=args.batch_size):
                text_norm = _normalize_doc(doc)
                if not text_norm:
                    continue
                fh.write(json.dumps({"text": text_norm}, ensure_ascii=False) + "\n")
                n_written += 1
                n_local += 1
                if n_written % args.log_every == 0:
                    log.info("written %d / %d  (%s=%d)",
                             n_written, args.target_paragraphs, kind, n_local)
                if n_local >= target:
                    return

        for text in stream:
            chunk.append(text)
            if len(chunk) >= chunk_size:
                _flush(chunk)
                chunk = []
                if n_local >= target:
                    return n_written, n_local
        _flush(chunk)
        return n_written, n_local

    n_written = 0
    n_wiki = 0
    n_news = 0
    with out_path.open("w") as fh:
        log.info("processing wikitext-103 ...")
        wiki_iter = _iter_wikitext(args.min_sentences, args.min_len, args.max_len)
        n_written, n_wiki = _process_stream(wiki_iter, target_wiki, "wiki", n_written, fh)

        if args.add_news and n_news < target_news:
            log.info("processing cc_news ...")
            news_iter = _iter_cc_news(args.min_sentences, args.min_len, args.max_len)
            n_written, n_news = _process_stream(news_iter, target_news, "news", n_written, fh)

    log.info("done — %d paragraphs (wiki=%d news=%d) → %s",
             n_written, n_wiki, n_news, out_path)


def _normalize_doc(doc) -> str:
    """Apply the same paragraph-scoped letter-suffix normalisation that
    ``normalize_paragraph()`` performs, but starting from a pre-parsed
    spaCy Doc instead of re-running NER. Mirrors the logic exactly.
    """
    from normalize_paragraph import (
        ENTITY_TYPE_MAP,
        LETTERS,
        _State,
        _surface_key,
    )
    state = _State()
    for ent in doc.ents:
        type_stem = ENTITY_TYPE_MAP.get(ent.label_)
        if type_stem is None:
            continue
        state.label_for(type_stem, _surface_key(ent.text))

    skip_indices: set[int] = set()
    placeholder_at: dict[int, str] = {}
    for ent in doc.ents:
        type_stem = ENTITY_TYPE_MAP.get(ent.label_)
        if type_stem is None:
            continue
        label = state.label_for(type_stem, _surface_key(ent.text))
        placeholder_at[ent.start] = label
        for i in range(ent.start + 1, ent.end):
            skip_indices.add(i)

    pieces: list[str] = []
    for i, tok in enumerate(doc):
        if i in skip_indices:
            continue
        if i in placeholder_at:
            pieces.append(placeholder_at[i])
            ent_end = next(
                (e.end for e in doc.ents if e.start == i),
                i + 1,
            )
            pieces.append(doc[ent_end - 1].whitespace_)
        else:
            pieces.append(tok.text)
            pieces.append(tok.whitespace_)
    return "".join(pieces).strip()


if __name__ == "__main__":
    main()

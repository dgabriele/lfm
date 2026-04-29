"""Replace numeric/temporal/structural entities with placeholder tokens before
cipher encoding. Eliminates the period-ambiguity problem (where decimals like
"$5.00" produce literal periods that the LM can't distinguish from sentence
endings) and reduces noise in the alien token distribution.

Uses spaCy NER (en_core_web_sm) for entity detection, with regex post-processing
for URL/email patterns spaCy doesn't tag by default.

Mapped entity types → placeholder:
    MONEY    → [MONEY]
    PERCENT  → [PERCENT]
    DATE     → [DATE]
    TIME     → [TIME]
    CARDINAL → [NUMBER]    (raw cardinal numbers, including standalone digits)
    ORDINAL  → [ORDINAL]   (1st, 2nd, third, etc.)
    QUANTITY → [QUANTITY]  (measurements: "5 km", "10 pounds")
    + regex: URLs → [URL], emails → [EMAIL]

Other spaCy entity types (PERSON, ORG, GPE, LOC, etc.) are LEFT INTACT — names
of people / organizations / places carry semantic content we want to preserve
in the alien language.

Usage:
    poetry run python scripts/normalize_corpus.py \\
        --input data/embeddings_qwen/passages.jsonl \\
        --output data/embeddings_qwen/passages_normalized.jsonl \\
        --n-process 8
"""

from __future__ import annotations

import argparse
import json
import logging
import re
from pathlib import Path

import spacy

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger(__name__)

NER_TO_PLACEHOLDER = {
    "MONEY":    "moneyamount",
    "PERCENT":  "percentvalue",
    "DATE":     "dateexpression",
    "TIME":     "timeexpression",
    "CARDINAL": "numbervalue",
    "ORDINAL":  "ordinalvalue",
    "QUANTITY": "quantityvalue",
}
URL_PLACEHOLDER   = "weburl"
EMAIL_PLACEHOLDER = "emailaddress"

# spaCy doesn't reliably tag URLs/emails as entities; catch them with regex.
URL_RE   = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
EMAIL_RE = re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b")


def normalize_text(text: str, doc) -> str:
    """Replace tagged entity spans + URL/email patterns with placeholder tokens."""
    spans = []
    for ent in doc.ents:
        ph = NER_TO_PLACEHOLDER.get(ent.label_)
        if ph is not None:
            spans.append((ent.start_char, ent.end_char, ph))
    spans.sort(key=lambda x: x[0])
    out: list[str] = []
    cursor = 0
    for start, end, ph in spans:
        if start < cursor:  # overlapping span (rare); skip the later one
            continue
        out.append(text[cursor:start])
        out.append(ph)
        cursor = end
    out.append(text[cursor:])
    text = "".join(out)
    # Order matters: emails contain @; URLs contain // — substitute both safely
    text = URL_RE.sub(URL_PLACEHOLDER, text)
    text = EMAIL_RE.sub(EMAIL_PLACEHOLDER, text)
    return text


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--n-process", type=int, default=4,
                   help="spaCy parallel processes (1 disables multiprocessing).")
    p.add_argument("--batch-size", type=int, default=128,
                   help="spaCy.pipe batch size.")
    p.add_argument("--model", default="en_core_web_sm")
    args = p.parse_args()

    logger.info("loading spaCy %s", args.model)
    nlp = spacy.load(args.model)
    # Disable components we don't need for NER speed
    for pipe in ("lemmatizer", "tagger", "attribute_ruler"):
        if pipe in nlp.pipe_names:
            nlp.disable_pipe(pipe)

    if args.input.suffix == ".jsonl":
        texts = [json.loads(l)["text"]
                 for l in args.input.read_text().splitlines() if l.strip()]
    else:
        texts = [l.strip() for l in args.input.read_text().splitlines() if l.strip()]
    logger.info("loaded %d sentences from %s", len(texts), args.input)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    counts = {ph: 0 for ph in set(NER_TO_PLACEHOLDER.values()) | {URL_PLACEHOLDER, EMAIL_PLACEHOLDER}}
    with args.output.open("w") as out:
        for i, doc in enumerate(nlp.pipe(
            texts, batch_size=args.batch_size, n_process=args.n_process,
        )):
            normalized = normalize_text(texts[i], doc)
            for ph in counts:
                counts[ph] += normalized.count(ph)
            out.write(json.dumps({"text": normalized}) + "\n")
            written += 1
            if written % 50_000 == 0:
                logger.info("normalized %d / %d", written, len(texts))

    logger.info("wrote %d normalized sentences → %s", written, args.output)
    total = sum(counts.values())
    logger.info("placeholder counts (%d total):", total)
    for ph, c in sorted(counts.items(), key=lambda x: -x[1]):
        logger.info("  %-12s %d", ph, c)


if __name__ == "__main__":
    main()

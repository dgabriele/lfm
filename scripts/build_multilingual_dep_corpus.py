#!/usr/bin/env python3
"""Build dependency-parsed IPA corpus for a single language.

Downloads Leipzig corpora, filters, converts to IPA via epitran,
parses dependencies via Stanza, and outputs JSONL in the same
format as the English dep-tree dataset.

Usage:
    poetry run python3 scripts/build_multilingual_dep_corpus.py \
        --lang ces --target 3000000 --workers 2

Supported languages (from multilingual DepTreeVAE plan):
    ces (Czech), tur (Turkish), fin (Finnish), ara (Arabic),
    jpn (Japanese), hin (Hindi), hun (Hungarian), kor (Korean),
    ind (Indonesian), vie (Vietnamese)
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import re
import time
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger(__name__)

# Language config: ISO-639-3 → (Stanza code, epitran code, script info)
LANG_CONFIG = {
    "ces": {"stanza": "cs", "epitran": "ces-Latn", "name": "Czech"},
    "tur": {"stanza": "tr", "epitran": "tur-Latn", "name": "Turkish"},
    "fin": {"stanza": "fi", "epitran": "fin-Latn", "name": "Finnish"},
    "ara": {"stanza": "ar", "epitran": "ara-Arab", "name": "Arabic"},
    "jpn": {"stanza": "ja", "epitran": "jpn-Kana", "name": "Japanese"},
    "hin": {"stanza": "hi", "epitran": "hin-Deva", "name": "Hindi"},
    "hun": {"stanza": "hu", "epitran": "hun-Latn", "name": "Hungarian"},
    "kor": {"stanza": "ko", "epitran": "kor-Hang", "name": "Korean"},
    "ind": {"stanza": "id", "epitran": "ind-Latn", "name": "Indonesian"},
    "vie": {"stanza": "vi", "epitran": "vie-Latn", "name": "Vietnamese"},
}

OUT_BASE = Path("data/datasets/multilingual-dep-trees")

# Filtering regexes
RE_URL = re.compile(r"https?://\S+|www\.\S+", re.I)
RE_EMAIL = re.compile(r"\S+@\S+\.\S+")
RE_DIGIT_HEAVY = re.compile(r"\d")
RE_QUOTED = re.compile(r'["\u201c\u201d\u00ab\u00bb\u2018\u2019]')


RE_NUM_SEQ = re.compile(r"\d+")
RE_SENTENCE_END = re.compile(r"[.!?。！？]$")
RE_STARTS_WITH_LETTER = re.compile(r"^\s*\w", re.UNICODE)
RE_BULLET_PREFIX = re.compile(r"^\s*[;•·\-–—►▪▸\*#\d]+[\s:.)]+")
RE_PARENS_HEAVY = re.compile(r"\([^)]*\)")


def filter_sentence(text: str, min_words: int = 5, max_words: int = 30) -> bool:
    """Return True if the sentence passes quality filters."""
    text = text.strip()
    words = text.split()
    if not (min_words <= len(words) <= max_words):
        return False
    if RE_URL.search(text):
        return False
    if RE_EMAIL.search(text):
        return False
    if RE_QUOTED.search(text):
        return False
    # Must start with a letter (not bullet, semicolon, number, etc.)
    if not text[0].isalpha():
        return False
    # Must end with sentence-ending punctuation
    if not RE_SENTENCE_END.search(text):
        return False
    # No bullet/list-like prefixes
    if RE_BULLET_PREFIX.match(text):
        return False
    # Too many digit sequences (>2 = likely a date list or table row)
    if len(RE_NUM_SEQ.findall(text)) > 2:
        return False
    # Too many digits (>15% of chars)
    digit_ratio = sum(1 for c in text if c.isdigit()) / max(len(text), 1)
    if digit_ratio > 0.15:
        return False
    # Too many uppercase (>40% — likely acronym soup or header)
    upper_ratio = sum(1 for c in text if c.isupper()) / max(len(text), 1)
    if upper_ratio > 0.4:
        return False
    # Too many parentheses (likely citations/references)
    if len(RE_PARENS_HEAVY.findall(text)) > 2:
        return False
    return True


# Source registry — maps filename patterns to human-readable source names + URLs
SOURCE_REGISTRY = {
    "newscrawl": ("Leipzig Newscrawl", "https://wortschatz-leipzig.de/en/download"),
    "news_": ("Leipzig News", "https://wortschatz-leipzig.de/en/download"),
    "wikipedia": ("Wikipedia", "https://dumps.wikimedia.org/"),
    "europarl": ("Europarl", "https://opus.nlpl.eu/Europarl.php"),
    "wikimatrix": ("WikiMatrix", "https://opus.nlpl.eu/WikiMatrix.php"),
}


def classify_source(filename: str) -> str:
    """Map a corpus filename to a source key."""
    fn = filename.lower()
    for key in SOURCE_REGISTRY:
        if key in fn:
            return key
    return "other"


def load_leipzig_sentences(lang_iso3: str) -> list[tuple[str, str]]:
    """Load sentences with source tags from corpus files.

    Returns list of (sentence, source_key) tuples.
    """
    corpus_dir = Path(f"data/corpora/{lang_iso3}")
    if not corpus_dir.exists():
        logger.warning(
            "No corpus directory at %s. "
            "Download Leipzig corpora and place sentence files there.",
            corpus_dir,
        )
        return []

    sentences: list[tuple[str, str]] = []
    for f in sorted(corpus_dir.glob("*-sentences.txt")):
        source_key = classify_source(f.name)
        logger.info("  loading %s (source: %s) ...", f.name, source_key)
        count = 0
        with open(f) as fh:
            for line in fh:
                parts = line.strip().split("\t")
                text = parts[1] if len(parts) >= 2 else parts[0]
                if text:
                    sentences.append((text, source_key))
                    count += 1
        logger.info("    %d sentences", count)
    logger.info("  loaded %d raw sentences for %s", len(sentences), lang_iso3)
    return sentences


def ipa_convert_batch(
    sentences: list[tuple[str, str]], epitran_code: str,
) -> list[tuple[str, str, str]]:
    """Convert sentences to IPA via epitran.

    Input: list of (sentence, source_key) tuples.
    Returns: list of (original, ipa, source_key) tuples.
    """
    import epitran

    epi = epitran.Epitran(epitran_code)
    results = []
    failed = 0
    for sent, source_key in sentences:
        try:
            ipa = epi.transliterate(sent)
            if ipa and len(ipa.split()) == len(sent.split()):
                results.append((sent, ipa, source_key))
            else:
                failed += 1
        except Exception:
            failed += 1
    if failed:
        logger.info("  IPA conversion: %d failed out of %d", failed, len(sentences))
    return results


def dep_parse_and_write(
    triples: list[tuple[str, str, str]],
    stanza_code: str,
    lang_id: int,
    out_path: Path,
    batch_size: int = 128,
    device: str = "gpu",
) -> int:
    """Dep-parse sentences and write JSONL.

    Args:
        triples: list of (original, ipa, source_key) tuples.
    """
    import stanza

    try:
        import orjson
        def dumps(obj):
            return orjson.dumps(obj, option=orjson.OPT_APPEND_NEWLINE).decode()
    except ImportError:
        def dumps(obj):
            return json.dumps(obj, ensure_ascii=False) + "\n"

    stanza.download(stanza_code, processors="tokenize,pos,lemma,depparse", verbose=False)
    nlp = stanza.Pipeline(
        stanza_code,
        processors="tokenize,pos,lemma,depparse",
        tokenize_pretokenized=True,
        use_gpu=(device == "gpu"),
        verbose=False,
    )
    logger.info("  Stanza %s pipeline loaded", stanza_code)

    failed = 0
    t0 = time.time()

    # Resume support: skip already-processed sentences
    existing = 0
    if out_path.exists():
        with open(out_path) as ef:
            existing = sum(1 for _ in ef)
        if existing > 0:
            logger.info("  Resuming: %d sentences already exist, skipping...", existing)
            # Skip the corresponding triples (approximate: 1 triple ≈ 1 output line)
            triples = triples[existing:]
            if not triples:
                logger.info("  Target already reached!")
                return existing

    mode = "a" if existing > 0 else "w"
    parsed = existing
    with open(out_path, mode) as f:
        for i in range(0, len(triples), batch_size):
            batch = triples[i : i + batch_size]
            originals = [orig for orig, _, _ in batch]
            word_lists = [orig.split() for orig in originals]

            try:
                doc = nlp(word_lists)
            except Exception as e:
                logger.warning("  batch %d failed: %s", i, e)
                failed += len(batch)
                continue

            if len(doc.sentences) != len(batch):
                failed += len(batch)
                continue

            lines = []
            for (orig, ipa, source_key), sent in zip(batch, doc.sentences):
                ipa_words = ipa.split()
                words = sent.words
                if len(words) != len(ipa_words):
                    failed += 1
                    continue

                dep_labels = [w.deprel for w in words]
                dep_heads = [w.head for w in words]
                tagged_ipa = " ".join(
                    f"[{lab}] {iw}" for lab, iw in zip(dep_labels, ipa_words)
                )
                lines.append(dumps({
                    "raw": orig,
                    "ipa": ipa,
                    "dep_labels": dep_labels,
                    "dep_heads": dep_heads,
                    "tagged_ipa": tagged_ipa,
                    "lang_id": lang_id,
                    "source": source_key,
                }))
                parsed += 1

            if lines:
                f.write("".join(lines))

            if parsed > 0 and parsed % 50_000 < batch_size:
                elapsed = time.time() - t0
                rate = parsed / max(elapsed, 1)
                remaining = (len(triples) - i) / max(rate, 1)
                logger.info(
                    "  [%s] %d/%d parsed (%d failed) %.0f sent/sec ETA %.0fm",
                    stanza_code, parsed, len(triples), failed, rate, remaining / 60,
                )

    elapsed = time.time() - t0
    logger.info(
        "  [%s] DONE: %d parsed, %d failed, %.0f sent/sec, %.1fm",
        stanza_code, parsed, failed, parsed / max(elapsed, 1), elapsed / 60,
    )
    return parsed


def main():
    parser = argparse.ArgumentParser(description="Build dep-parsed IPA corpus for one language")
    parser.add_argument("--lang", required=True, choices=list(LANG_CONFIG.keys()))
    parser.add_argument("--target", type=int, default=3_000_000)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--device", default="gpu", choices=["gpu", "cpu"])
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    cfg = LANG_CONFIG[args.lang]
    lang_id = list(LANG_CONFIG.keys()).index(args.lang)
    logger.info(
        "Building %s (%s) corpus — target %d samples",
        cfg["name"], args.lang, args.target,
    )

    # 1. Load raw sentences with source tags
    tagged_sentences = load_leipzig_sentences(args.lang)
    if not tagged_sentences:
        logger.error("No sentences found. Download Leipzig corpora first.")
        return

    # 2. Filter (on text only, carry source through)
    filtered = [(s, src) for s, src in tagged_sentences if filter_sentence(s)]
    logger.info("Filtered: %d → %d (%.1f%% kept)", len(tagged_sentences), len(filtered),
                100 * len(filtered) / max(len(tagged_sentences), 1))

    # Log source distribution
    from collections import Counter
    src_counts = Counter(src for _, src in filtered)
    for src, cnt in src_counts.most_common():
        name = SOURCE_REGISTRY.get(src, (src,))[0]
        logger.info("  source %s: %d (%.1f%%)", name, cnt, 100 * cnt / len(filtered))

    # 3. Sample if we have more than target
    random.seed(args.seed)
    if len(filtered) > args.target:
        random.shuffle(filtered)
        filtered = filtered[:args.target]
    logger.info("Sampled %d sentences", len(filtered))

    # 4. IPA conversion
    logger.info("Converting to IPA via epitran (%s)...", cfg["epitran"])
    triples = ipa_convert_batch(filtered, cfg["epitran"])
    logger.info("IPA triples: %d (%.1f%% aligned)", len(triples),
                100 * len(triples) / max(len(filtered), 1))

    # 5. Dep parse + write
    out_dir = OUT_BASE / args.lang
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "sentences.jsonl"

    n = dep_parse_and_write(
        triples, cfg["stanza"], lang_id,
        out_path=out_path, batch_size=args.batch_size, device=args.device,
    )
    logger.info("Final: %d sentences written to %s", n, out_path)


if __name__ == "__main__":
    main()

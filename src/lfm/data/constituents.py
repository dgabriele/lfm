"""Constituency parsing for phrase-level dataset augmentation.

Extracts phrase constituents (NP, VP, PP, S, etc.) from raw text
using Stanza's constituency parser.  Each constituent becomes a
separate training sample, teaching the decoder to produce variable-
length output — from short noun phrases to full sentences.

Supported languages: de, en, es, id, pt, tr, vi (those with Stanza
constituency models).  Other languages pass through with full
sentences only.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)

# Languages with Stanza constituency parsers
CONSTITUENCY_LANGS: dict[str, str] = {
    "deu": "de", "eng": "en", "spa": "es", "ind": "id",
    "por": "pt", "tur": "tr", "vie": "vi",
}

# Minimum constituent length (characters) to include as a sample
MIN_CONSTITUENT_LENGTH = 10

# Constituent labels to extract (standard Penn Treebank / universal)
EXTRACT_LABELS = {
    "S", "SBAR", "SBARQ", "SQ", "SINV",  # clauses
    "NP", "NP-TMP",                         # noun phrases
    "VP",                                    # verb phrases
    "PP",                                    # prepositional phrases
    "ADJP", "ADVP",                          # modifier phrases
}


@dataclass
class Constituent:
    """A phrase constituent extracted from a sentence."""

    text: str
    label: str
    depth: int
    language: str


class ConstituencyExtractor:
    """Extract phrase constituents from text via Stanza.

    Lazily loads Stanza pipelines per language on first use.
    Thread-safe for single-language use (Stanza pipelines are not
    thread-safe across languages).

    Args:
        min_length: Minimum character length for extracted constituents.
        labels: Set of constituency labels to extract.
    """

    def __init__(
        self,
        min_length: int = MIN_CONSTITUENT_LENGTH,
        labels: set[str] | None = None,
    ) -> None:
        self.min_length = min_length
        self.labels = labels or EXTRACT_LABELS
        self._pipelines: dict[str, object] = {}

    def _get_pipeline(self, lang_iso2: str) -> object:
        """Lazily load a Stanza pipeline for the given language."""
        if lang_iso2 not in self._pipelines:
            import stanza

            logger.info("Loading Stanza constituency parser for %s...", lang_iso2)
            stanza.download(lang_iso2, processors="tokenize,pos,constituency", verbose=False)
            self._pipelines[lang_iso2] = stanza.Pipeline(
                lang_iso2,
                processors="tokenize,pos,constituency",
                use_gpu=False,  # CPU — leave GPU free, workers run in parallel
                verbose=False,
            )
        return self._pipelines[lang_iso2]

    def extract(
        self,
        text: str,
        language: str,
    ) -> list[Constituent]:
        """Extract phrase constituents from a sentence.

        Args:
            text: Raw sentence text.
            language: ISO 639-3 language code.

        Returns:
            List of Constituent objects. Always includes the full
            sentence as a constituent (label="S", depth=0).
        """
        iso2 = CONSTITUENCY_LANGS.get(language)
        if iso2 is None:
            return []  # Language not supported — caller keeps full sentence

        try:
            nlp = self._get_pipeline(iso2)
            doc = nlp(text)  # type: ignore[operator]
        except Exception as e:
            logger.debug("Parse failed for %s: %s", language, e)
            return []

        constituents: list[Constituent] = []
        for sentence in doc.sentences:
            tree = sentence.constituency
            if tree is None:
                continue
            self._walk_tree(tree, language, constituents, depth=0)

        return constituents

    def _walk_tree(
        self,
        node: object,
        language: str,
        out: list[Constituent],
        depth: int,
    ) -> None:
        """Recursively extract constituents from a parse tree."""
        label = getattr(node, "label", "")
        children = getattr(node, "children", [])

        if label in self.labels and depth > 0:
            # Extract the text of this constituent
            text = self._tree_text(node)
            if len(text) >= self.min_length:
                out.append(Constituent(
                    text=text, label=label, depth=depth, language=language,
                ))

        for child in children:
            if hasattr(child, "children") and child.children:
                self._walk_tree(child, language, out, depth + 1)

    @staticmethod
    def _tree_text(node: object) -> str:
        """Extract the leaf text from a parse tree node."""
        leaves: list[str] = []

        def _collect(n: object) -> None:
            children = getattr(n, "children", [])
            if not children:
                label = getattr(n, "label", "")
                if label:
                    leaves.append(label)
            else:
                for c in children:
                    _collect(c)

        _collect(node)
        return " ".join(leaves)


def _parse_language_worker(
    args: tuple[str, list[str], int, None],
) -> list[tuple[str, str, str]]:
    """Worker: parse all sentences for one language, extract constituents.

    Runs in a separate process with its own Stanza pipeline.
    Progress written directly to a shared log file (no queues).
    """
    import stanza

    lang, texts, min_length, progress_file = args
    iso2 = CONSTITUENCY_LANGS[lang]

    stanza.download(iso2, processors="tokenize,pos,constituency", verbose=False)
    nlp = stanza.Pipeline(
        iso2, processors="tokenize,pos,constituency",
        use_gpu=True, verbose=False,
    )

    labels = EXTRACT_LABELS
    results: list[tuple[str, str, str]] = []
    batch_size = 64
    processed = 0

    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        doc = nlp("\n\n".join(batch))

        for sentence in doc.sentences:
            if sentence.constituency is None:
                continue
            _extract_from_tree(
                sentence.constituency, lang, labels, min_length, results, depth=0,
            )

        processed += len(batch)
        if processed % 500 == 0:
            import sys

            sys.stderr.write(
                f"  [{lang}] {processed}/{len(texts)} → {len(results)} constituents\n"
            )
            sys.stderr.flush()

    import sys

    sys.stderr.write(
        f"  [{lang}] {processed}/{len(texts)} → {len(results)} constituents (done)\n"
    )
    sys.stderr.flush()

    return results


def _write_progress(path: str, lang: str, done: int, total: int, n_const: int) -> None:
    """Append progress line to shared file (atomic write via append mode)."""
    import time

    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    line = f"{ts}   [{lang}] {done}/{total} sentences → {n_const} constituents\n"
    with open(path, "a") as f:
        f.write(line)
        f.flush()


def _extract_from_tree(
    node: object,
    language: str,
    labels: set[str],
    min_length: int,
    out: list[tuple[str, str, str]],
    depth: int,
) -> None:
    """Recursively extract constituents from a parse tree node."""
    label = getattr(node, "label", "")
    children = getattr(node, "children", [])

    if label in labels and depth > 0:
        text = _tree_text(node)
        if len(text) >= min_length:
            out.append((language, text, label))

    for child in children:
        if hasattr(child, "children") and child.children:
            _extract_from_tree(child, language, labels, min_length, out, depth + 1)


def _tree_text(node: object) -> str:
    """Extract leaf text from a parse tree node."""
    leaves: list[str] = []

    def _collect(n: object) -> None:
        children = getattr(n, "children", [])
        if not children:
            label = getattr(n, "label", "")
            if label:
                leaves.append(label)
        else:
            for c in children:
                _collect(c)

    _collect(node)
    return " ".join(leaves)


def extract_constituents_parallel(
    samples: list[tuple[str, str]],
    min_length: int = MIN_CONSTITUENT_LENGTH,
    num_workers: int | None = None,
) -> list[tuple[str, str, str]]:
    """Extract constituents with per-language parallelism.

    Groups samples by language, then processes each supported language
    in a separate worker process (each with its own Stanza pipeline
    on CPU).  Unsupported languages pass through with full sentences.

    Args:
        samples: List of (iso3_language, raw_text) tuples.
        min_length: Minimum constituent character length.
        num_workers: Max parallel workers. None = 90% of CPU cores.

    Returns:
        List of (language, text, label) tuples.
    """
    import multiprocessing as mp
    import os
    from collections import defaultdict

    if num_workers is None:
        num_workers = max(1, int(os.cpu_count() * 0.9))

    # Group by language
    by_lang: dict[str, list[str]] = defaultdict(list)
    for lang, text in samples:
        by_lang[lang].append(text)

    # All samples start as full sentences
    results: list[tuple[str, str, str]] = [(lang, text, "S") for lang, text in samples]

    # Build work items for supported languages only
    work_items: list[tuple[str, list[str], int]] = []
    for lang, texts in by_lang.items():
        if lang in CONSTITUENCY_LANGS:
            work_items.append((lang, texts, min_length))

    if not work_items:
        logger.info("No supported languages for constituency extraction")
        return results

    import signal

    import torch

    use_gpu = torch.cuda.is_available()
    # 2 GPU workers (~2 GB each, fits 8 GB) or parallel CPU
    if use_gpu:
        actual_workers = min(2, len(work_items))
    else:
        actual_workers = min(num_workers, len(work_items))

    total_sents = sum(len(t) for _, t, _ in work_items)
    logger.info(
        "Constituency extraction: %d languages, %d workers (%s), %d sentences",
        len(work_items), actual_workers,
        "GPU" if use_gpu else "CPU", total_sents,
    )

    # Workers write progress to stderr (captured by 2>&1 redirect).
    work_items_final = [
        (lang, texts, ml, None) for lang, texts, ml in work_items
    ]

    if actual_workers == 1:
        # Sequential — no subprocesses to orphan
        lang_results = [_parse_language_worker(item) for item in work_items_final]
    else:
        # Parallel CPU — use spawn, register cleanup on SIGTERM/SIGINT
        ctx = mp.get_context("spawn")
        pool = ctx.Pool(actual_workers)

        def _cleanup(signum: int, frame: object) -> None:
            pool.terminate()
            pool.join()
            raise SystemExit(1)

        old_term = signal.signal(signal.SIGTERM, _cleanup)
        old_int = signal.signal(signal.SIGINT, _cleanup)
        try:
            lang_results = pool.map(_parse_language_worker, work_items_final)
        finally:
            pool.close()
            pool.join()
            signal.signal(signal.SIGTERM, old_term)
            signal.signal(signal.SIGINT, old_int)

    # Merge results
    constituent_count = 0
    for lang_constituents in lang_results:
        results.extend(lang_constituents)
        constituent_count += len(lang_constituents)

    logger.info(
        "Extracted %d constituents → %d total samples",
        constituent_count, len(results),
    )
    return results

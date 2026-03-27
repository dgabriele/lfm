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
    args: tuple[str, list[str], int, object | None],
) -> list[tuple[str, str, str]]:
    """Worker: parse all sentences for one language, extract constituents.

    Runs in a separate process with its own Stanza pipeline.
    Processes sentences in batches for throughput.
    Progress is reported via an mp.Queue if provided.
    """
    import stanza

    lang, texts, min_length, progress_queue = args
    iso2 = CONSTITUENCY_LANGS[lang]
    use_gpu = not _is_multiprocess_mode()

    stanza.download(iso2, processors="tokenize,pos,constituency", verbose=False)
    nlp = stanza.Pipeline(
        iso2, processors="tokenize,pos,constituency",
        use_gpu=use_gpu, verbose=False,
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
        if processed % 500 == 0 and progress_queue is not None:
            progress_queue.put((lang, processed, len(texts), len(results)))

    if progress_queue is not None:
        progress_queue.put((lang, processed, len(texts), len(results)))

    return results


def _is_multiprocess_mode() -> bool:
    """Check if we're running in a multiprocessing worker."""
    import multiprocessing

    return multiprocessing.current_process().name != "MainProcess"


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

    import threading

    import torch

    use_gpu = torch.cuda.is_available()
    if use_gpu:
        # 2 GPU parsers in parallel (~2 GB VRAM each, fits in 8 GB)
        actual_workers = min(2, len(work_items))
    else:
        actual_workers = min(num_workers, len(work_items))

    total_sents = sum(len(t) for _, t, _ in work_items)
    logger.info(
        "Constituency extraction: %d languages, %d workers (%s), %d sentences",
        len(work_items), actual_workers,
        "GPU" if use_gpu else "CPU", total_sents,
    )

    # Progress queue: workers report (lang, processed, total, constituents)
    # Use manager queue for spawn context compatibility
    manager = mp.Manager()
    progress_queue = manager.Queue()

    # Add queue to work items
    work_with_queue = [
        (lang, texts, ml, progress_queue) for lang, texts, ml in work_items
    ]

    # Progress printer thread
    def _progress_printer() -> None:
        while True:
            try:
                msg = progress_queue.get(timeout=1.0)
                if msg is None:
                    break
                lang, done, total, n_const = msg
                logger.info(
                    "  [%s] %d/%d sentences → %d constituents",
                    lang, done, total, n_const,
                )
            except Exception:
                continue

    printer = threading.Thread(target=_progress_printer, daemon=True)
    printer.start()

    if actual_workers == 1:
        lang_results = [_parse_language_worker(item) for item in work_with_queue]
    else:
        # Use spawn context for GPU safety (fork + CUDA = crashes)
        ctx = mp.get_context("spawn")
        with ctx.Pool(actual_workers) as pool:
            lang_results = pool.map(_parse_language_worker, work_with_queue)

    # Signal printer to stop
    progress_queue.put(None)
    printer.join(timeout=5.0)

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

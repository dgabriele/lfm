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


def extract_constituents_batch(
    samples: list[tuple[str, str]],
    min_length: int = MIN_CONSTITUENT_LENGTH,
) -> list[tuple[str, str, str]]:
    """Extract constituents from a batch of (language, text) samples.

    For languages with constituency parsers, extracts sub-phrases.
    For others, returns the original sentence only.

    Args:
        samples: List of (iso3_language, raw_text) tuples.
        min_length: Minimum constituent character length.

    Returns:
        List of (language, text, label) tuples. Label is "S" for
        full sentences and the constituent tag for sub-phrases.
    """
    extractor = ConstituencyExtractor(min_length=min_length)
    results: list[tuple[str, str, str]] = []

    supported_count = 0
    constituent_count = 0

    for lang, text in samples:
        # Always include the full sentence
        results.append((lang, text, "S"))

        if lang in CONSTITUENCY_LANGS:
            supported_count += 1
            constituents = extractor.extract(text, lang)
            for c in constituents:
                results.append((lang, c.text, c.label))
                constituent_count += 1

    logger.info(
        "Constituency extraction: %d supported sentences → %d constituents "
        "(%d total samples including originals)",
        supported_count, constituent_count, len(results),
    )
    return results

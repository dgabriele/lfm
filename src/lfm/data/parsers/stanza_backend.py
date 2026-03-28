"""Stanza constituency parsing backend.

Supports: de, en, es, id, pt, tr, vi (languages with Stanza
constituency models).
"""

from __future__ import annotations

import logging

from lfm.data.parsers.base import ConstituencyBackend, ParseTree

logger = logging.getLogger(__name__)

STANZA_LANGS: dict[str, str] = {
    "deu": "de", "eng": "en", "spa": "es", "ind": "id",
    "por": "pt", "tur": "tr", "vie": "vi",
}


def _stanza_tree_to_parse_tree(node: object) -> ParseTree:
    """Convert a Stanza Tree to our ParseTree."""
    label = getattr(node, "label", "")
    children = getattr(node, "children", [])
    if not children:
        return ParseTree(label=label)
    return ParseTree(
        label=label,
        children=[_stanza_tree_to_parse_tree(c) for c in children],
    )


class StanzaBackend:
    """Constituency parsing via Stanza."""

    def __init__(self, lang_iso3: str, use_gpu: bool = True) -> None:
        import stanza

        iso2 = STANZA_LANGS[lang_iso3]
        logger.info("Loading Stanza constituency parser for %s...", iso2)
        stanza.download(iso2, processors="tokenize,pos,constituency", verbose=False)
        self._nlp = stanza.Pipeline(
            iso2, processors="tokenize,pos,constituency",
            use_gpu=use_gpu, verbose=False,
        )
        self._lang = lang_iso3

    def parse(self, sentences: list[str]) -> list[ParseTree | None]:
        doc = self._nlp("\n\n".join(sentences))
        results: list[ParseTree | None] = []
        for sent in doc.sentences:
            if sent.constituency is None:
                results.append(None)
            else:
                results.append(_stanza_tree_to_parse_tree(sent.constituency))
        return results

    def supports(self, lang_iso3: str) -> bool:
        return lang_iso3 in STANZA_LANGS

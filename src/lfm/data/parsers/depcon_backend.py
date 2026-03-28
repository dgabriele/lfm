"""Dependency-to-constituency conversion backend.

For languages without pretrained constituency parsers. Uses Stanza
dependency parsing + heuristic subtree extraction to approximate
constituency phrases.

Supports: cs, et, fi, hi, ru (and any other language with a Stanza
dependency parser).

The conversion extracts headed subtrees from dependency parses:
each non-leaf word and its descendants form a constituent-like phrase.
Labels are inferred from the dependency relation of the head word
(nsubj → NP, obj/dobj → NP, obl/nmod → PP, etc.).
"""

from __future__ import annotations

import logging

from lfm.data.parsers.base import ConstituencyBackend, ParseTree

logger = logging.getLogger(__name__)

# Languages that use dep→con conversion (no constituency parser available)
DEPCON_LANGS: dict[str, str] = {
    "ces": "cs", "est": "et", "fin": "fi",
    "hin": "hi", "rus": "ru",
}

# Map UD dependency relations to approximate constituency labels
_DEPREL_TO_LABEL: dict[str, str] = {
    "nsubj": "NP", "nsubj:pass": "NP",
    "obj": "NP", "dobj": "NP", "iobj": "NP",
    "obl": "PP", "obl:arg": "PP", "nmod": "PP",
    "advcl": "SBAR", "acl": "SBAR", "acl:relcl": "SBAR",
    "ccomp": "SBAR", "xcomp": "VP",
    "amod": "ADJP", "advmod": "ADVP",
    "conj": "S",
}


def _build_tree_from_dep(
    words: list[dict],
    head_idx: int,
    children_map: dict[int, list[int]],
) -> ParseTree:
    """Recursively build a ParseTree from dependency structure.

    Args:
        words: List of word dicts with 'text', 'deprel', 'id' fields.
        head_idx: Index of the current head word (1-based).
        children_map: Maps head index → list of dependent indices.
    """
    word = words[head_idx - 1]  # 1-based to 0-based
    deps = children_map.get(head_idx, [])

    if not deps:
        # Leaf — terminal word
        return ParseTree(label=word["text"])

    # Build children in linear order
    child_trees: list[ParseTree] = []
    all_indices = sorted(deps + [head_idx])

    for idx in all_indices:
        if idx == head_idx:
            child_trees.append(ParseTree(label=word["text"]))
        else:
            child_trees.append(
                _build_tree_from_dep(words, idx, children_map)
            )

    # Infer label from dependency relation
    label = _DEPREL_TO_LABEL.get(word["deprel"], "XP")
    if word["deprel"] == "root":
        label = "S"

    return ParseTree(label=label, children=child_trees)


class DepConBackend:
    """Dependency → constituency conversion via Stanza."""

    def __init__(self, lang_iso3: str, use_gpu: bool = False) -> None:
        import stanza

        iso2 = DEPCON_LANGS[lang_iso3]
        logger.info("Loading Stanza dependency parser for %s...", iso2)
        stanza.download(iso2, processors="tokenize,pos,depparse", verbose=False)
        self._nlp = stanza.Pipeline(
            iso2, processors="tokenize,pos,depparse",
            use_gpu=use_gpu, verbose=False,
        )
        self._lang = lang_iso3

    def parse(self, sentences: list[str]) -> list[ParseTree | None]:
        doc = self._nlp("\n\n".join(sentences))
        results: list[ParseTree | None] = []

        for sent in doc.sentences:
            try:
                words = [
                    {"text": w.text, "deprel": w.deprel, "id": w.id, "head": w.head}
                    for w in sent.words
                ]

                # Build children map
                children_map: dict[int, list[int]] = {}
                root_idx = None
                for w in words:
                    head = w["head"]
                    if head == 0:
                        root_idx = w["id"]
                    else:
                        children_map.setdefault(head, []).append(w["id"])

                if root_idx is None:
                    results.append(None)
                    continue

                tree = _build_tree_from_dep(words, root_idx, children_map)
                results.append(tree)
            except Exception as e:
                logger.debug("DepCon parse failed: %s", e)
                results.append(None)

        return results

    def supports(self, lang_iso3: str) -> bool:
        return lang_iso3 in DEPCON_LANGS

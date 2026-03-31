"""Dependency-to-constituency conversion backend.

Uses Stanza dependency parsing + heuristic subtree extraction to
approximate constituency phrases for all languages uniformly.

The key heuristic: synthesize VP constituents by grouping each verb
with its argument dependents (obj, iobj, obl, xcomp, ccomp, advmod).
In dependency grammar, the verb is the head with arguments hanging
off it — there is no VP node. We reconstruct it by identifying
verb-argument clusters.

Supports 14/16 LFM training languages (all with Stanza dep parsers;
swa and tgl lack support).
"""

from __future__ import annotations

import logging

from lfm.data.parsers.base import ConstituencyBackend, ParseTree

logger = logging.getLogger(__name__)

DEPCON_LANGS: dict[str, str] = {
    "ara": "ar", "deu": "de", "eng": "en", "fin": "fi",
    "hin": "hi", "hun": "hu", "ind": "id", "kat": "ka",
    "kor": "ko", "por": "pt", "rus": "ru", "tha": "th",
    "tur": "tr", "vie": "vi",
}

# UD deprels that map to specific phrase labels
_DEPREL_TO_LABEL: dict[str, str] = {
    "nsubj": "NP", "nsubj:pass": "NP",
    "obj": "NP", "dobj": "NP", "iobj": "NP",
    "obl": "PP", "obl:arg": "PP", "nmod": "PP",
    "advcl": "SBAR", "acl": "SBAR", "acl:relcl": "SBAR",
    "ccomp": "SBAR", "xcomp": "VP",
    "amod": "ADJP", "advmod": "ADVP",
    "conj": "S",
}

# UD deprels that are verb arguments (grouped into synthetic VP)
_VP_ARGUMENT_DEPRELS = {
    "obj", "dobj", "iobj", "obl", "obl:arg",
    "xcomp", "ccomp", "advmod", "advcl",
}

# UD deprels that are subject-like (excluded from VP, grouped as NP)
_SUBJECT_DEPRELS = {"nsubj", "nsubj:pass"}


def _build_tree_from_dep(
    words: list[dict],
    head_idx: int,
    children_map: dict[int, list[int]],
    is_root: bool = False,
) -> ParseTree:
    """Build a ParseTree from dependency structure with synthetic VPs.

    When processing the root verb (or any verb head), splits dependents
    into subject-side (NP) and predicate-side (VP).  The VP contains
    the verb plus its arguments.
    """
    word = words[head_idx - 1]
    deps = children_map.get(head_idx, [])

    if not deps:
        return ParseTree(label=word["text"])

    # Check if this head is a verb that should produce a VP
    is_verb = word.get("upos") in ("VERB", "AUX") or word["deprel"] == "root"

    if is_verb and is_root:
        return _build_clause_with_vp(words, head_idx, deps, children_map)

    # Non-verb or non-root: standard subtree
    child_trees: list[ParseTree] = []
    all_indices = sorted(deps + [head_idx])

    for idx in all_indices:
        if idx == head_idx:
            child_trees.append(ParseTree(label=word["text"]))
        else:
            child_trees.append(
                _build_tree_from_dep(words, idx, children_map),
            )

    label = _DEPREL_TO_LABEL.get(word["deprel"], "XP")
    if word["deprel"] == "root":
        label = "S"

    return ParseTree(label=label, children=child_trees)


def _build_clause_with_vp(
    words: list[dict],
    verb_idx: int,
    deps: list[int],
    children_map: dict[int, list[int]],
) -> ParseTree:
    """Build a clause node (S) with a synthetic VP constituent.

    Splits the verb's dependents into:
    - Pre-verbal subjects/topics → individual NP/ADJP subtrees
    - Verb + arguments → grouped into a VP node
    - Remaining modifiers → attached to VP or S as appropriate

    This produces: S → [NP_subj] [VP → verb + arguments]
    """
    verb_word = words[verb_idx - 1]

    # Partition dependents
    subject_deps: list[int] = []
    vp_deps: list[int] = []
    other_deps: list[int] = []

    for dep_idx in deps:
        dep_word = words[dep_idx - 1]
        deprel = dep_word["deprel"].split(":")[0] if ":" in dep_word["deprel"] else dep_word["deprel"]
        full_deprel = dep_word["deprel"]

        if full_deprel in _SUBJECT_DEPRELS:
            subject_deps.append(dep_idx)
        elif full_deprel in _VP_ARGUMENT_DEPRELS or deprel in {"obj", "obl", "xcomp", "ccomp", "advmod", "advcl"}:
            vp_deps.append(dep_idx)
        else:
            other_deps.append(dep_idx)

    # Build subject subtrees
    clause_children: list[ParseTree] = []
    for dep_idx in sorted(subject_deps):
        clause_children.append(
            _build_tree_from_dep(words, dep_idx, children_map),
        )

    # Build VP: verb + argument subtrees in linear order
    vp_children: list[ParseTree] = []
    vp_indices = sorted(vp_deps + [verb_idx])

    for idx in vp_indices:
        if idx == verb_idx:
            vp_children.append(ParseTree(label=verb_word["text"]))
        else:
            vp_children.append(
                _build_tree_from_dep(words, idx, children_map),
            )

    if vp_children:
        # Insert VP after subjects, before other
        clause_children.append(ParseTree(label="VP", children=vp_children))

    # Attach remaining dependents (punct, discourse, etc.) to clause
    for dep_idx in sorted(other_deps):
        dep_word = words[dep_idx - 1]
        if dep_word["deprel"] in ("punct", "discourse", "vocative"):
            # Terminal — don't recurse
            clause_children.append(ParseTree(label=dep_word["text"]))
        else:
            clause_children.append(
                _build_tree_from_dep(words, dep_idx, children_map),
            )

    # Sort all clause children by their position in the sentence
    # (they were added out of order due to partitioning)
    def _min_position(tree: ParseTree) -> int:
        """Get the leftmost word position for ordering."""
        if tree.is_leaf:
            for i, w in enumerate(words):
                if w["text"] == tree.label:
                    return i
            return 999
        return min((_min_position(c) for c in tree.children), default=999)

    clause_children.sort(key=_min_position)

    return ParseTree(label="S", children=clause_children)


class DepConBackend:
    """Dependency → constituency conversion via Stanza."""

    def __init__(self, lang_iso3: str, use_gpu: bool = False) -> None:
        import stanza

        iso2 = DEPCON_LANGS[lang_iso3]
        logger.info("Loading Stanza dependency parser for %s...", iso2)
        stanza.download(iso2, processors="tokenize,pos,lemma,depparse", verbose=False)
        self._nlp = stanza.Pipeline(
            iso2, processors="tokenize,pos,lemma,depparse",
            use_gpu=use_gpu, verbose=False,
        )
        self._lang = lang_iso3

    def parse(self, sentences: list[str]) -> list[ParseTree | None]:
        doc = self._nlp("\n\n".join(sentences))
        results: list[ParseTree | None] = []

        for sent in doc.sentences:
            try:
                words = [
                    {
                        "text": w.text,
                        "deprel": w.deprel,
                        "id": w.id,
                        "head": w.head,
                        "upos": w.upos,
                    }
                    for w in sent.words
                ]

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

                tree = _build_tree_from_dep(
                    words, root_idx, children_map, is_root=True,
                )
                results.append(tree)
            except Exception as e:
                logger.debug("DepCon parse failed: %s", e)
                results.append(None)

        return results

    def supports(self, lang_iso3: str) -> bool:
        return lang_iso3 in DEPCON_LANGS

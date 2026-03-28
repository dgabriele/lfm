"""Berkeley Neural Parser (benepar) constituency backend.

Supports: ar, hu, ko, pl (languages with pretrained benepar models).
"""

from __future__ import annotations

import logging

from lfm.data.parsers.base import ConstituencyBackend, ParseTree

logger = logging.getLogger(__name__)

BENEPAR_MODELS: dict[str, str] = {
    "ara": "benepar_ar2",
    "hun": "benepar_hu2",
    "kor": "benepar_ko2",
    "pol": "benepar_pl2",
}

# spaCy model names for tokenization (benepar needs spaCy pipeline)
SPACY_MODELS: dict[str, str] = {
    "ara": "xx_ent_wiki_sm",   # multilingual fallback
    "hun": "xx_ent_wiki_sm",
    "kor": "ko_core_news_sm",
    "pol": "pl_core_news_sm",
}


def _benepar_tree_to_parse_tree(tree_str: str) -> ParseTree:
    """Parse a bracketed tree string into our ParseTree.

    Benepar outputs Penn Treebank bracketed format via spaCy's
    ._.parse_string attribute.
    """
    # Simple recursive descent parser for bracketed trees
    tokens = tree_str.replace("(", " ( ").replace(")", " ) ").split()
    pos = [0]

    def _parse() -> ParseTree:
        if pos[0] >= len(tokens):
            return ParseTree(label="")
        if tokens[pos[0]] == "(":
            pos[0] += 1  # consume (
            label = tokens[pos[0]]
            pos[0] += 1  # consume label
            children: list[ParseTree] = []
            while pos[0] < len(tokens) and tokens[pos[0]] != ")":
                children.append(_parse())
            pos[0] += 1  # consume )
            if not children:
                return ParseTree(label=label)
            return ParseTree(label=label, children=children)
        else:
            # Terminal
            word = tokens[pos[0]]
            pos[0] += 1
            return ParseTree(label=word)

    return _parse()


class BeneparBackend:
    """Constituency parsing via Berkeley Neural Parser."""

    def __init__(self, lang_iso3: str) -> None:
        import benepar
        import spacy

        model_name = BENEPAR_MODELS[lang_iso3]
        spacy_model = SPACY_MODELS.get(lang_iso3, "xx_ent_wiki_sm")

        logger.info(
            "Loading benepar %s with spaCy %s...", model_name, spacy_model,
        )
        try:
            self._nlp = spacy.load(spacy_model)
        except OSError:
            from spacy.cli import download
            download(spacy_model)
            self._nlp = spacy.load(spacy_model)

        if "benepar" not in self._nlp.pipe_names:
            self._nlp.add_pipe("benepar", config={"model": model_name})

        self._lang = lang_iso3

    def parse(self, sentences: list[str]) -> list[ParseTree | None]:
        results: list[ParseTree | None] = []
        # Process one at a time to handle failures gracefully
        for sent in sentences:
            try:
                doc = self._nlp(sent)
                for spacy_sent in doc.sents:
                    tree_str = spacy_sent._.parse_string
                    if tree_str:
                        results.append(_benepar_tree_to_parse_tree(tree_str))
                    else:
                        results.append(None)
                    break  # only first sentence per input
            except Exception as e:
                logger.debug("Benepar parse failed for %s: %s", self._lang, e)
                results.append(None)
        return results

    def supports(self, lang_iso3: str) -> bool:
        return lang_iso3 in BENEPAR_MODELS

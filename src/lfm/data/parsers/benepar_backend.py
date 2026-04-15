"""Berkeley Neural Parser (benepar) constituency backend.

Supports: ar, en, hu, ko, pl (languages with pretrained benepar models).

Benepar is typically 5-10x faster than Stanza's CRF constituency parser
on GPU and produces Penn Treebank-compatible bracketed trees (same
labels).
"""

from __future__ import annotations

import logging

from lfm.data.parsers.base import ConstituencyBackend, ParseTree

logger = logging.getLogger(__name__)


def _patch_t5_tokenizer() -> None:
    """Compatibility shim for benepar 0.2 + transformers ≥ 5.

    Benepar's underlying benepar_en3 parser is built on T5 and calls
    ``T5Tokenizer.build_inputs_with_special_tokens``, which recent
    transformers removed.  We restore the original behavior inline:
    wrap ids with EOS, optionally concatenate a second sequence.
    """
    try:
        from transformers import T5Tokenizer, T5TokenizerFast
    except ImportError:
        return
    for cls in (T5Tokenizer, T5TokenizerFast):
        if not hasattr(cls, "build_inputs_with_special_tokens"):
            def _bi(self, token_ids_0, token_ids_1=None):
                if token_ids_1 is None:
                    return list(token_ids_0) + [self.eos_token_id]
                return (
                    list(token_ids_0) + [self.eos_token_id]
                    + list(token_ids_1) + [self.eos_token_id]
                )
            cls.build_inputs_with_special_tokens = _bi


_patch_t5_tokenizer()

BENEPAR_MODELS: dict[str, str] = {
    "ara": "benepar_ar2",
    "hun": "benepar_hu2",
    "kor": "benepar_ko2",
    "pol": "benepar_pl2",
    # English: benepar_en3 (standard) ~90MB, benepar_en3_large ~1.5GB.
    # en3 is much faster with only a small F1 drop; preferred for LFM.
    "eng": "benepar_en3",
}

# spaCy model names for tokenization (benepar needs spaCy pipeline)
SPACY_MODELS: dict[str, str] = {
    "ara": "xx_ent_wiki_sm",   # multilingual fallback
    "hun": "xx_ent_wiki_sm",
    "kor": "ko_core_news_sm",
    "pol": "pl_core_news_sm",
    "eng": "en_core_web_sm",
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

    def __init__(self, lang_iso3: str, use_gpu: bool = True) -> None:
        import benepar
        import spacy

        if use_gpu:
            try:
                spacy.require_gpu()
            except Exception:
                logger.warning("benepar: GPU requested but unavailable; CPU fallback")

        model_name = BENEPAR_MODELS[lang_iso3]
        spacy_model = SPACY_MODELS.get(lang_iso3, "xx_ent_wiki_sm")

        logger.info(
            "Loading benepar %s with spaCy %s (use_gpu=%s)...",
            model_name, spacy_model, use_gpu,
        )
        try:
            self._nlp = spacy.load(spacy_model)
        except OSError:
            from spacy.cli import download
            download(spacy_model)
            self._nlp = spacy.load(spacy_model)

        if "benepar" not in self._nlp.pipe_names:
            try:
                self._nlp.add_pipe("benepar", config={"model": model_name})
            except LookupError:
                benepar.download(model_name)
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

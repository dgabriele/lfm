"""Per-language sentencepiece tokenizers with a disjoint global vocabulary.

Neuroglot and English each get their own BPE model trained on their
own corpus.  The two vocabularies are concatenated into a single
global index space that the seq2seq model uses::

    global id   meaning
    ─────────   ────────────────────────────────
    0           <pad>
    1           <unk>
    2           <bos>
    3           <eos>
    4           <ng>         (Neuroglot language tag)
    5           <en>         (English language tag)
    6           <mask>       (DAE mask token)
    7 .. 7+Vn   Neuroglot BPE units (Vn = neuroglot_vocab_size)
    7+Vn .. 7+Vn+Ve   English BPE units (Ve = english_vocab_size)

Each per-language sentencepiece model is trained with minimal specials
(only ``<unk>`` at local id 0) so its entire vocab is BPE units that
can be offset directly into the global space.  The per-language UNK at
local 0 becomes a dead slot in the global vocab — it never appears in
encoded output because we explicitly route unknowns to global id 1.
"""

from __future__ import annotations

import json
import logging
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import sentencepiece as spm

from lfm.unmt.config import UNMTConfig

logger = logging.getLogger(__name__)


PAD_ID = 0
UNK_ID = 1
BOS_ID = 2
EOS_ID = 3
NG_TAG_ID = 4
EN_TAG_ID = 5
MASK_ID = 6
NUM_SPECIAL = 7


@dataclass(frozen=True)
class TokenizerArtifacts:
    """Paths to the trained tokenizer files for a config."""

    neuroglot_model: Path
    english_model: Path


def _iter_corpus_lines(path: Path) -> Iterator[str]:
    """Yield text lines from a corpus, auto-detecting plain text vs JSONL."""
    with open(path, encoding="utf-8") as f:
        first = ""
        for line in f:
            first = line.strip()
            if first:
                break
        if not first:
            return

        is_jsonl = first.startswith("{")
        if is_jsonl:
            try:
                yield json.loads(first)["text"]
            except (KeyError, json.JSONDecodeError):
                pass
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    yield json.loads(line)["text"]
                except (KeyError, json.JSONDecodeError):
                    continue
        else:
            yield first
            for line in f:
                line = line.strip()
                if line:
                    yield line


def _write_training_text(
    corpus_path: Path,
    dest: Path,
    max_sentence_length: int,
    max_lines: int,
) -> int:
    """Extract plain-text training data from a single corpus.

    Returns the number of lines written.
    """
    dest.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with open(dest, "w", encoding="utf-8") as out:
        for text in _iter_corpus_lines(corpus_path):
            if written >= max_lines:
                break
            if len(text) > max_sentence_length:
                continue
            out.write(text)
            out.write("\n")
            written += 1
    return written


def _train_single_tokenizer(
    corpus_path: Path,
    model_prefix: Path,
    vocab_size: int,
    character_coverage: float,
    max_sentence_length: int,
    max_lines: int,
) -> None:
    """Train one sentencepiece BPE model on a monolingual corpus.

    The trained model has only ``<unk>`` at local id 0 — no pad/bos/eos
    — so the full vocab_size range is usable BPE units.  Specials are
    managed at the global-vocabulary level by :class:`BilingualTokenizer`.
    """
    model_prefix.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.NamedTemporaryFile(
        "w", delete=False, suffix=".txt", dir=str(model_prefix.parent),
    ) as tmp:
        tmp_path = Path(tmp.name)

    try:
        written = _write_training_text(
            corpus_path, tmp_path,
            max_sentence_length=max_sentence_length,
            max_lines=max_lines,
        )
        if written == 0:
            raise RuntimeError(f"No training lines extracted from {corpus_path}")
        logger.info(
            "Training sentencepiece on %d lines from %s (vocab=%d) → %s",
            written, corpus_path.name, vocab_size, model_prefix,
        )
        spm.SentencePieceTrainer.train(
            input=str(tmp_path),
            model_prefix=str(model_prefix),
            vocab_size=vocab_size,
            character_coverage=character_coverage,
            model_type="bpe",
            pad_id=-1,
            unk_id=0,
            bos_id=-1,
            eos_id=-1,
            input_sentence_size=written,
            shuffle_input_sentence=True,
            max_sentence_length=max_sentence_length,
        )
    finally:
        if tmp_path.exists():
            tmp_path.unlink()


def train_tokenizers(config: UNMTConfig) -> TokenizerArtifacts:
    """Train both per-language tokenizers if they don't already exist.

    Returns paths to the two trained sentencepiece ``.model`` files.
    Existing models are reused — tokenizer training is expensive and
    deterministic.
    """
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    ng_model_path = output_dir / f"{config.neuroglot_tokenizer_prefix}.model"
    en_model_path = output_dir / f"{config.english_tokenizer_prefix}.model"

    if not ng_model_path.exists():
        neuroglot_path = Path(config.neuroglot_corpus)
        if not neuroglot_path.exists():
            raise FileNotFoundError(f"Neuroglot corpus not found: {neuroglot_path}")
        _train_single_tokenizer(
            corpus_path=neuroglot_path,
            model_prefix=output_dir / config.neuroglot_tokenizer_prefix,
            vocab_size=config.neuroglot_vocab_size,
            character_coverage=config.character_coverage_neuroglot,
            max_sentence_length=config.max_sentence_length,
            max_lines=config.max_tokenizer_lines,
        )
    else:
        logger.info("Neuroglot tokenizer already trained at %s", ng_model_path)

    if not en_model_path.exists():
        english_path = Path(config.english_corpus)
        if not english_path.exists():
            raise FileNotFoundError(f"English corpus not found: {english_path}")
        _train_single_tokenizer(
            corpus_path=english_path,
            model_prefix=output_dir / config.english_tokenizer_prefix,
            vocab_size=config.english_vocab_size,
            character_coverage=config.character_coverage_english,
            max_sentence_length=config.max_sentence_length,
            max_lines=config.max_tokenizer_lines,
        )
    else:
        logger.info("English tokenizer already trained at %s", en_model_path)

    return TokenizerArtifacts(
        neuroglot_model=ng_model_path,
        english_model=en_model_path,
    )


class BilingualTokenizer:
    """Two per-language sentencepiece models behind a shared global vocab.

    All encode/decode operations speak in **global** vocabulary ids.
    Callers indicate which language a string belongs to via the
    ``lang`` argument (``"ng"`` or ``"en"``).

    Global vocab layout::

        0..6                            shared specials
        NUM_SPECIAL..NG_END             Neuroglot BPE
        NG_END..EN_END                  English BPE

    ``global_vocab_size`` returns the total size of the concatenated
    vocabulary.  The seq2seq model in Stage 3 uses exactly this many
    embedding rows.
    """

    def __init__(self, artifacts: TokenizerArtifacts) -> None:
        self._ng = spm.SentencePieceProcessor()
        self._ng.Load(str(artifacts.neuroglot_model))
        self._en = spm.SentencePieceProcessor()
        self._en.Load(str(artifacts.english_model))

        self._ng_vocab = self._ng.GetPieceSize()
        self._en_vocab = self._en.GetPieceSize()

        self._ng_offset = NUM_SPECIAL
        self._en_offset = NUM_SPECIAL + self._ng_vocab
        self._global_vocab_size = NUM_SPECIAL + self._ng_vocab + self._en_vocab

    @property
    def global_vocab_size(self) -> int:
        return self._global_vocab_size

    @property
    def neuroglot_range(self) -> tuple[int, int]:
        """``[start, end)`` of Neuroglot BPE units in global ids."""
        return self._ng_offset, self._ng_offset + self._ng_vocab

    @property
    def english_range(self) -> tuple[int, int]:
        """``[start, end)`` of English BPE units in global ids."""
        return self._en_offset, self._en_offset + self._en_vocab

    def lang_tag_id(self, lang: str) -> int:
        if lang == "ng":
            return NG_TAG_ID
        if lang == "en":
            return EN_TAG_ID
        raise ValueError(f"Unknown language code: {lang!r}")

    def encode(self, text: str, lang: str) -> list[int]:
        """Encode ``text`` into global vocabulary ids.

        Local ``<unk>`` tokens (id 0 in the per-language model) are
        remapped to the global ``<unk>`` id.  Special tokens (lang
        tags, ``<bos>``, ``<eos>``) are *not* added here — they are the
        caller's responsibility.
        """
        if lang == "ng":
            local_ids = self._ng.EncodeAsIds(text)
            offset = self._ng_offset
        elif lang == "en":
            local_ids = self._en.EncodeAsIds(text)
            offset = self._en_offset
        else:
            raise ValueError(f"Unknown language code: {lang!r}")
        return [
            UNK_ID if lid == 0 else lid + offset
            for lid in local_ids
        ]

    def decode(self, ids: list[int], lang: str) -> str:
        """Decode a sequence of global ids back to text in ``lang``.

        Ids outside the target language's range, or inside the shared
        special range, are skipped silently.
        """
        if lang == "ng":
            start, end = self.neuroglot_range
            offset = self._ng_offset
            sp = self._ng
        elif lang == "en":
            start, end = self.english_range
            offset = self._en_offset
            sp = self._en
        else:
            raise ValueError(f"Unknown language code: {lang!r}")
        local_ids = [gid - offset for gid in ids if start <= gid < end]
        return sp.DecodeIds(local_ids)


def load_tokenizer(config: UNMTConfig) -> BilingualTokenizer:
    """Load both per-language tokenizers for a config.

    Raises ``FileNotFoundError`` if either model has not been trained.
    """
    output_dir = Path(config.output_dir)
    ng_model = output_dir / f"{config.neuroglot_tokenizer_prefix}.model"
    en_model = output_dir / f"{config.english_tokenizer_prefix}.model"
    if not ng_model.exists() or not en_model.exists():
        raise FileNotFoundError(
            f"Tokenizers not trained: expected {ng_model} and {en_model}. "
            f"Run `lfm unmt tokenize <config>` first."
        )
    return BilingualTokenizer(
        TokenizerArtifacts(neuroglot_model=ng_model, english_model=en_model),
    )

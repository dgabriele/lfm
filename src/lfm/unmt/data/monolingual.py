"""Monolingual dataset loaders for unsupervised NMT.

A ``MonolingualDataset`` reads a single-language corpus, encodes each
line into global vocabulary ids on demand via the shared
:class:`BilingualTokenizer`, and emits ``(clean, noised)`` pairs for
denoising autoencoder training.  During backtranslation the clean
sequence becomes the target and the current translator produces the
source from the other language.

Memory model: tokenization is lazy and per-sample.  The corpus is
loaded as a list of raw strings — for the 900K Neuroglot and 300K
English corpora that is a few hundred MB, comfortably within budget.
"""

from __future__ import annotations

import json
import logging
import random
from pathlib import Path

import torch
from torch.utils.data import Dataset

from lfm.unmt.config import UNMTConfig
from lfm.unmt.data.noise import NoiseConfig, apply_noise
from lfm.unmt.tokenizer import (
    BOS_ID,
    EOS_ID,
    MASK_ID,
    PAD_ID,
    BilingualTokenizer,
)

logger = logging.getLogger(__name__)


def _read_corpus_lines(path: Path) -> list[str]:
    """Read a corpus into a list of text lines, auto-detecting JSONL."""
    lines: list[str] = []
    with open(path, encoding="utf-8") as f:
        first = ""
        for line in f:
            first = line.strip()
            if first:
                break
        if not first:
            return lines

        is_jsonl = first.startswith("{")
        if is_jsonl:
            try:
                lines.append(json.loads(first)["text"].strip())
            except (KeyError, json.JSONDecodeError):
                pass
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    lines.append(json.loads(line)["text"].strip())
                except (KeyError, json.JSONDecodeError):
                    continue
        else:
            lines.append(first)
            for line in f:
                line = line.strip()
                if line:
                    lines.append(line)
    return lines


class MonolingualDataset(Dataset):
    """Monolingual corpus yielding ``(clean, noised)`` token pairs.

    Each ``__getitem__`` call returns a dict with:

    * ``clean_ids``: clean token sequence as ``torch.LongTensor``.
      Layout: ``[BOS, <lang>, bpe_ids..., EOS]``.  Length capped at
      ``max_len``.
    * ``noised_ids``: noise-perturbed version of the BPE ids, wrapped
      in the same ``[BOS, <lang>, ..., EOS]`` envelope.  Used as the
      denoising autoencoder input.

    The noise is applied to the BPE interior only — the language tag
    and EOS are always preserved so the model never has to guess the
    language from corrupted input.

    Args:
        corpus_path: Path to the monolingual corpus file.
        tokenizer: Shared bilingual tokenizer.
        lang: Language code (``"ng"`` or ``"en"``).
        max_len: Maximum sequence length including special tokens.
        noise: Noise configuration for DAE corruption.
        seed: Base seed used to derive per-example RNGs.
    """

    def __init__(
        self,
        corpus_path: str | Path,
        tokenizer: BilingualTokenizer,
        lang: str,
        max_len: int,
        noise: NoiseConfig,
        seed: int = 0,
    ) -> None:
        corpus_path = Path(corpus_path)
        if not corpus_path.exists():
            raise FileNotFoundError(f"Corpus not found: {corpus_path}")

        self._tokenizer = tokenizer
        self._lang = lang
        self._lang_tag_id = tokenizer.lang_tag_id(lang)
        self._max_len = max_len
        self._noise = noise
        self._seed = seed

        logger.info("Reading monolingual corpus from %s", corpus_path)
        self._lines = _read_corpus_lines(corpus_path)
        logger.info(
            "  loaded %d %s lines from %s",
            len(self._lines), lang, corpus_path.name,
        )

    def __len__(self) -> int:
        return len(self._lines)

    @property
    def lang(self) -> str:
        return self._lang

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        text = self._lines[idx]
        # Deterministic per-example rng so noise is reproducible.
        # Python 3.12+ dropped tuple seeds, so derive a scalar from
        # (seed, idx) using a stable large-prime hash.
        rng = random.Random(self._seed * 2_147_483_647 + idx)

        bpe_ids = self._tokenizer.encode(text, self._lang)
        # Reserve 3 positions for BOS, lang tag, EOS
        bpe_ids = bpe_ids[: self._max_len - 3]

        clean = [BOS_ID, self._lang_tag_id] + bpe_ids + [EOS_ID]

        noised_bpe = apply_noise(bpe_ids, self._noise, rng)
        noised = [BOS_ID, self._lang_tag_id] + noised_bpe + [EOS_ID]

        return {
            "clean_ids": torch.tensor(clean, dtype=torch.long),
            "noised_ids": torch.tensor(noised, dtype=torch.long),
        }


def build_noise_config(config: UNMTConfig) -> NoiseConfig:
    """Build a ``NoiseConfig`` from a ``UNMTConfig``.

    The mask token id is fixed at the shared global ``<mask>`` position.
    """
    return NoiseConfig(
        drop_prob=config.word_drop_prob,
        mask_prob=config.word_mask_prob,
        swap_window=config.word_swap_window,
        mask_token_id=MASK_ID,
    )


def pad_batch(
    batch: list[dict[str, torch.Tensor]],
    pad_id: int = PAD_ID,
) -> dict[str, torch.Tensor]:
    """Collate variable-length examples, padding to the batch maximum."""
    def _pad_field(name: str) -> tuple[torch.Tensor, torch.Tensor]:
        tensors = [ex[name] for ex in batch]
        max_len = max(t.size(0) for t in tensors)
        padded = torch.full(
            (len(tensors), max_len), pad_id, dtype=torch.long,
        )
        mask = torch.zeros((len(tensors), max_len), dtype=torch.long)
        for i, t in enumerate(tensors):
            padded[i, : t.size(0)] = t
            mask[i, : t.size(0)] = 1
        return padded, mask

    clean_ids, clean_mask = _pad_field("clean_ids")
    noised_ids, noised_mask = _pad_field("noised_ids")
    return {
        "clean_ids": clean_ids,
        "clean_mask": clean_mask,
        "noised_ids": noised_ids,
        "noised_mask": noised_mask,
    }

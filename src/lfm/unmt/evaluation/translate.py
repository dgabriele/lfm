"""Batch translation utility for a trained UNMT model.

Thin wrapper around :func:`lfm.unmt.training.backtranslation.greedy_translate`
that handles text-level input/output: accepts raw strings in the
source language, tokenizes them with the appropriate per-language
sentencepiece model, runs greedy decode, strips special tokens from
the output, and decodes back to text.

Language restriction is enabled by default so the decoder cannot
emit source-language tokens in its output — this is the same safety
rail used during backtranslation.  It can be turned off for
experimentation.
"""

from __future__ import annotations

import logging
from pathlib import Path

import torch

from lfm.unmt.config import UNMTConfig
from lfm.unmt.model.transformer import build_model
from lfm.unmt.tokenizer import (
    BOS_ID,
    EN_TAG_ID,
    EOS_ID,
    NG_TAG_ID,
    PAD_ID,
    BilingualTokenizer,
    load_tokenizer,
)
from lfm.unmt.training.backtranslation import greedy_translate

logger = logging.getLogger(__name__)


def load_model_from_checkpoint(
    config: UNMTConfig,
    checkpoint_path: Path | str | None = None,
):
    """Load a trained model from a training checkpoint.

    Returns ``(model, tokenizer)``.  The model is placed on the
    configured device and put in eval mode.
    """
    device = torch.device(
        config.device if torch.cuda.is_available() else "cpu",
    )
    tokenizer = load_tokenizer(config)
    model = build_model(config, tokenizer).to(device)

    ckpt_path = Path(checkpoint_path or Path(config.output_dir) / "latest.pt")
    if not ckpt_path.exists():
        raise FileNotFoundError(
            f"UNMT checkpoint not found: {ckpt_path}. "
            f"Run `lfm unmt train <config>` first."
        )
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"])
    model.eval()
    logger.info(
        "Loaded UNMT model from %s (step=%d)",
        ckpt_path, ckpt.get("step", 0),
    )
    return model, tokenizer


def _encode_source(
    tokenizer: BilingualTokenizer,
    texts: list[str],
    source_lang: str,
    max_len: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Tokenize source texts into padded ``(ids, mask)`` tensors."""
    lang_tag = tokenizer.lang_tag_id(source_lang)
    sequences: list[list[int]] = []
    for text in texts:
        bpe = tokenizer.encode(text, source_lang)
        bpe = bpe[: max_len - 3]
        ids = [BOS_ID, lang_tag] + bpe + [EOS_ID]
        sequences.append(ids)

    width = max(len(s) for s in sequences)
    ids = torch.full(
        (len(sequences), width), PAD_ID, dtype=torch.long, device=device,
    )
    mask = torch.zeros(
        (len(sequences), width), dtype=torch.long, device=device,
    )
    for i, s in enumerate(sequences):
        ids[i, : len(s)] = torch.tensor(s, dtype=torch.long, device=device)
        mask[i, : len(s)] = 1
    return ids, mask


def _decode_output(
    tokenizer: BilingualTokenizer,
    token_grid: torch.Tensor,
    target_lang: str,
) -> list[str]:
    """Strip BOS/tag/EOS/PAD and decode to text."""
    decoded = []
    for row in token_grid.tolist():
        trimmed: list[int] = []
        for tid in row:
            if tid == PAD_ID:
                break
            if tid == EOS_ID:
                break
            if tid in (BOS_ID, NG_TAG_ID, EN_TAG_ID):
                continue
            trimmed.append(tid)
        decoded.append(tokenizer.decode(trimmed, target_lang))
    return decoded


def translate_texts(
    model,
    tokenizer: BilingualTokenizer,
    texts: list[str],
    source_lang: str,
    target_lang: str,
    max_len: int,
    restrict_target_language: bool = True,
) -> list[str]:
    """Translate a list of source-language strings to the target language.

    Args:
        model: Trained :class:`SharedNMTTransformer`.
        tokenizer: Shared bilingual tokenizer.
        texts: Source-language strings.
        source_lang: ``"ng"`` or ``"en"``.
        target_lang: The other of ``"ng"`` or ``"en"``.
        max_len: Maximum target length (including framing tokens).
        restrict_target_language: If ``True``, restrict the decoder's
            logits to the target language's BPE range + shared
            specials.  Default ``True``.

    Returns:
        Translated strings, in the same order.
    """
    if source_lang == target_lang:
        raise ValueError("source_lang and target_lang must differ")

    device = next(model.parameters()).device
    src_ids, src_mask = _encode_source(
        tokenizer, texts, source_lang, max_len, device,
    )

    if target_lang == "en":
        tag_id = EN_TAG_ID
        restrict = tokenizer.english_range if restrict_target_language else None
    else:
        tag_id = NG_TAG_ID
        restrict = tokenizer.neuroglot_range if restrict_target_language else None

    output = greedy_translate(
        model,
        src_ids=src_ids,
        src_mask=src_mask,
        target_lang_tag_id=tag_id,
        max_len=max_len,
        restrict_range=restrict,
    )
    return _decode_output(tokenizer, output, target_lang)

"""Backtranslation loss for unsupervised NMT.

Backtranslation is the core bootstrap mechanism of the Lample/Artetxe
recipe: the model generates its own pseudo-pairs and then trains on
them in the reverse direction.  Once the model has any translation
capability at all (even weak), BT iteratively improves it.

Per direction (e.g. Neuroglot → English → Neuroglot):

1. **Generate step** — with the current model frozen in eval mode and
   no gradients, greedily translate the clean source batch into the
   target language.  The output is a synthetic ``(src, pseudo_tgt)``
   pair where the pseudo-target is whatever the model currently
   believes the source means in the other language.
2. **Train step** — now treat the pseudo-target as *input* and the
   original source as *output*, and run a standard teacher-forced
   cross-entropy step.  Gradients flow through the model's ability to
   *recover* the source from the pseudo-target, which drives the
   model to encode meaning that survives the round trip.

The full trainer alternates BT steps between ng→en→ng and en→ng→en
so that both directions get training signal.

To avoid the model producing source-language tokens inside the
target, the logits can be optionally restricted to a target-language
token range during the greedy decode.  This is a well-known
stabilization trick — without it, early in training the model will
freely emit source-language tokens because nothing has taught it not
to yet.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from lfm.unmt.model.transformer import SharedNMTTransformer
from lfm.unmt.tokenizer import (
    BOS_ID,
    EOS_ID,
    NUM_SPECIAL,
    PAD_ID,
)


@torch.no_grad()
def greedy_translate(
    model: SharedNMTTransformer,
    src_ids: torch.Tensor,
    src_mask: torch.Tensor,
    target_lang_tag_id: int,
    max_len: int,
    restrict_range: tuple[int, int] | None = None,
) -> torch.Tensor:
    """Greedy decode from source to target language.

    Args:
        model: Shared seq2seq model.
        src_ids: ``(B, S)`` source token ids including BOS + src-lang
            tag + ... + EOS.
        src_mask: ``(B, S)`` attention mask, ``1`` for real tokens.
        target_lang_tag_id: Global vocabulary id of the target language
            tag (``NG_TAG_ID`` or ``EN_TAG_ID``).
        max_len: Maximum target length including BOS + tag + EOS.
        restrict_range: Optional ``(start, end)`` tuple over the global
            vocabulary.  If set, logits outside this range are masked
            to ``-inf`` before argmax, effectively forbidding the
            model from generating non-target-language tokens.  The
            shared specials and EOS are always allowed regardless.

    Returns:
        ``(B, T)`` target token tensor on the source's device.  All
        sequences are padded with ``PAD_ID`` after EOS.
    """
    device = src_ids.device
    was_training = model.training
    model.eval()
    try:
        src_pad_mask = src_mask == 0
        memory = model.encode(src_ids, src_pad_mask)

        batch = src_ids.size(0)
        tgt = torch.full(
            (batch, 2), BOS_ID, dtype=torch.long, device=device,
        )
        tgt[:, 1] = target_lang_tag_id
        finished = torch.zeros(batch, dtype=torch.bool, device=device)

        # Precompute the language-restriction additive mask if needed.
        allow_bias: torch.Tensor | None = None
        if restrict_range is not None:
            start, end = restrict_range
            allow_bias = torch.full(
                (model.vocab_size,), float("-inf"), device=device,
            )
            allow_bias[:NUM_SPECIAL] = 0.0
            allow_bias[start:end] = 0.0

        for _ in range(max_len - 2):
            tgt_pad_mask = torch.zeros_like(tgt, dtype=torch.bool)
            hidden = model.decode(
                tgt, tgt_pad_mask, memory, src_pad_mask,
            )
            last_logits = model.logits_from_hidden(hidden[:, -1])
            if allow_bias is not None:
                last_logits = last_logits + allow_bias

            next_tok = last_logits.argmax(dim=-1)
            pad_tensor = torch.full_like(next_tok, PAD_ID)
            next_tok = torch.where(finished, pad_tensor, next_tok)
            tgt = torch.cat([tgt, next_tok.unsqueeze(1)], dim=1)
            finished = finished | (next_tok == EOS_ID)
            if finished.all():
                break

        return tgt
    finally:
        if was_training:
            model.train()


def compute_bt_loss(
    model: SharedNMTTransformer,
    batch: dict[str, torch.Tensor],
    target_lang_tag_id: int,
    target_token_range: tuple[int, int],
    max_len: int,
) -> torch.Tensor:
    """One backtranslation step from the given language to the other.

    Args:
        model: Shared seq2seq model.
        batch: Clean monolingual batch for the *source* language (same
            format as :class:`MonolingualDataset` output).
        target_lang_tag_id: Global id of the *target* language tag.
            During the generate step this is what the decoder is
            seeded with.  During the train step the source's own tag
            appears in its own sequence.
        target_token_range: ``(start, end)`` range of target-language
            BPE tokens in the global vocabulary, used to restrict
            logits during greedy generation.
        max_len: Maximum target length for the greedy decode.

    Returns:
        Scalar BT loss.
    """
    clean_src = batch["clean_ids"]
    clean_src_mask = batch["clean_mask"]

    # Generate pseudo-target with no grad and restricted logits.
    pseudo_tgt = greedy_translate(
        model,
        clean_src,
        clean_src_mask,
        target_lang_tag_id=target_lang_tag_id,
        max_len=max_len,
        restrict_range=target_token_range,
    )
    pseudo_tgt_mask = (pseudo_tgt != PAD_ID).long()

    # Train the reverse direction: pseudo_tgt → clean_src.
    tgt_in = clean_src[:, :-1]
    tgt_in_mask = clean_src_mask[:, :-1]
    tgt_out = clean_src[:, 1:]

    logits = model(
        pseudo_tgt, pseudo_tgt_mask, tgt_in, tgt_in_mask,
    )
    loss = F.cross_entropy(
        logits.reshape(-1, logits.size(-1)),
        tgt_out.reshape(-1),
        ignore_index=PAD_ID,
    )
    return loss

"""Denoising autoencoder loss for unsupervised NMT.

The DAE objective is the simplest and most stable of the two losses
the UNMT trainer combines.  Given a clean monolingual sequence, we
feed the model a noise-corrupted version of it as the encoder input
and train the decoder (via teacher forcing) to reconstruct the clean
version.  Applied symmetrically to both languages with shared weights,
this forces the encoder to learn to extract language-agnostic meaning
from noisy surface forms — the foundation that backtranslation builds
on top of.

The noise is applied upstream by
:class:`lfm.unmt.data.monolingual.MonolingualDataset`, which emits
``(clean_ids, noised_ids)`` pairs already framed with BOS + lang-tag
+ EOS.  This module only sees the batched tensors and computes the
cross-entropy loss.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from lfm.unmt.model.transformer import SharedNMTTransformer
from lfm.unmt.tokenizer import PAD_ID


def compute_dae_loss(
    model: SharedNMTTransformer,
    batch: dict[str, torch.Tensor],
) -> torch.Tensor:
    """Cross-entropy reconstruction loss for one language's DAE step.

    Args:
        model: Shared-weight seq2seq transformer.
        batch: Dict with ``clean_ids``, ``clean_mask``, ``noised_ids``,
            ``noised_mask`` — all ``(B, *)`` long tensors on the
            model's device.

    Returns:
        Scalar loss.
    """
    src_ids = batch["noised_ids"]
    src_mask = batch["noised_mask"]
    tgt_ids = batch["clean_ids"]
    tgt_mask = batch["clean_mask"]

    # Teacher-forcing: decoder reads everything up to position t-1
    # and predicts the token at position t.
    tgt_in = tgt_ids[:, :-1]
    tgt_in_mask = tgt_mask[:, :-1]
    tgt_out = tgt_ids[:, 1:]

    logits = model(src_ids, src_mask, tgt_in, tgt_in_mask)
    loss = F.cross_entropy(
        logits.reshape(-1, logits.size(-1)),
        tgt_out.reshape(-1),
        ignore_index=PAD_ID,
    )
    return loss

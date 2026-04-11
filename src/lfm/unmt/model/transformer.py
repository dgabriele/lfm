"""Shared-weight seq2seq transformer for unsupervised NMT.

One model handles both translation directions (Neuroglot↔English) by
sharing all encoder-decoder weights.  The language of each input or
target sequence is indicated by the language tag token at position 1
(after ``<bos>``), which the model reads naturally as context without
any special routing logic.

Design choices:

* **Tied embeddings**: a single ``nn.Embedding`` matrix of size
  ``(global_vocab_size, model_dim)`` is used for encoder input,
  decoder input, and the output projection (weight-tied head).  This
  halves the parameter count and forces the model to build a shared
  representation space across both languages — the same mechanism that
  enables cross-lingual transfer in multilingual LLMs.
* **Pre-LN transformer**: layer normalization applied before each
  sublayer.  More stable than post-LN at deep depths and with small
  batch sizes.
* **Sinusoidal positional encoding**: no learned positional embeddings,
  so the model can generalize to slightly longer sequences at
  inference than it saw during training.
* **GELU activation** in feed-forward layers, matching modern
  transformer defaults.
* **MUSE initialization**: the embedding matrix is populated from the
  Stage 2 skip-gram + MUSE alignment outputs whenever available.
  Neuroglot rows are multiplied by the learned rotation so that both
  languages start in a shared coordinate frame.  Special tokens get
  standard small-Gaussian init.
"""

from __future__ import annotations

import logging
import math
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from lfm.unmt.config import UNMTConfig
from lfm.unmt.embeddings.muse_align import load_alignment
from lfm.unmt.embeddings.skipgram import load_embeddings
from lfm.unmt.tokenizer import PAD_ID, BilingualTokenizer

logger = logging.getLogger(__name__)


def _sinusoidal_positional_encoding(
    max_len: int, d_model: int,
) -> torch.Tensor:
    """Classic sinusoidal encoding from Vaswani et al. 2017.

    Shape: ``(max_len, d_model)``.  Computed once and stored as a
    buffer so it moves with the model when placed on a device.
    """
    pe = torch.zeros(max_len, d_model)
    position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
    div_term = torch.exp(
        torch.arange(0, d_model, 2, dtype=torch.float)
        * (-math.log(10000.0) / d_model)
    )
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    return pe


class SharedNMTTransformer(nn.Module):
    """Bidirectional encoder-decoder transformer with tied vocabulary.

    The same module runs both translation directions: feed Neuroglot
    as source and English as target for ng→en, swap for en→ng.  No
    parameters are language-specific beyond the shared embedding rows,
    which themselves sit in a single matrix.

    Args:
        config: UNMT configuration (model_dim, n_layers, n_heads,
            ff_dim, dropout, max_len come from here).
        global_vocab_size: Size of the shared concatenated vocabulary
            from :class:`BilingualTokenizer`.
    """

    def __init__(
        self,
        config: UNMTConfig,
        global_vocab_size: int,
    ) -> None:
        super().__init__()
        self.config = config
        self.d_model = config.model_dim
        self.vocab_size = global_vocab_size

        # Shared embedding table used by encoder, decoder, and output head.
        self.embed = nn.Embedding(
            self.vocab_size, self.d_model, padding_idx=PAD_ID,
        )
        # Small-Gaussian init matching pre-LN transformer defaults.
        nn.init.normal_(self.embed.weight, mean=0.0, std=self.d_model**-0.5)
        with torch.no_grad():
            self.embed.weight[PAD_ID].zero_()

        # Sinusoidal positional encoding, buffer so it follows .to(device).
        # Allow a bit of slack over max_len for inference-time longer outputs.
        pe = _sinusoidal_positional_encoding(
            max_len=max(512, config.max_len * 2),
            d_model=self.d_model,
        )
        self.register_buffer("pos_encoding", pe, persistent=False)

        # Dropout applied after embedding + positional add, matching
        # the original transformer paper.
        self.embed_dropout = nn.Dropout(config.dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=config.n_heads,
            dim_feedforward=config.ff_dim,
            dropout=config.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=config.n_layers,
            norm=nn.LayerNorm(self.d_model),
        )

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=self.d_model,
            nhead=config.n_heads,
            dim_feedforward=config.ff_dim,
            dropout=config.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(
            decoder_layer,
            num_layers=config.n_layers,
            norm=nn.LayerNorm(self.d_model),
        )

        # Output projection is tied to the embedding matrix — the
        # forward pass explicitly does ``hidden @ self.embed.weight.t()``
        # so there's no standalone head module to configure.

    # -- Encoding / decoding helpers --

    def _add_positions(self, embedded: torch.Tensor) -> torch.Tensor:
        """Add sinusoidal positional encoding to an embedded batch."""
        seq_len = embedded.size(1)
        return embedded + self.pos_encoding[:seq_len].unsqueeze(0)

    def encode(
        self,
        src_ids: torch.Tensor,
        src_pad_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Encode a source batch to memory.

        Args:
            src_ids: ``(B, S)`` long tensor of global token ids.
            src_pad_mask: ``(B, S)`` bool tensor — ``True`` where the
                position is padding and should be masked out of
                attention.

        Returns:
            ``(B, S, D)`` contextualized encoder memory.
        """
        x = self.embed(src_ids) * math.sqrt(self.d_model)
        x = self._add_positions(x)
        x = self.embed_dropout(x)
        return self.encoder(x, src_key_padding_mask=src_pad_mask)

    def decode(
        self,
        tgt_ids: torch.Tensor,
        tgt_pad_mask: torch.Tensor,
        memory: torch.Tensor,
        memory_pad_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Decode target tokens against encoder memory.

        Returns hidden states ``(B, T, D)``.  Apply
        :meth:`logits_from_hidden` to get vocabulary logits.
        """
        y = self.embed(tgt_ids) * math.sqrt(self.d_model)
        y = self._add_positions(y)
        y = self.embed_dropout(y)

        seq_len = tgt_ids.size(1)
        causal_mask = torch.triu(
            torch.ones(
                seq_len, seq_len, dtype=torch.bool, device=tgt_ids.device,
            ),
            diagonal=1,
        )

        return self.decoder(
            y, memory,
            tgt_mask=causal_mask,
            tgt_key_padding_mask=tgt_pad_mask,
            memory_key_padding_mask=memory_pad_mask,
        )

    def logits_from_hidden(self, hidden: torch.Tensor) -> torch.Tensor:
        """Project decoder hidden states to vocabulary logits via tied weights."""
        return hidden @ self.embed.weight.t()

    def forward(
        self,
        src_ids: torch.Tensor,
        src_mask: torch.Tensor,
        tgt_ids: torch.Tensor,
        tgt_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Full forward pass returning vocabulary logits for the decoder.

        Args:
            src_ids: ``(B, S)`` source token ids.
            src_mask: ``(B, S)`` attention mask — ``1`` for real tokens,
                ``0`` for padding.  (Gets inverted to pad mask
                internally.)
            tgt_ids: ``(B, T)`` target token ids.
            tgt_mask: ``(B, T)`` target attention mask.

        Returns:
            ``(B, T, V)`` logits over the full global vocabulary.
        """
        src_pad_mask = src_mask == 0
        tgt_pad_mask = tgt_mask == 0
        memory = self.encode(src_ids, src_pad_mask)
        hidden = self.decode(tgt_ids, tgt_pad_mask, memory, src_pad_mask)
        return self.logits_from_hidden(hidden)

    # -- Parameter accounting --

    def num_parameters(self) -> dict[str, int]:
        """Return parameter counts broken down by component."""
        counts = {
            "embedding": self.embed.weight.numel(),
            "encoder": sum(p.numel() for p in self.encoder.parameters()),
            "decoder": sum(p.numel() for p in self.decoder.parameters()),
        }
        counts["total"] = sum(counts.values())
        return counts


def build_model(
    config: UNMTConfig, tokenizer: BilingualTokenizer,
) -> SharedNMTTransformer:
    """Construct a :class:`SharedNMTTransformer` for the given tokenizer."""
    return SharedNMTTransformer(
        config=config,
        global_vocab_size=tokenizer.global_vocab_size,
    )


def initialize_from_muse(
    model: SharedNMTTransformer,
    config: UNMTConfig,
    tokenizer: BilingualTokenizer,
) -> dict[str, bool]:
    """Populate the embedding matrix from Stage 2 artifacts.

    - Neuroglot rows ``[ng_start, ng_end)`` are filled from
      ``embed_neuroglot.pt`` rotated by the MUSE alignment matrix so
      they end up in the English coordinate frame.
    - English rows ``[en_start, en_end)`` are filled from
      ``embed_english.pt`` unchanged (they already define the target
      frame).
    - Special rows ``[0, 7)`` keep their Gaussian init.

    Returns a dict summarizing which sources were found and applied.
    If the alignment file is missing the Neuroglot rows are copied
    un-rotated (still useful, just not cross-lingually aligned).
    """
    output_dir = Path(config.output_dir)
    ng_emb_path = output_dir / "embed_neuroglot.pt"
    en_emb_path = output_dir / "embed_english.pt"
    align_path = output_dir / "alignment_ng2en.pt"

    applied = {"neuroglot": False, "english": False, "rotation": False}

    if not ng_emb_path.exists() or not en_emb_path.exists():
        logger.warning(
            "MUSE artifacts not found (ng=%s, en=%s) — embedding matrix "
            "left at random init.",
            ng_emb_path.exists(), en_emb_path.exists(),
        )
        return applied

    ng_weights, ng_meta = load_embeddings(ng_emb_path)
    en_weights, en_meta = load_embeddings(en_emb_path)

    if ng_weights.size(1) != model.d_model:
        raise ValueError(
            f"Neuroglot embedding dim {ng_weights.size(1)} "
            f"!= model dim {model.d_model}"
        )
    if en_weights.size(1) != model.d_model:
        raise ValueError(
            f"English embedding dim {en_weights.size(1)} "
            f"!= model dim {model.d_model}"
        )

    W = None
    if align_path.exists():
        alignment = load_alignment(align_path)
        W = alignment.W
        applied["rotation"] = True

    ng_start, ng_end = tokenizer.neuroglot_range
    en_start, en_end = tokenizer.english_range

    with torch.no_grad():
        if W is not None:
            ng_rotated = ng_weights @ W.t()
        else:
            ng_rotated = ng_weights
        model.embed.weight[ng_start:ng_end].copy_(ng_rotated)
        applied["neuroglot"] = True

        model.embed.weight[en_start:en_end].copy_(en_weights)
        applied["english"] = True

        model.embed.weight[PAD_ID].zero_()

    logger.info(
        "Embedding init: neuroglot=%s english=%s rotation=%s",
        applied["neuroglot"], applied["english"], applied["rotation"],
    )
    return applied

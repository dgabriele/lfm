"""Sentence encoder for DepTreeVAE."""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor

from lfm.generator.dep_tree_vae.config import DepTreeVAEConfig


class SentenceEncoder(nn.Module):
    """Encode a token sequence to (mu, logvar) via transformer + mean pooling.

    Produces the full latent vector; the caller splits into z_struct
    and z_content.
    """

    def __init__(self, vocab_size: int, cfg: DepTreeVAEConfig) -> None:
        super().__init__()
        h = cfg.encoder_hidden_dim
        latent_dim = cfg.latent.total_dim

        self.token_embedding = nn.Embedding(vocab_size, h)
        self.pos_embedding = nn.Embedding(cfg.max_seq_len, h)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=h,
            nhead=cfg.encoder_num_heads,
            dim_feedforward=h * 4,
            dropout=cfg.encoder_dropout,
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=cfg.encoder_num_layers,
        )

        # Project pooled representation to 2x latent dim (mu + logvar)
        self.to_latent = nn.Linear(h, latent_dim * 2)

    def forward(
        self,
        tokens: Tensor,
        lengths: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """Encode tokens to (mu, logvar).

        Args:
            tokens: ``(B, S)`` token ids (role + IPA interleaved).
            lengths: ``(B,)`` actual sequence lengths.

        Returns:
            mu: ``(B, latent_dim)``
            logvar: ``(B, latent_dim)``
        """
        b, s = tokens.shape
        device = tokens.device

        mask = torch.arange(s, device=device).unsqueeze(0) < lengths.unsqueeze(1)
        pos = torch.arange(s, device=device).unsqueeze(0)

        x = self.token_embedding(tokens) + self.pos_embedding(pos)
        x = self.encoder(x, src_key_padding_mask=~mask)

        # Mean pooling over valid positions
        x_masked = x * mask.unsqueeze(-1).float()
        pooled = x_masked.sum(dim=1) / lengths.unsqueeze(-1).float().clamp(min=1)

        params = self.to_latent(pooled)
        mu, logvar = params.chunk(2, dim=-1)
        return mu, logvar

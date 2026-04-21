"""Completeness scorer model — small transformer binary classifier.

Judges whether a token sequence forms a structurally complete thought.
Operates on token embeddings (shared with the VAE decoder) so it can
score soft-token output during training via differentiable forward pass.

The scorer is trained to recognize distributional syntax — structural
patterns that make a sentence complete — purely from positional and
relational cues. A word is recognized as a "verb" because of where it
appears, not because it's in a vocabulary. This generalizes to fully
novel/alien vocabulary.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor

from lfm.config import LFMBaseConfig


class CompletenessConfig(LFMBaseConfig):
    """Configuration for the completeness scorer."""

    vocab_size: int = 8050
    d_model: int = 256
    num_layers: int = 4
    num_heads: int = 4
    max_seq_len: int = 200
    dropout: float = 0.1

    # Training
    batch_size: int = 256
    lr: float = 1e-3
    num_epochs: int = 5
    seed: int = 42


class CompletenessScorer(nn.Module):
    """Binary classifier: is this token sequence a complete thought?

    Architecture: token embedding → positional embedding → transformer
    encoder → mean pool → MLP → sigmoid score.

    For differentiable use during VAE training, accepts soft token
    inputs (logits or probabilities over vocabulary) via score_soft().
    """

    def __init__(self, cfg: CompletenessConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.token_embedding = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.pos_embedding = nn.Embedding(cfg.max_seq_len, cfg.d_model)
        self.input_norm = nn.LayerNorm(cfg.d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=cfg.d_model,
            nhead=cfg.num_heads,
            dim_feedforward=cfg.d_model * 4,
            dropout=cfg.dropout,
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=cfg.num_layers)

        self.head = nn.Sequential(
            nn.Linear(cfg.d_model, cfg.d_model),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.d_model, 1),
        )

    def forward(self, token_ids: Tensor, lengths: Tensor) -> Tensor:
        """Score hard token sequences.

        Args:
            token_ids: (B, S) integer token IDs.
            lengths: (B,) valid sequence lengths.

        Returns:
            scores: (B,) logit scores (before sigmoid).
        """
        b, s = token_ids.shape
        device = token_ids.device

        pos = torch.arange(s, device=device).unsqueeze(0)
        h = self.token_embedding(token_ids) + self.pos_embedding(pos.clamp(max=self.cfg.max_seq_len - 1))
        h = self.input_norm(h)

        padding_mask = torch.arange(s, device=device).unsqueeze(0) >= lengths.unsqueeze(1)
        h = self.encoder(h, src_key_padding_mask=padding_mask)

        # Mean pool over valid positions
        valid = (~padding_mask).unsqueeze(-1).float()
        pooled = (h * valid).sum(dim=1) / valid.sum(dim=1).clamp(min=1)

        return self.head(pooled).squeeze(-1)

    def score_soft(self, token_logits: Tensor, lengths: Tensor) -> Tensor:
        """Score soft token output (differentiable).

        Used during VAE training: the diffusion decoder outputs logits,
        softmax gives a distribution over vocabulary, matrix multiply
        with embedding weights gives soft token embeddings.

        Args:
            token_logits: (B, S, V) raw logits from decoder.
            lengths: (B,) valid sequence lengths.

        Returns:
            scores: (B,) logit scores (differentiable to decoder).
        """
        b, s, v = token_logits.shape
        device = token_logits.device

        # Soft token embeddings: weighted sum of embedding vectors
        probs = torch.softmax(token_logits, dim=-1)
        h = probs @ self.token_embedding.weight  # (B, S, d_model)

        pos = torch.arange(s, device=device).unsqueeze(0)
        h = h + self.pos_embedding(pos.clamp(max=self.cfg.max_seq_len - 1))
        h = self.input_norm(h)

        padding_mask = torch.arange(s, device=device).unsqueeze(0) >= lengths.unsqueeze(1)
        h = self.encoder(h, src_key_padding_mask=padding_mask)

        valid = (~padding_mask).unsqueeze(-1).float()
        pooled = (h * valid).sum(dim=1) / valid.sum(dim=1).clamp(min=1)

        return self.head(pooled).squeeze(-1)

"""SynthLM: mT5-large with alien decoder vocabulary and embedding conditioning.

Two operating modes:
  * Phase 1 — uses the mT5 encoder (English text -> encoder hidden states).
    Only decoder embed_tokens and lm_head are new; encoder is frozen.
  * Phase 2 — replaces the mT5 encoder with a learned EmbeddingProjector.
    The mT5 decoder is frozen; only projector and length_head are trained.

The encoder and decoder use *separate* embedding tables so the encoder
retains its full multilingual English vocabulary while the decoder uses
the alien syllable vocabulary.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from transformers import MT5ForConditionalGeneration
from transformers.modeling_outputs import BaseModelOutput

from lfm.synth.config import SynthConfig


class EmbeddingProjector(nn.Module):
    """Maps a flat source embedding to n_prefix encoder-like hidden states."""

    def __init__(self, embedding_dim: int, d_model: int, n_prefix: int) -> None:
        super().__init__()
        self.n_prefix = n_prefix
        self.proj = nn.Sequential(
            nn.Linear(embedding_dim, d_model),
            nn.Tanh(),
            nn.Linear(d_model, d_model * n_prefix),
        )

    def forward(self, embedding: Tensor) -> tuple[Tensor, Tensor]:
        """Returns (hidden_states, attention_mask) — (B, n_prefix, d_model) and (B, n_prefix)."""
        B = embedding.size(0)
        hidden = self.proj(embedding).view(B, self.n_prefix, -1)
        mask = torch.ones(B, self.n_prefix, device=embedding.device, dtype=torch.long)
        return hidden, mask


class LengthHead(nn.Module):
    """Predicts alien token count from source embedding."""

    def __init__(self, embedding_dim: int, hidden_dim: int = 256) -> None:
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, embedding: Tensor) -> Tensor:
        """Returns scalar predicted length per sample — (B,)."""
        return self.mlp(embedding).squeeze(-1).clamp(min=4.0)


class SynthLM(nn.Module):
    """mT5-large with alien decoder vocab and two-mode conditioning.

    Args:
        config: SynthConfig.
        alien_vocab_size: Number of tokens in the alien vocabulary.
    """

    def __init__(self, config: SynthConfig, alien_vocab_size: int) -> None:
        super().__init__()
        self.config = config

        mt5: MT5ForConditionalGeneration = MT5ForConditionalGeneration.from_pretrained(
            config.base_model_name,
        )
        d_model: int = mt5.config.d_model

        # Detach decoder from the shared English embedding table.
        mt5.decoder.embed_tokens = nn.Embedding(alien_vocab_size, d_model)
        mt5.lm_head = nn.Linear(d_model, alien_vocab_size, bias=False)
        mt5.config.tie_word_embeddings = False

        # Align generation config to alien special token IDs.
        mt5.config.decoder_start_token_id = 2  # BOS_ID
        mt5.config.eos_token_id = 3            # EOS_ID
        mt5.config.pad_token_id = 0            # PAD_ID
        mt5.generation_config.eos_token_id = 3
        mt5.generation_config.pad_token_id = 0
        mt5.generation_config.decoder_start_token_id = 2
        mt5.generation_config.forced_eos_token_id = None

        self.mt5 = mt5
        self.projector = EmbeddingProjector(config.source_embedding_dim, d_model, config.n_prefix_tokens)
        self.length_head = LengthHead(config.source_embedding_dim)

    # ---- Phase 1 ----

    def forward_phase1(
        self,
        input_ids: Tensor,
        attention_mask: Tensor,
        labels: Tensor,
    ) -> Tensor:
        """CE loss: English text -> alien tokens via mT5 encoder."""
        return self.mt5(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
        ).loss

    # ---- Phase 2 ----

    def forward_phase2(
        self,
        source_embedding: Tensor,
        labels: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """CE loss + length MSE: source embedding -> alien tokens via projector.

        Returns:
            (lm_loss, length_loss) — both scalars.
        """
        hidden, mask = self.projector(source_embedding)
        encoder_outputs = BaseModelOutput(last_hidden_state=hidden)
        lm_loss = self.mt5(
            encoder_outputs=encoder_outputs,
            attention_mask=mask,
            labels=labels,
        ).loss

        length_pred = self.length_head(source_embedding)
        length_target = (labels != -100).float().sum(dim=-1)
        length_loss = F.mse_loss(length_pred, length_target)

        return lm_loss, length_loss

    # ---- Inference ----

    @torch.no_grad()
    def generate(
        self,
        source_embedding: Tensor,
        max_length: int | None = None,
    ) -> Tensor:
        """Generate alien token IDs from source embedding.

        Args:
            source_embedding: (B, embedding_dim).
            max_length: Hard cap on output tokens.  If None, predicted from
                the length head plus config.length_slack.

        Returns:
            (B, T) token ID tensor.
        """
        hidden, mask = self.projector(source_embedding)
        if max_length is None:
            predicted = self.length_head(source_embedding).round().long()
            max_length = int(predicted.max().item()) + self.config.length_slack

        encoder_outputs = BaseModelOutput(last_hidden_state=hidden)
        return self.mt5.generate(
            encoder_outputs=encoder_outputs,
            attention_mask=mask,
            max_length=max_length,
            num_beams=1,
        )

    # ---- Persistence helpers ----

    def save_phase1(self, path: str) -> None:
        # decoder_body state_dict already contains embed_tokens weights.
        torch.save({
            "lm_head": self.mt5.lm_head.state_dict(),
            "decoder_body": self.mt5.decoder.state_dict(),
        }, path)

    def load_phase1(self, path: str) -> None:
        ckpt = torch.load(path, map_location="cpu", weights_only=True)
        self.mt5.lm_head.load_state_dict(ckpt["lm_head"])
        self.mt5.decoder.load_state_dict(ckpt["decoder_body"])

    def save_phase2(self, path: str) -> None:
        torch.save({
            "projector": self.projector.state_dict(),
            "length_head": self.length_head.state_dict(),
        }, path)

    def load_phase2(self, path: str) -> None:
        ckpt = torch.load(path, map_location="cpu")
        self.projector.load_state_dict(ckpt["projector"])
        self.length_head.load_state_dict(ckpt["length_head"])

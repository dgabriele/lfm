"""Decoder backend abstraction for SynthLM.

DecoderBackend(ABC) defines the interface any causal-LM backend must implement.
CausalDecoderBackend wraps any HuggingFace AutoModelForCausalLM.

Body freeze state is managed explicitly:
  - __init__: body frozen (safe default for Phase 2).
  - unfreeze_body(): called by CipherTrainer so Phase 1 can fine-tune the body.
  - freeze_body(): called by ConditioningTrainer to lock the body for Phase 2.

embed_tokens is kept frozen throughout — native embedding space must stay
consistent so the Phase 2 projector can map into it reliably.

To add a new backend: subclass DecoderBackend and implement all abstract methods.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import torch
import torch.nn as nn
from torch import Tensor


class DecoderBackend(nn.Module, ABC):
    """Interface: LM body (freeze-controlled) with trainable alien cipher sub-vocabulary."""

    @abstractmethod
    def embed_native(self, input_ids: Tensor) -> Tensor:
        """(B, T) int64 → (B, T, D) via native token embedding."""

    @abstractmethod
    def embed_alien(self, alien_ids: Tensor) -> Tensor:
        """(B, T) int64 → (B, T, D) via alien token embedding table."""

    @abstractmethod
    def forward_hidden(
        self,
        inputs_embeds: Tensor,
        attention_mask: Tensor | None = None,
    ) -> Tensor:
        """(B, T, D) → (B, T, D) last hidden states from the LM body."""

    @abstractmethod
    def alien_logits(self, hidden: Tensor) -> Tensor:
        """(B, T, D) → (B, T, alien_vocab_size) via alien projection head."""

    @property
    @abstractmethod
    def d_model(self) -> int:
        """Hidden dimension of the LM body."""

    @property
    @abstractmethod
    def eos_token_id(self) -> int:
        """Native EOS token ID — used as the Phase 1 English/alien separator."""

    @abstractmethod
    def cipher_params(self) -> list[nn.Parameter]:
        """alien_emb + alien_head parameters (always trainable)."""

    @abstractmethod
    def body_params(self) -> list[nn.Parameter]:
        """Transformer layer parameters (trainable during Phase 1, frozen during Phase 2).
        Excludes embed_tokens and lm_head."""

    @abstractmethod
    def freeze_body(self) -> None:
        """Set body transformer layers to requires_grad=False."""

    @abstractmethod
    def unfreeze_body(self) -> None:
        """Set body transformer layers to requires_grad=True."""

    @property
    @abstractmethod
    def dtype(self) -> torch.dtype:
        """Compute dtype matching the LM body (e.g. bfloat16 for Qwen)."""


class CausalDecoderBackend(DecoderBackend):
    """Wraps any HuggingFace AutoModelForCausalLM.

    alien_emb and alien_head are always trainable.
    embed_tokens and lm_head are always frozen.
    Transformer layers (body_params) are frozen by default; call unfreeze_body()
    before Phase 1 training and freeze_body() before Phase 2.

    Body detection tries: ('model', 'transformer', 'gpt_neox', 'decoder').
    Embed detection tries: ('embed_tokens', 'wte', 'word_embeddings').
    """

    _BODY_ATTRS: tuple[str, ...] = ('model', 'transformer', 'gpt_neox', 'decoder')
    _EMBED_ATTRS: tuple[str, ...] = ('embed_tokens', 'wte', 'word_embeddings')

    def __init__(self, model_name: str, alien_vocab_size: int) -> None:
        super().__init__()
        from transformers import AutoModelForCausalLM

        lm = AutoModelForCausalLM.from_pretrained(model_name)
        for p in lm.parameters():
            p.requires_grad_(False)

        self._lm = lm
        d = lm.config.hidden_size
        dtype = next(lm.parameters()).dtype

        self._alien_emb = nn.Embedding(alien_vocab_size, d).to(dtype)
        self._alien_head = nn.Linear(d, alien_vocab_size, bias=False).to(dtype)

        self._body = self._find(lm, self._BODY_ATTRS, 'transformer body')
        self._embed_fn: nn.Embedding = self._find(self._body, self._EMBED_ATTRS, 'embed_tokens')
        # Prefix used to exclude native embed_tokens from body_params.
        self._embed_prefix: str = next(
            a for a in self._EMBED_ATTRS if hasattr(self._body, a)
        )

    @staticmethod
    def _find(obj: object, attrs: tuple[str, ...], label: str) -> nn.Module:
        for attr in attrs:
            if hasattr(obj, attr):
                return getattr(obj, attr)
        raise AttributeError(
            f"Cannot locate {label} on {type(obj).__name__}. Tried: {attrs}"
        )

    def _layer_params(self) -> list[nn.Parameter]:
        """Transformer layers + norm, excluding native embed_tokens."""
        return [
            p for name, p in self._body.named_parameters()
            if not name.startswith(self._embed_prefix)
        ]

    def freeze_body(self) -> None:
        for p in self._layer_params():
            p.requires_grad_(False)

    def unfreeze_body(self) -> None:
        for p in self._layer_params():
            p.requires_grad_(True)

    def body_params(self) -> list[nn.Parameter]:
        return [p for p in self._layer_params() if p.requires_grad]

    def embed_native(self, input_ids: Tensor) -> Tensor:
        return self._embed_fn(input_ids)

    def embed_alien(self, alien_ids: Tensor) -> Tensor:
        return self._alien_emb(alien_ids)

    def forward_hidden(
        self,
        inputs_embeds: Tensor,
        attention_mask: Tensor | None = None,
    ) -> Tensor:
        out = self._body(inputs_embeds=inputs_embeds, attention_mask=attention_mask)
        return out.last_hidden_state

    def alien_logits(self, hidden: Tensor) -> Tensor:
        return self._alien_head(hidden)

    @property
    def d_model(self) -> int:
        return self._lm.config.hidden_size

    @property
    def eos_token_id(self) -> int:
        eos = self._lm.config.eos_token_id
        return eos[0] if isinstance(eos, (list, tuple)) else eos

    def cipher_params(self) -> list[nn.Parameter]:
        return list(self._alien_emb.parameters()) + list(self._alien_head.parameters())

    @property
    def dtype(self) -> torch.dtype:
        return self._alien_emb.weight.dtype

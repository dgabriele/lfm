"""Decoder backend abstraction for SynthLM.

DecoderBackend(ABC) defines the interface any causal-LM backend must implement.
CausalDecoderBackend wraps any HuggingFace AutoModelForCausalLM; the LM body is
always fully frozen.  Only the alien embedding table and alien projection head
are trainable.

To add a new backend: subclass DecoderBackend and implement all abstract methods.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import torch
import torch.nn as nn
from torch import Tensor


class DecoderBackend(nn.Module, ABC):
    """Interface: frozen LM body with trainable alien cipher sub-vocabulary."""

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
        """(B, T, D) → (B, T, D) last hidden states from the frozen LM body."""

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
        """Phase 1 trainable parameters: alien_emb + alien_head only."""

    @property
    @abstractmethod
    def dtype(self) -> torch.dtype:
        """Compute dtype matching the LM body (e.g. bfloat16 for Qwen)."""


class CausalDecoderBackend(DecoderBackend):
    """Wraps any HuggingFace AutoModelForCausalLM. The LM body is always frozen.

    The alien embedding table (_alien_emb) and alien projection head (_alien_head)
    are separate from the native LM vocabulary and are trainable.  Native lm_head
    and embed_tokens are frozen.

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
        # Match alien modules to the model's dtype (e.g. bfloat16 for Qwen).
        dtype = next(lm.parameters()).dtype

        self._alien_emb = nn.Embedding(alien_vocab_size, d).to(dtype)
        self._alien_head = nn.Linear(d, alien_vocab_size, bias=False).to(dtype)

        self._body = self._find(lm, self._BODY_ATTRS, 'transformer body')
        self._embed_fn: nn.Embedding = self._find(self._body, self._EMBED_ATTRS, 'embed_tokens')

    @staticmethod
    def _find(obj: object, attrs: tuple[str, ...], label: str) -> nn.Module:
        for attr in attrs:
            if hasattr(obj, attr):
                return getattr(obj, attr)
        raise AttributeError(
            f"Cannot locate {label} on {type(obj).__name__}. Tried: {attrs}"
        )

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
        # Some models return a list of EOS ids; take the first.
        return eos[0] if isinstance(eos, (list, tuple)) else eos

    def cipher_params(self) -> list[nn.Parameter]:
        return list(self._alien_emb.parameters()) + list(self._alien_head.parameters())

    @property
    def dtype(self) -> torch.dtype:
        return self._alien_emb.weight.dtype

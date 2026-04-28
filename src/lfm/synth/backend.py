"""Decoder backend: LM body with trainable alien cipher sub-vocabulary.

DecoderBackend(ABC) defines the interface.
CausalDecoderBackend wraps any HuggingFace AutoModelForCausalLM.

Body freeze lifecycle:
  - __init__: body frozen (Phase 2 safe default).
  - unfreeze_body(): called by AlienLMTrainer before Phase 1.
  - freeze_body(): called by ConditioningTrainer before Phase 2.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import torch
import torch.nn as nn
from torch import Tensor


class DecoderBackend(nn.Module, ABC):
    """LM body (freeze-controlled) with trainable alien cipher sub-vocabulary."""

    @abstractmethod
    def embed_alien(self, alien_ids: Tensor) -> Tensor:
        """(B, T) int64 → (B, T, D)"""

    @abstractmethod
    def forward_hidden(
        self, inputs_embeds: Tensor, attention_mask: Tensor | None = None
    ) -> Tensor:
        """(B, T, D) → (B, T, D) last hidden states"""

    @abstractmethod
    def alien_logits(self, hidden: Tensor) -> Tensor:
        """(B, T, D) → (B, T, alien_vocab_size)"""

    @property
    @abstractmethod
    def d_model(self) -> int: ...

    @property
    @abstractmethod
    def dtype(self) -> torch.dtype: ...

    @abstractmethod
    def cipher_params(self) -> list[nn.Parameter]: ...

    @abstractmethod
    def body_params(self) -> list[nn.Parameter]: ...

    @abstractmethod
    def freeze_body(self) -> None: ...

    @abstractmethod
    def unfreeze_body(self) -> None: ...


class CausalDecoderBackend(DecoderBackend):
    """Wraps any HuggingFace AutoModelForCausalLM.

    alien_emb and alien_head are always trainable.
    Native embed_tokens and lm_head are always frozen.
    Transformer layers are frozen by default; call unfreeze_body() before Phase 1.

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
        d, dtype = lm.config.hidden_size, next(lm.parameters()).dtype
        self._alien_emb = nn.Embedding(alien_vocab_size, d).to(dtype)
        self._alien_head = nn.Linear(d, alien_vocab_size, bias=False).to(dtype)
        self._body = self._find(lm, self._BODY_ATTRS, 'transformer body')
        self._embed_prefix = next(a for a in self._EMBED_ATTRS if hasattr(self._body, a))

    @staticmethod
    def _find(obj: object, attrs: tuple[str, ...], label: str) -> nn.Module:
        for a in attrs:
            if hasattr(obj, a):
                return getattr(obj, a)
        raise AttributeError(f"Cannot locate {label} on {type(obj).__name__}. Tried: {attrs}")

    def _layer_params(self) -> list[nn.Parameter]:
        """Transformer layers excluding native embed_tokens."""
        return [p for n, p in self._body.named_parameters() if not n.startswith(self._embed_prefix)]

    def embed_alien(self, alien_ids: Tensor) -> Tensor:
        return self._alien_emb(alien_ids)

    def forward_hidden(
        self, inputs_embeds: Tensor, attention_mask: Tensor | None = None
    ) -> Tensor:
        return self._body(inputs_embeds=inputs_embeds, attention_mask=attention_mask).last_hidden_state

    def alien_logits(self, hidden: Tensor) -> Tensor:
        return self._alien_head(hidden)

    def cipher_params(self) -> list[nn.Parameter]:
        return list(self._alien_emb.parameters()) + list(self._alien_head.parameters())

    def body_params(self) -> list[nn.Parameter]:
        return [p for p in self._layer_params() if p.requires_grad]

    def freeze_body(self) -> None:
        for p in self._layer_params(): p.requires_grad_(False)

    def unfreeze_body(self) -> None:
        for p in self._layer_params(): p.requires_grad_(True)

    @property
    def d_model(self) -> int:
        return self._lm.config.hidden_size

    @property
    def dtype(self) -> torch.dtype:
        return self._alien_emb.weight.dtype

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

    @property
    def has_reference(self) -> bool:
        """True if a frozen reference body is available for MSE regularization."""
        return False

    def reference_hidden(
        self, inputs_embeds: Tensor, attention_mask: Tensor | None = None
    ) -> Tensor:
        """Hidden states from frozen reference body. Raises if not initialised."""
        raise RuntimeError(
            f"{type(self).__name__} was not constructed with a reference body."
        )


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

    def __init__(
        self,
        model_name: str,
        alien_vocab_size: int,
        with_reference_body: bool = False,
    ) -> None:
        super().__init__()
        import copy
        from transformers import AutoModelForCausalLM

        lm = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16)
        for p in lm.parameters():
            p.requires_grad_(False)

        self._lm = lm
        d, dtype = lm.config.hidden_size, next(lm.parameters()).dtype
        self._alien_emb = nn.Embedding(alien_vocab_size, d).to(dtype)
        self._alien_head = nn.Linear(d, alien_vocab_size, bias=False).to(dtype)
        self._body = self._find(lm, self._BODY_ATTRS, 'transformer body')
        self._embed_prefix = next(a for a in self._EMBED_ATTRS if hasattr(self._body, a))

        # Frozen reference body lives on CPU (float32) — zero VRAM cost.
        # Stored via object.__setattr__ to bypass nn.Module registration so that
        # model.to(device) never touches it.
        object.__setattr__(self, '_ref_body', None)
        if with_reference_body:
            ref = copy.deepcopy(self._body).to(dtype=torch.float32, device="cpu").eval()
            for p in ref.parameters():
                p.requires_grad_(False)
            object.__setattr__(self, '_ref_body', ref)

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
        self, inputs_embeds: Tensor, attention_mask: Tensor | None = None,
        past_key_values=None, use_cache: bool = False,
    ):
        out = self._body(
            inputs_embeds=inputs_embeds, attention_mask=attention_mask,
            past_key_values=past_key_values, use_cache=use_cache,
        )
        if use_cache:
            return out.last_hidden_state, out.past_key_values
        return out.last_hidden_state

    def alien_logits(self, hidden: Tensor) -> Tensor:
        return self._alien_head(hidden)

    @property
    def has_reference(self) -> bool:
        return self._ref_body is not None

    def move_reference_to(self, device: torch.device, dtype: torch.dtype) -> None:
        """Relocate the frozen reference body (e.g. CPU → CUDA after model.to(device))."""
        if self._ref_body is not None:
            object.__setattr__(self, '_ref_body', self._ref_body.to(device=device, dtype=dtype))

    @torch.no_grad()
    def reference_hidden(
        self, inputs_embeds: Tensor, attention_mask: Tensor | None = None
    ) -> Tensor:
        """Forward through frozen reference body; device-agnostic, returns on input device."""
        if self._ref_body is None:
            raise RuntimeError(
                "CausalDecoderBackend was not constructed with with_reference_body=True."
            )
        ref_device = next(iter(self._ref_body.parameters())).device
        ref_dtype = next(iter(self._ref_body.parameters())).dtype
        out = self._ref_body(
            inputs_embeds=inputs_embeds.detach().to(device=ref_device, dtype=ref_dtype),
            attention_mask=attention_mask,
        ).last_hidden_state
        return out.to(dtype=inputs_embeds.dtype, device=inputs_embeds.device)

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

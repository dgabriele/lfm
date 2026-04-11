"""Hidden-state extraction from a frozen causal LM.

Wraps a HuggingFace causal LM with an ergonomic batch-encoding API.
Supports multiple layer choices and pooling strategies so experiments
can compare last-token residuals, mean-pooled, or BOS-token
representations without touching the rest of the pipeline.

The returned embeddings are unit-normalized by default (cosine space),
matching both the downstream k-means clusterer and the
:class:`~lfm.embeddings.store.EmbeddingStore` convention used by the
dialogue game.
"""

from __future__ import annotations

import logging
from typing import Iterable, Iterator

import torch
import torch.nn.functional as F

from lfm.qwen_targets.config import ExtractorConfig

logger = logging.getLogger(__name__)


_DTYPE_MAP = {
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
    "float32": torch.float32,
}


class HiddenStateExtractor:
    """Frozen causal LM + configurable pooling.

    Instantiation downloads (or loads from cache) the configured model
    and puts it in eval mode with all parameters frozen.  Call
    :meth:`encode` for ad-hoc batches or :meth:`encode_stream` to
    iterate over a generator of texts.

    Args:
        config: Extractor configuration.
        device: Torch device the model lives on.
    """

    def __init__(
        self,
        config: ExtractorConfig,
        device: torch.device | str = "cuda",
    ) -> None:
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.config = config
        self.device = torch.device(device)
        self.dtype = _DTYPE_MAP[config.dtype]

        # Enable TF32 tensor cores for any float32 matmul paths (e.g.
        # inductor-generated reductions when compiled).  Harmless for
        # extraction — we never use float32 model math, only post-hoc
        # normalize of pooled vectors.
        torch.set_float32_matmul_precision("high")

        logger.info(
            "Loading causal LM %s (%s) for hidden-state extraction",
            config.model_name, config.dtype,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(config.model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        load_kwargs: dict = {"torch_dtype": self.dtype}
        if config.attn_implementation is not None:
            load_kwargs["attn_implementation"] = config.attn_implementation
        self.model = AutoModelForCausalLM.from_pretrained(
            config.model_name, **load_kwargs,
        ).to(self.device)
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad_(False)

        # Drop the LM head — we never use it and it's pointlessly large.
        # Qwen stores the final classifier as ``lm_head``; replacing with
        # Identity frees ~500 MB on Qwen 2.5 0.5B and more on larger variants.
        if hasattr(self.model, "lm_head"):
            self.model.lm_head = torch.nn.Identity()

        # Register a forward hook on the target layer so we can drop
        # ``output_hidden_states=True`` (which materializes ALL layers'
        # hidden states and costs hundreds of MB per batch).  With the
        # hook, only the requested layer is retained.
        self._captured: torch.Tensor | None = None

        def _hook(_mod, _inp, out) -> None:
            # Transformer decoder layers return a tuple; hidden state is [0].
            if isinstance(out, tuple):
                self._captured = out[0]
            else:
                self._captured = out

        target_layer = self._resolve_target_layer(config.layer)
        self._hook_handle = target_layer.register_forward_hook(_hook)

        # Optional torch.compile — amortize Python dispatch overhead.
        # Use "reduce-overhead" mode for inference: enables CUDA graphs
        # capture after a few warm-up calls and gives big speedups on
        # fixed-shape loops.  Requires shape stability, which we get
        # by running at a constant batch_size.
        if config.compile:
            logger.info("torch.compile(mode='reduce-overhead') on extractor forward")
            self.model = torch.compile(self.model, mode="reduce-overhead")

        # Probe hidden size from the loaded model
        self.hidden_size = int(self.model.config.hidden_size)
        logger.info(
            "Extractor ready: hidden_size=%d layer=%d pooling=%s compile=%s attn=%s",
            self.hidden_size, config.layer, config.pooling,
            config.compile, config.attn_implementation,
        )

    def _resolve_target_layer(self, layer_idx: int) -> torch.nn.Module:
        """Return the decoder layer module at ``layer_idx`` (supports negative).

        Works for Qwen/Llama-style architectures that expose
        ``model.model.layers`` as an ``nn.ModuleList`` of decoder blocks.
        """
        # ``AutoModelForCausalLM`` wraps the base model at ``.model``.
        base = getattr(self.model, "model", self.model)
        layers = getattr(base, "layers", None)
        if layers is None:
            raise RuntimeError(
                f"Cannot find decoder layers on {type(self.model).__name__}; "
                "this extractor currently supports Qwen/Llama-style models.",
            )
        n_layers = len(layers)
        idx = layer_idx if layer_idx >= 0 else n_layers + layer_idx
        if idx < 0 or idx >= n_layers:
            raise ValueError(
                f"layer={layer_idx} out of range for model with {n_layers} layers",
            )
        return layers[idx]

    def _pool(
        self,
        last_hidden: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Reduce ``(B, T, D)`` to ``(B, D)`` per the configured pooling."""
        pooling = self.config.pooling
        if pooling == "last_token":
            lengths = attention_mask.sum(dim=1) - 1
            lengths = lengths.clamp(min=0)
            batch_idx = torch.arange(last_hidden.size(0), device=last_hidden.device)
            pooled = last_hidden[batch_idx, lengths]
        elif pooling == "mean":
            mask = attention_mask.unsqueeze(-1).float()
            summed = (last_hidden * mask).sum(dim=1)
            counts = mask.sum(dim=1).clamp(min=1)
            pooled = summed / counts
        elif pooling == "bos":
            pooled = last_hidden[:, 0]
        else:
            raise ValueError(f"Unknown pooling mode: {pooling!r}")
        return pooled

    @torch.inference_mode()
    def encode(self, texts: list[str]) -> torch.Tensor:
        """Encode a batch of texts to normalized pooled hidden states.

        Returns a ``(len(texts), hidden_size)`` float32 tensor on CPU,
        unit-normalized.  Uses a forward hook on the configured layer
        so we never materialize the full per-layer hidden-state tuple.
        """
        # With torch.compile in reduce-overhead mode we need fixed
        # shapes for CUDA graph capture.  Pad every batch to max_len
        # (and accept the wasted compute on short batches) so that the
        # graph is reused on every call.  Without compile, pad-to-
        # longest-in-batch to save work.
        padding = "max_length" if self.config.compile else True
        enc = self.tokenizer(
            texts,
            padding=padding,
            truncation=True,
            max_length=self.config.max_len,
            return_tensors="pt",
        ).to(self.device)

        self._captured = None
        # The LM head is an Identity now, so logits == final hidden state;
        # we still rely on the hook to grab the correct (configurable) layer.
        self.model(**enc, use_cache=False)
        last_hidden = self._captured
        if last_hidden is None:
            raise RuntimeError(
                "Forward hook did not fire — layer capture failed.",
            )
        pooled = self._pool(last_hidden, enc["attention_mask"])
        pooled = F.normalize(pooled.float(), dim=-1).cpu()
        self._captured = None  # release reference
        return pooled

    @torch.no_grad()
    def encode_stream(
        self,
        texts: Iterable[str],
        progress_every: int = 2000,
    ) -> Iterator[torch.Tensor]:
        """Yield embeddings batch-by-batch from a text iterator.

        The iterator is consumed lazily in chunks of ``config.batch_size``
        so the caller never needs to materialize the full corpus in
        memory.

        Yields:
            ``(batch_size, hidden_size)`` float32 tensors, unit-normalized.
        """
        batch: list[str] = []
        processed = 0
        for text in texts:
            batch.append(text)
            if len(batch) >= self.config.batch_size:
                yield self.encode(batch)
                processed += len(batch)
                if processed % progress_every == 0:
                    logger.info("  extractor: %d texts processed", processed)
                batch = []
        if batch:
            yield self.encode(batch)
            processed += len(batch)
            logger.info("  extractor: %d texts processed (final)", processed)

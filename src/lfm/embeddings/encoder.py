"""Text encoders for producing dense passage embeddings.

Provides an abstract ``TextEncoder`` base class and a concrete
``SentenceTransformersEncoder`` implementation. These are NOT ``nn.Module``
subclasses -- they wrap frozen, inference-only models and produce numpy arrays.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import numpy as np

from lfm._registry import register
from lfm.embeddings.config import TextEncoderConfig
from lfm.utils.logging import get_logger

logger = get_logger(__name__)


class TextEncoder(ABC):
    """Abstract base for swappable LLM text encoders.

    Subclasses must implement :meth:`encode`, :meth:`get_tokenizer`, and the
    :attr:`embedding_dim` property.  The :meth:`encode_batched` convenience
    method is provided for processing large passage lists in micro-batches.
    """

    def __init__(self, config: TextEncoderConfig) -> None:
        self.config = config

    @abstractmethod
    def encode(self, passages: list[str]) -> np.ndarray:
        """Encode passages into dense vectors.

        Args:
            passages: List of text passages to encode.

        Returns:
            NumPy array of shape ``(len(passages), embedding_dim)``.
        """

    @abstractmethod
    def get_tokenizer(self) -> Any:
        """Return the tokenizer associated with this encoder."""

    @property
    @abstractmethod
    def embedding_dim(self) -> int:
        """Dimensionality of the output embedding vectors."""

    def encode_batched(self, passages: list[str], show_progress: bool = True) -> np.ndarray:
        """Encode passages in micro-batches and concatenate the results.

        Args:
            passages: Full list of passages to encode.
            show_progress: Whether to log encoding progress.

        Returns:
            NumPy array of shape ``(len(passages), embedding_dim)``.
        """
        results: list[np.ndarray] = []
        bs = self.config.batch_size
        total = len(passages)
        for i in range(0, total, bs):
            chunk = passages[i : i + bs]
            results.append(self.encode(chunk))
            if show_progress and i % (bs * 10) == 0:
                logger.info("Encoded %d / %d passages", i, total)
        if show_progress:
            logger.info("Encoded %d / %d passages (done)", total, total)
        return np.concatenate(results, axis=0)


@register("text_encoder", "sentence_transformers")
class SentenceTransformersEncoder(TextEncoder):
    """Encoder backed by the ``sentence-transformers`` library.

    Works with any model supported by ``SentenceTransformer``, including
    stella, Qwen3-Embedding, GTE, and E5 variants.  The library is imported
    lazily to avoid a hard dependency for users who don't need this encoder.
    """

    def __init__(self, config: TextEncoderConfig) -> None:
        super().__init__(config)
        # Lazy import so sentence-transformers is only required when this
        # encoder is actually instantiated.
        from sentence_transformers import SentenceTransformer

        logger.info(
            "Loading SentenceTransformer model: %s (device=%s, dtype=%s)",
            config.model_id,
            config.device,
            config.dtype,
        )
        self._model = SentenceTransformer(
            config.model_id,
            device=config.device,
            trust_remote_code=config.trust_remote_code,
        )
        if config.dtype == "float16":
            self._model = self._model.half()

    def encode(self, passages: list[str]) -> np.ndarray:
        """Encode a single micro-batch of passages.

        Args:
            passages: Passages to encode (length should not exceed
                ``config.batch_size``).

        Returns:
            L2-normalized embeddings as a NumPy array of shape
            ``(len(passages), embedding_dim)``.
        """
        kwargs: dict[str, Any] = {
            "batch_size": len(passages),
            "convert_to_numpy": True,
            "normalize_embeddings": True,
            "show_progress_bar": False,
        }
        if self.config.prompt_name is not None:
            kwargs["prompt_name"] = self.config.prompt_name
        return self._model.encode(passages, **kwargs)

    def get_tokenizer(self) -> Any:
        """Return the underlying HuggingFace tokenizer."""
        return self._model.tokenizer

    @property
    def embedding_dim(self) -> int:
        """Dimensionality of the sentence embedding vectors."""
        dim = self._model.get_sentence_embedding_dimension()
        if dim is None:
            raise RuntimeError(
                "Model did not report an embedding dimension. "
                "Set TextEncoderConfig.embedding_dim explicitly."
            )
        return int(dim)

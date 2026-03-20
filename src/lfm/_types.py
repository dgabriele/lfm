"""Semantic tensor type aliases for the LFM framework.

These aliases exist purely for documentation and readability. They provide
no runtime enforcement but make function signatures self-documenting by
conveying the intended shape and dtype of each tensor argument.
"""

from __future__ import annotations

from typing import TypeAlias

from torch import Tensor

AgentState: TypeAlias = Tensor
"""Agent internal state — shape ``(batch, dim)``."""

TokenIds: TypeAlias = Tensor
"""Integer token indices — shape ``(batch, seq_len)``, dtype ``int64``."""

TokenEmbeddings: TypeAlias = Tensor
"""Dense token embeddings — shape ``(batch, seq_len, dim)``."""

Logits: TypeAlias = Tensor
"""Un-normalized log-probabilities — shape ``(batch, seq_len, vocab_size)``."""

Mask: TypeAlias = Tensor
"""Boolean attention/padding mask — shape ``(batch, seq_len)``, dtype ``bool``."""

TreeLogProbs: TypeAlias = Tensor
"""Log-probability of a parse tree — shape ``(batch,)``."""

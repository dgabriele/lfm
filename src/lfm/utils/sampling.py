"""Sampling utilities for the LFM framework.

Provides differentiable sampling primitives — Gumbel-Softmax, straight-through
estimators, and categorical sampling — used throughout the quantization and
discrete-structure modules.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F  # noqa: N812
from torch import Tensor


def gumbel_softmax(
    logits: Tensor,
    tau: float = 1.0,
    hard: bool = False,
    dim: int = -1,
) -> Tensor:
    """Sample from the Gumbel-Softmax distribution.

    Draws a differentiable sample from a categorical distribution parameterised
    by ``logits``.  When ``hard=True`` the forward pass is one-hot but
    gradients flow through the soft sample (straight-through estimator).

    Args:
        logits: Un-normalized log-probabilities of shape ``(..., num_classes)``.
        tau: Temperature parameter.  Lower values produce sharper distributions.
        hard: If ``True``, return one-hot in the forward pass with
            straight-through gradients.
        dim: Dimension along which softmax is computed.

    Returns:
        Tensor of same shape as ``logits``.
    """
    # Sample Gumbel noise
    gumbels = -torch.empty_like(logits).exponential_().log()
    gumbels = (logits + gumbels) / tau
    soft = F.softmax(gumbels, dim=dim)

    if hard:
        index = soft.argmax(dim=dim, keepdim=True)
        hard_sample = torch.zeros_like(soft).scatter_(dim, index, 1.0)
        return straight_through(soft, hard_sample)

    return soft


def straight_through(soft: Tensor, hard: Tensor) -> Tensor:
    """Straight-through estimator: ``hard`` in the forward pass, ``soft`` gradient in backward.

    This is the core trick behind VQ-VAE and Gumbel-Softmax with ``hard=True``.
    The returned tensor equals ``hard`` numerically, but its gradient is that
    of ``soft``.

    Args:
        soft: Differentiable tensor whose gradient we want to keep.
        hard: Non-differentiable (or discretised) tensor for the forward pass.

    Returns:
        Tensor with value of ``hard`` and gradient of ``soft``.
    """
    return hard - soft.detach() + soft


def sample_categorical(logits: Tensor) -> Tensor:
    """Sample from a categorical distribution with straight-through gradients.

    Draws a hard (one-hot) sample from the categorical distribution defined by
    ``logits`` along the last dimension, using the straight-through estimator
    so gradients flow back through the soft probabilities.

    Args:
        logits: Un-normalized log-probabilities of shape ``(..., num_classes)``.

    Returns:
        One-hot tensor of the same shape as ``logits``, with straight-through
        gradients.
    """
    probs = F.softmax(logits, dim=-1)
    indices = torch.multinomial(probs.reshape(-1, probs.size(-1)), num_samples=1)
    indices = indices.reshape(*probs.shape[:-1], 1)
    hard = torch.zeros_like(probs).scatter_(-1, indices, 1.0)
    return straight_through(probs, hard)

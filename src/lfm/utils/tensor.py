"""Tensor manipulation utilities for the LFM framework.

Small, self-contained helpers for padding, masking, and aggregation that are
used throughout the pipeline.
"""

from __future__ import annotations

import torch
from torch import Tensor


def create_padding_mask(lengths: Tensor, max_len: int) -> Tensor:
    """Create a boolean padding mask from sequence lengths.

    Positions *within* the valid length are ``True``; padding positions are
    ``False``.

    Args:
        lengths: Integer tensor of shape ``(batch,)`` giving each sequence's
            valid length.
        max_len: Maximum sequence length (width of the returned mask).

    Returns:
        Boolean tensor of shape ``(batch, max_len)``.
    """
    # arange broadcasts against lengths to produce the mask
    return torch.arange(max_len, device=lengths.device).unsqueeze(0) < lengths.unsqueeze(1)


def pad_sequence(
    sequences: list[Tensor],
    pad_value: float = 0.0,
) -> tuple[Tensor, Tensor]:
    """Pad a list of variable-length tensors into a single batch tensor.

    Each element of *sequences* should have shape ``(length, *)`` where ``*``
    denotes any number of trailing dimensions (all sequences must agree on
    trailing dims).

    Args:
        sequences: List of tensors to pad.
        pad_value: Value used for padding positions.

    Returns:
        A tuple ``(padded, lengths)`` where:
        - ``padded`` has shape ``(batch, max_len, *)``.
        - ``lengths`` is an ``int64`` tensor of shape ``(batch,)``.
    """
    lengths = torch.tensor(
        [s.size(0) for s in sequences],
        dtype=torch.int64,
        device=sequences[0].device,
    )
    max_len = int(lengths.max().item())
    trailing_shape = sequences[0].shape[1:]
    padded = torch.full(
        (len(sequences), max_len, *trailing_shape),
        fill_value=pad_value,
        dtype=sequences[0].dtype,
        device=sequences[0].device,
    )
    for i, seq in enumerate(sequences):
        padded[i, : seq.size(0)] = seq
    return padded, lengths


def masked_mean(x: Tensor, mask: Tensor, dim: int = -1) -> Tensor:
    """Compute the mean of ``x`` over ``dim``, considering only masked positions.

    Positions where ``mask`` is ``False`` are excluded from the mean.  If *all*
    positions along ``dim`` are masked out the result is zero (avoids NaN).

    Args:
        x: Input tensor.
        mask: Boolean tensor broadcastable to ``x``.  ``True`` positions are
            included in the mean.
        dim: Dimension along which to compute the mean.

    Returns:
        Tensor with ``dim`` reduced.
    """
    mask_float = mask.to(dtype=x.dtype)
    # Expand mask to match x's shape if needed (e.g. x is (B, T, D), mask is (B, T))
    while mask_float.dim() < x.dim():
        mask_float = mask_float.unsqueeze(-1)
    masked = x * mask_float
    count = mask_float.sum(dim=dim).clamp(min=1.0)
    return masked.sum(dim=dim) / count

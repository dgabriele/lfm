"""Collation utilities for variable-length sequences.

Provides collate functions for use with PyTorch DataLoaders that handle
variable-length sequences by padding to the maximum length within each batch.
"""

from __future__ import annotations

import torch
from torch import Tensor


def variable_length_collate(
    batch: list[tuple[Tensor, int]],
) -> tuple[Tensor, Tensor]:
    """Collate variable-length ``(tokens, length)`` tuples into a batch.

    Pads token sequences to the maximum length within the batch and stacks
    them into a single tensor.

    Args:
        batch: List of ``(token_ids, length)`` tuples from the dataset.

    Returns:
        Tuple of ``(batch_tokens, batch_lengths)`` where ``batch_tokens``
        is ``(batch_size, max_len)`` and ``batch_lengths`` is ``(batch_size,)``.
    """
    tokens_list, lengths = zip(*batch)
    batch_tokens = torch.stack(tokens_list, dim=0)
    batch_lengths = torch.tensor(lengths, dtype=torch.long)
    return batch_tokens, batch_lengths


def _pad_to_length(tensor: Tensor, target_length: int, pad_value: float = 0.0) -> Tensor:
    """Pad a tensor along dimension 0 to the target length.

    Args:
        tensor: Input tensor to pad.
        target_length: Desired length along dimension 0.
        pad_value: Value to use for padding.

    Returns:
        Padded tensor of shape ``(target_length, ...)``.
    """
    if tensor.size(0) >= target_length:
        return tensor[:target_length]
    pad_size = target_length - tensor.size(0)
    pad_shape = (pad_size, *tensor.shape[1:])
    padding = torch.full(pad_shape, pad_value, dtype=tensor.dtype, device=tensor.device)
    return torch.cat([tensor, padding], dim=0)

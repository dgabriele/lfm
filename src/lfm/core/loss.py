"""Abstract loss classes and composite loss aggregation for LFM.

Structural losses in LFM reward well-formedness and compositional consistency
rather than semantic correctness relative to human language.  Every concrete
loss inherits from ``LFMLoss`` and is registered via ``@register("loss", ...)``.

``CompositeLoss`` aggregates multiple named ``LFMLoss`` instances into a single
weighted sum, which is the standard loss interface consumed by the training loop.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import torch
from torch import Tensor, nn

from lfm._registry import create, list_registered


class LFMLoss(ABC, nn.Module):
    """Abstract base class for all structural losses in LFM.

    Each loss receives the full pipeline outputs dict (and optional targets)
    and returns a scalar loss tensor.

    Args:
        config: Configuration object for this loss.  May be ``None`` for
            losses that require no configuration beyond their weight.
        weight: Multiplicative weight applied when this loss is combined
            with others inside a ``CompositeLoss``.
    """

    def __init__(self, config: object = None, weight: float = 1.0) -> None:
        super().__init__()
        self.config = config
        self.weight = weight

    @abstractmethod
    def forward(
        self,
        outputs: dict[str, Tensor],
        targets: dict[str, Tensor] | None = None,
    ) -> Tensor:
        """Compute the loss value.

        Args:
            outputs: Combined output dictionary from the ``LanguageFaculty``
                forward pass.
            targets: Optional ground-truth tensors (used by supervised or
                semi-supervised losses).

        Returns:
            A scalar loss tensor.
        """


class CompositeLoss(nn.Module):
    """Weighted sum of multiple ``LFMLoss`` instances.

    This is the standard loss interface consumed by the LFM training loop.
    Individual loss values are tracked for logging and diagnostics.

    Args:
        losses: Dictionary mapping human-readable loss names to ``LFMLoss``
            instances.  The ``weight`` attribute on each instance controls its
            contribution to the total.
    """

    def __init__(self, losses: dict[str, LFMLoss]) -> None:
        super().__init__()
        self.losses = nn.ModuleDict(losses)

    def forward(
        self,
        outputs: dict[str, Tensor],
        targets: dict[str, Tensor] | None = None,
    ) -> tuple[Tensor, dict[str, Tensor]]:
        """Compute the weighted sum of all constituent losses.

        Args:
            outputs: Combined pipeline output dictionary.
            targets: Optional ground-truth tensors.

        Returns:
            A tuple of ``(total_loss, per_loss_dict)`` where ``per_loss_dict``
            maps each loss name to its *weighted* scalar value.
        """
        per_loss: dict[str, Tensor] = {}
        total = torch.tensor(0.0, device=_infer_device(outputs))

        for name, loss_fn in self.losses.items():
            assert isinstance(loss_fn, LFMLoss)
            value = loss_fn(outputs, targets) * loss_fn.weight
            per_loss[name] = value
            total = total + value

        return total, per_loss

    @classmethod
    def from_config(cls, loss_weights: dict[str, float]) -> CompositeLoss:
        """Build a ``CompositeLoss`` from a ``{name: weight}`` mapping.

        Each *name* must correspond to a class registered under the ``"loss"``
        category in the global registry.  The class is instantiated with
        ``config=None`` and the given weight.

        Args:
            loss_weights: Mapping of registered loss names to their weights.

        Returns:
            A ``CompositeLoss`` combining all specified losses.

        Raises:
            KeyError: If a loss name is not found in the registry.
        """
        available = list_registered("loss")
        losses: dict[str, LFMLoss] = {}
        for name, weight in loss_weights.items():
            if name not in available:
                raise KeyError(f"Loss {name!r} is not registered. Available: {available}")
            loss_instance = create("loss", name, config=None, weight=weight)
            assert isinstance(loss_instance, LFMLoss)
            losses[name] = loss_instance
        return cls(losses)


def _infer_device(tensors: dict[str, Tensor]) -> torch.device:
    """Return the device of the first tensor in the dict, falling back to CPU."""
    for v in tensors.values():
        if isinstance(v, Tensor):
            return v.device
    return torch.device("cpu")

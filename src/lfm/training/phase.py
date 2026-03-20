"""Base training-phase abstraction for the LFM training loop.

A training run is split into sequential phases, each with its own loss
weighting, frozen-module set, and (optionally) data source.  Concrete phases
inherit from ``TrainingPhase`` and implement ``step()``.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

from lfm.core.loss import CompositeLoss

if TYPE_CHECKING:
    from torch import Tensor, nn

    from lfm.training.config import PhaseConfig


class TrainingPhase(ABC):
    """Base class for training phases.

    Each phase wraps a ``LanguageFaculty`` model and applies phase-specific
    behaviour: freezing/unfreezing sub-modules, constructing the composite
    loss, and defining the training step.

    Args:
        config: Phase configuration (loss weights, frozen modules, etc.).
        faculty: The ``LanguageFaculty`` model being trained.
    """

    def __init__(self, config: PhaseConfig, faculty: nn.Module) -> None:
        self.config = config
        self.faculty = faculty

    def build_loss(self) -> CompositeLoss:
        """Build a composite loss from the phase's loss configuration.

        Uses ``CompositeLoss.from_config`` which looks up each loss name
        in the global registry.

        Returns:
            A ``CompositeLoss`` combining all losses specified in
            ``self.config.losses``.
        """
        return CompositeLoss.from_config(self.config.losses)

    def on_phase_start(self) -> None:
        """Freeze sub-modules listed in ``config.modules_frozen``."""
        for module_name in self.config.modules_frozen:
            module = getattr(self.faculty, module_name, None)
            if module is not None:
                for param in module.parameters():
                    param.requires_grad_(False)

    def on_phase_end(self) -> None:
        """Unfreeze all sub-modules that were frozen during this phase."""
        for module_name in self.config.modules_frozen:
            module = getattr(self.faculty, module_name, None)
            if module is not None:
                for param in module.parameters():
                    param.requires_grad_(True)

    @abstractmethod
    def step(self, batch: dict[str, Tensor]) -> tuple[dict[str, Tensor], dict[str, Tensor]]:
        """Execute one training step.

        Args:
            batch: Input batch dictionary.  Expected to contain at least
                ``"agent_state"`` and optionally ``"mask"``.

        Returns:
            A tuple ``(outputs, losses)`` where *outputs* is the model output
            dict and *losses* maps loss names to scalar tensors (including a
            ``"total"`` key).
        """
        ...

    def default_step(
        self, batch: dict[str, Tensor], loss_fn: CompositeLoss
    ) -> tuple[dict[str, Tensor], dict[str, Tensor]]:
        """Default step implementation: forward pass followed by loss computation.

        Runs the faculty model on the batch, computes the composite loss,
        and appends any auxiliary losses from the faculty.

        Args:
            batch: Input batch dictionary with ``"agent_state"`` and optional
                ``"mask"`` keys.
            loss_fn: Composite loss function to evaluate on model outputs.

        Returns:
            A tuple ``(outputs, losses)`` where *losses* includes per-component
            values plus a ``"total"`` key.
        """
        outputs = self.faculty(batch["agent_state"], mask=batch.get("mask"))
        total_loss, loss_dict = loss_fn(outputs)

        # Incorporate auxiliary losses from the faculty (e.g. commitment loss).
        extra = self.faculty.extra_losses()
        for k, v in extra.items():
            loss_dict[f"extra.{k}"] = v
            total_loss = total_loss + v

        loss_dict["total"] = total_loss
        return outputs, loss_dict

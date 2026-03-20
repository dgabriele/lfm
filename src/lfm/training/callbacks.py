"""Training callback system for the LFM training loop.

Provides a ``Callback`` protocol and concrete implementations for logging,
checkpointing, and metric evaluation.
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING, Protocol, runtime_checkable

import torch

from lfm.utils.logging import get_logger

if TYPE_CHECKING:
    from torch import Tensor, nn

    from lfm.training.config import PhaseConfig

logger = get_logger(__name__)


@runtime_checkable
class Callback(Protocol):
    """Protocol defining the callback interface for training events."""

    def on_train_start(self) -> None:
        """Called once at the beginning of the entire training run."""
        ...

    def on_train_end(self) -> None:
        """Called once at the end of the entire training run."""
        ...

    def on_phase_start(self, phase_name: str, phase_config: PhaseConfig) -> None:
        """Called when a training phase begins.

        Args:
            phase_name: Name of the phase.
            phase_config: Configuration for this phase.
        """
        ...

    def on_phase_end(self, phase_name: str) -> None:
        """Called when a training phase ends.

        Args:
            phase_name: Name of the phase that just completed.
        """
        ...

    def on_step_start(self, global_step: int) -> None:
        """Called before each training step.

        Args:
            global_step: The global step counter (across all phases).
        """
        ...

    def on_step_end(
        self,
        global_step: int,
        outputs: dict[str, Tensor],
        losses: dict[str, Tensor],
    ) -> None:
        """Called after each training step.

        Args:
            global_step: The global step counter.
            outputs: Model outputs from the step.
            losses: Dictionary of loss values from the step.
        """
        ...


class LoggingCallback:
    """Logs training metrics at configurable intervals.

    Logs the total loss, individual loss components, and the current learning
    rate at every ``log_every`` steps.

    Args:
        log_every: Logging interval in steps.
        optimizer: The optimizer, used to read the current learning rate.
    """

    def __init__(self, log_every: int, optimizer: torch.optim.Optimizer) -> None:
        self.log_every = log_every
        self.optimizer = optimizer

    def on_train_start(self) -> None:
        logger.info("Training started")

    def on_train_end(self) -> None:
        logger.info("Training finished")

    def on_phase_start(self, phase_name: str, phase_config: PhaseConfig) -> None:
        logger.info(
            "Phase '%s' started — %d steps, losses: %s",
            phase_name,
            phase_config.steps,
            phase_config.losses,
        )

    def on_phase_end(self, phase_name: str) -> None:
        logger.info("Phase '%s' completed", phase_name)

    def on_step_start(self, global_step: int) -> None:  # noqa: ARG002
        pass

    def on_step_end(
        self,
        global_step: int,
        outputs: dict[str, Tensor],  # noqa: ARG002
        losses: dict[str, Tensor],
    ) -> None:
        if global_step % self.log_every != 0:
            return

        current_lr = self.optimizer.param_groups[0]["lr"]
        total_loss = losses.get("total")
        total_val = total_loss.item() if total_loss is not None else float("nan")

        parts = []
        for name, value in sorted(losses.items()):
            if name != "total":
                parts.append(f"{name}={value.item():.4f}")
        loss_str = ", ".join(parts)

        logger.info(
            "step %d | loss=%.4f | lr=%.2e | %s",
            global_step,
            total_val,
            current_lr,
            loss_str,
        )


class CheckpointCallback:
    """Saves model checkpoints at configurable intervals.

    Persists model and optimizer state dictionaries to disk every
    ``checkpoint_every`` steps.

    Args:
        checkpoint_dir: Directory in which to save checkpoint files.
        checkpoint_every: Checkpointing interval in steps.
        model: The model whose ``state_dict`` will be saved.
        optimizer: The optimizer whose ``state_dict`` will be saved.
    """

    def __init__(
        self,
        checkpoint_dir: str,
        checkpoint_every: int,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
    ) -> None:
        self.checkpoint_dir = checkpoint_dir
        self.checkpoint_every = checkpoint_every
        self.model = model
        self.optimizer = optimizer

    def on_train_start(self) -> None:
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        logger.info("Checkpoints will be saved to '%s'", self.checkpoint_dir)

    def on_train_end(self) -> None:
        pass

    def on_phase_start(
        self,
        phase_name: str,  # noqa: ARG002
        phase_config: PhaseConfig,  # noqa: ARG002
    ) -> None:
        pass

    def on_phase_end(self, phase_name: str) -> None:  # noqa: ARG002
        pass

    def on_step_start(self, global_step: int) -> None:  # noqa: ARG002
        pass

    def on_step_end(
        self,
        global_step: int,
        outputs: dict[str, Tensor],  # noqa: ARG002
        losses: dict[str, Tensor],  # noqa: ARG002
    ) -> None:
        if global_step % self.checkpoint_every != 0:
            return

        path = os.path.join(self.checkpoint_dir, f"checkpoint_{global_step}.pt")
        torch.save(
            {
                "global_step": global_step,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
            },
            path,
        )
        logger.info("Saved checkpoint at step %d to '%s'", global_step, path)


class MetricsCallback:
    """Computes and logs evaluation metrics during training.

    Currently a minimal implementation that accumulates step losses and
    computes running averages.

    Args:
        metrics: List of metric names to track.
    """

    def __init__(self, metrics: list[str] | None = None) -> None:
        self.metric_names = metrics or []
        self._step_losses: list[float] = []

    def on_train_start(self) -> None:
        self._step_losses.clear()

    def on_train_end(self) -> None:
        if self._step_losses:
            avg = sum(self._step_losses) / len(self._step_losses)
            logger.info("Training average loss: %.4f", avg)

    def on_phase_start(
        self,
        phase_name: str,  # noqa: ARG002
        phase_config: PhaseConfig,  # noqa: ARG002
    ) -> None:
        self._step_losses.clear()

    def on_phase_end(self, phase_name: str) -> None:
        if self._step_losses:
            avg = sum(self._step_losses) / len(self._step_losses)
            logger.info(
                "Phase '%s' average loss: %.4f over %d steps",
                phase_name,
                avg,
                len(self._step_losses),
            )

    def on_step_start(self, global_step: int) -> None:  # noqa: ARG002
        pass

    def on_step_end(
        self,
        global_step: int,  # noqa: ARG002
        outputs: dict[str, Tensor],  # noqa: ARG002
        losses: dict[str, Tensor],
    ) -> None:
        total = losses.get("total")
        if total is not None:
            self._step_losses.append(total.item())

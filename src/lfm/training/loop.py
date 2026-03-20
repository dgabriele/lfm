"""Main training loop for the LFM framework.

``TrainingLoop`` orchestrates multi-phase training: it iterates through an
ordered list of training phases, manages the optimizer and learning-rate
schedule, fires callbacks, and handles checkpointing.
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING, Any

import torch

from lfm._registry import create
from lfm.training.callbacks import (
    Callback,
    CheckpointCallback,
    LoggingCallback,
)
from lfm.training.config import TrainingConfig
from lfm.training.phase import TrainingPhase
from lfm.utils.logging import get_logger

if TYPE_CHECKING:
    from torch import nn

logger = get_logger(__name__)


class TrainingLoop:
    """Orchestrates multi-phase training for a ``LanguageFaculty`` model.

    The loop:
    1. Builds an optimizer and (optionally) an LR scheduler from config.
    2. Iterates through each configured training phase in order.
    3. For each phase: freezes/unfreezes modules, builds the phase loss,
       runs training steps, and fires callback hooks.

    Args:
        faculty: The ``LanguageFaculty`` model to train.
        training_config: Top-level training configuration.
        device: Device string (e.g. ``"cuda"``, ``"cpu"``).
    """

    def __init__(
        self,
        faculty: nn.Module,
        training_config: TrainingConfig,
        device: str = "cuda",
    ) -> None:
        self.faculty = faculty
        self.config = training_config
        self.device = torch.device(device)

        self.faculty.to(self.device)

        self.optimizer = self._build_optimizer()
        self.scheduler = self._build_scheduler()

        self._callbacks: list[Callback] = []
        self._phases: list[TrainingPhase] = self._build_phases()
        self._global_step = 0

        # Register default callbacks.
        self.add_callback(
            LoggingCallback(
                log_every=self.config.log_every,
                optimizer=self.optimizer,
            )
        )
        self.add_callback(
            CheckpointCallback(
                checkpoint_dir=self.config.checkpoint_dir,
                checkpoint_every=self.config.checkpoint_every,
                model=self.faculty,
                optimizer=self.optimizer,
            )
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self, data_iter: Any = None) -> None:
        """Execute all training phases in sequence.

        Args:
            data_iter: An iterable (or iterator) that yields ``dict[str, Tensor]``
                batches.  If ``None``, the caller must ensure that phases can
                generate their own data.  In practice this is supplied by the
                experiment runner.
        """
        self._fire("on_train_start")

        for phase in self._phases:
            phase_cfg = phase.config
            logger.info("Starting phase '%s' (%d steps)", phase_cfg.name, phase_cfg.steps)
            self._fire("on_phase_start", phase_cfg.name, phase_cfg)
            phase.on_phase_start()

            # Adjust LR for this phase.
            self._set_lr_scale(phase_cfg.lr_scale)

            if data_iter is None:
                logger.warning(
                    "No data iterator provided — skipping phase '%s'",
                    phase_cfg.name,
                )
                phase.on_phase_end()
                self._fire("on_phase_end", phase_cfg.name)
                continue

            for _ in range(phase_cfg.steps):
                if self._global_step >= self.config.total_steps:
                    logger.info(
                        "Reached total_steps=%d, stopping early.",
                        self.config.total_steps,
                    )
                    phase.on_phase_end()
                    self._fire("on_phase_end", phase_cfg.name)
                    self._fire("on_train_end")
                    return

                self._fire("on_step_start", self._global_step)

                batch = next(data_iter)
                batch = self._to_device(batch)

                # Forward + loss.
                outputs, losses = phase.step(batch)
                total_loss = losses["total"]

                # Backward + optimize.
                self.optimizer.zero_grad()
                total_loss.backward()

                if self.config.gradient_clip > 0:
                    torch.nn.utils.clip_grad_norm_(
                        self.faculty.parameters(),
                        self.config.gradient_clip,
                    )

                self.optimizer.step()

                if self.scheduler is not None:
                    self.scheduler.step()

                self._fire("on_step_end", self._global_step, outputs, losses)
                self._global_step += 1

            phase.on_phase_end()
            self._fire("on_phase_end", phase_cfg.name)

        self._fire("on_train_end")
        logger.info("Training complete at global_step=%d", self._global_step)

    def add_callback(self, callback: Callback) -> None:
        """Add a training callback.

        Args:
            callback: An object implementing the ``Callback`` protocol.
        """
        self._callbacks.append(callback)

    def save_checkpoint(self, path: str, global_step: int) -> None:
        """Save model and optimizer state to disk.

        Args:
            path: File path for the checkpoint.
            global_step: Current global step to store in the checkpoint.
        """
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        torch.save(
            {
                "global_step": global_step,
                "model_state_dict": self.faculty.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "scheduler_state_dict": (
                    self.scheduler.state_dict() if self.scheduler is not None else None
                ),
            },
            path,
        )
        logger.info("Checkpoint saved to '%s' (step %d)", path, global_step)

    def load_checkpoint(self, path: str) -> int:
        """Load model and optimizer state from a checkpoint.

        Args:
            path: Path to the checkpoint file.

        Returns:
            The ``global_step`` stored in the checkpoint.
        """
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        self.faculty.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        scheduler_state = checkpoint.get("scheduler_state_dict")
        if scheduler_state is not None and self.scheduler is not None:
            self.scheduler.load_state_dict(scheduler_state)

        self._global_step = checkpoint.get("global_step", 0)
        logger.info("Loaded checkpoint from '%s' (step %d)", path, self._global_step)
        return self._global_step

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_phases(self) -> list[TrainingPhase]:
        """Instantiate training phases from the config using the registry."""
        # Ensure phase modules are imported so decorators run.
        import lfm.training.phases  # noqa: F401

        phases: list[TrainingPhase] = []
        for phase_cfg in self.config.phases:
            phase = create("phase", phase_cfg.name, config=phase_cfg, faculty=self.faculty)
            if not isinstance(phase, TrainingPhase):
                raise TypeError(
                    f"Phase '{phase_cfg.name}' resolved to {type(phase).__name__}, "
                    f"which is not a TrainingPhase subclass."
                )
            phases.append(phase)
        return phases

    def _build_optimizer(self) -> torch.optim.Optimizer:
        """Build the optimizer from config."""
        opt_cfg = self.config.optimizer
        name = opt_cfg.name.lower()

        if name == "adamw":
            return torch.optim.AdamW(
                self.faculty.parameters(),
                lr=opt_cfg.lr,
                weight_decay=opt_cfg.weight_decay,
                betas=opt_cfg.betas,
                eps=opt_cfg.eps,
            )
        if name == "adam":
            return torch.optim.Adam(
                self.faculty.parameters(),
                lr=opt_cfg.lr,
                weight_decay=opt_cfg.weight_decay,
                betas=opt_cfg.betas,
                eps=opt_cfg.eps,
            )
        if name == "sgd":
            return torch.optim.SGD(
                self.faculty.parameters(),
                lr=opt_cfg.lr,
                weight_decay=opt_cfg.weight_decay,
            )
        raise ValueError(f"Unknown optimizer name: {name!r}. Supported: 'adamw', 'adam', 'sgd'.")

    def _build_scheduler(self) -> torch.optim.lr_scheduler.LRScheduler | None:
        """Build the learning-rate scheduler from config."""
        sched_cfg = self.config.scheduler
        name = sched_cfg.name.lower()

        if name == "cosine":
            return torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=self.config.total_steps - sched_cfg.warmup_steps,
                eta_min=sched_cfg.min_lr,
            )
        if name == "linear":
            return torch.optim.lr_scheduler.LinearLR(
                self.optimizer,
                start_factor=1.0,
                end_factor=sched_cfg.min_lr / self.config.optimizer.lr,
                total_iters=self.config.total_steps,
            )
        if name == "none":
            return None
        raise ValueError(
            f"Unknown scheduler name: {name!r}. Supported: 'cosine', 'linear', 'none'."
        )

    def _set_lr_scale(self, scale: float) -> None:
        """Scale the learning rate for all optimizer parameter groups."""
        base_lr = self.config.optimizer.lr
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = base_lr * scale

    def _to_device(self, batch: dict[str, Any]) -> dict[str, Any]:
        """Move all tensors in a batch to the training device."""
        moved: dict[str, Any] = {}
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                moved[k] = v.to(self.device)
            else:
                moved[k] = v
        return moved

    def _fire(self, event: str, *args: Any) -> None:
        """Dispatch a callback event to all registered callbacks."""
        for cb in self._callbacks:
            handler = getattr(cb, event, None)
            if handler is not None:
                handler(*args)

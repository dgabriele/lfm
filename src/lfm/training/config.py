"""Configuration models for the LFM training pipeline.

Defines optimizer, scheduler, training-phase, and top-level training
configuration.  All models inherit from ``LFMBaseConfig`` for immutability
and strict validation.
"""

from __future__ import annotations

from lfm.config.base import LFMBaseConfig


class OptimizerConfig(LFMBaseConfig):
    """Configuration for the optimizer.

    Attributes:
        name: Optimizer name (e.g. ``"adamw"``, ``"sgd"``).
        lr: Base learning rate.
        weight_decay: L2 regularization coefficient.
        betas: Adam-family momentum coefficients.
        eps: Numerical stability term for Adam-family optimizers.
    """

    name: str = "adamw"
    lr: float = 1e-4
    weight_decay: float = 0.01
    betas: tuple[float, float] = (0.9, 0.999)
    eps: float = 1e-8


class SchedulerConfig(LFMBaseConfig):
    """Configuration for the learning-rate scheduler.

    Attributes:
        name: Scheduler name (e.g. ``"cosine"``, ``"linear"``).
        warmup_steps: Number of linear-warmup steps at the start.
        min_lr: Minimum learning rate at the end of the schedule.
    """

    name: str = "cosine"
    warmup_steps: int = 1000
    min_lr: float = 1e-6


class PhaseConfig(LFMBaseConfig):
    """Configuration for a single training phase.

    A training run consists of one or more phases executed in sequence.
    Each phase may use a different loss weighting, freeze different
    sub-modules, or draw data from different sources.

    Attributes:
        name: Registry lookup key for the phase implementation.
        steps: Number of training steps in this phase.
        losses: Mapping of registered loss names to their weights.
        modules_frozen: Sub-module names to freeze during this phase
            (e.g. ``["quantizer"]``).
        lr_scale: Multiplier applied to the base learning rate for this
            phase.
        data_source: Data source identifier (``"corpus"`` or ``"agent"``).
    """

    name: str
    steps: int
    losses: dict[str, float]
    modules_frozen: list[str] = []
    lr_scale: float = 1.0
    data_source: str = "corpus"


class TrainingConfig(LFMBaseConfig):
    """Top-level training configuration.

    Attributes:
        phases: Ordered list of training phases to execute.
        optimizer: Optimizer configuration.
        scheduler: Learning-rate scheduler configuration.
        total_steps: Maximum total training steps across all phases.
        checkpoint_dir: Directory for saving checkpoints.
        checkpoint_every: Save a checkpoint every *N* steps.
        log_every: Log training metrics every *N* steps.
        gradient_clip: Maximum gradient norm for clipping.
        device: Device string (e.g. ``"cuda"``, ``"cpu"``).
    """

    phases: list[PhaseConfig] = []
    optimizer: OptimizerConfig = OptimizerConfig()
    scheduler: SchedulerConfig = SchedulerConfig()
    total_steps: int = 100_000
    checkpoint_dir: str = "checkpoints"
    checkpoint_every: int = 5000
    log_every: int = 100
    gradient_clip: float = 1.0
    device: str = "cuda"

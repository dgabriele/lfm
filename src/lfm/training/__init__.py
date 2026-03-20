"""LFM training pipeline.

Provides a multi-phase training loop, configurable callbacks, and a registry
of training phases.  Importing this package triggers registration of all
built-in phase implementations.
"""

from __future__ import annotations

# Import phases sub-package to trigger registration of built-in phases.
import lfm.training.phases as phases  # noqa: F401
from lfm.training.callbacks import (
    Callback,
    CheckpointCallback,
    LoggingCallback,
    MetricsCallback,
)
from lfm.training.config import (
    OptimizerConfig,
    PhaseConfig,
    SchedulerConfig,
    TrainingConfig,
)
from lfm.training.loop import TrainingLoop
from lfm.training.phase import TrainingPhase

__all__ = [
    "Callback",
    "CheckpointCallback",
    "LoggingCallback",
    "MetricsCallback",
    "OptimizerConfig",
    "PhaseConfig",
    "SchedulerConfig",
    "TrainingConfig",
    "TrainingLoop",
    "TrainingPhase",
]

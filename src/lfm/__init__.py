"""LFM — Language Faculty Model.

A constraint-driven, compositional, morphosyntactic prior that enables
structured communication without predefined semantics for GPU-based
multi-agent systems.

Quick start::

    from lfm import ExperimentConfig, FacultyConfig, LanguageFaculty

    faculty = LanguageFaculty(FacultyConfig(dim=128))
"""

from __future__ import annotations

__version__ = "0.1.0"

# --- Core abstractions ---
from lfm._registry import create, list_registered, register

# --- Configs ---
from lfm.channel.config import ChannelConfig
from lfm.config.base import LFMBaseConfig
from lfm.config.experiment import ExperimentConfig
from lfm.core.loss import CompositeLoss, LFMLoss
from lfm.core.module import LFMModule
from lfm.data.config import DataConfig
from lfm.faculty.config import FacultyConfig

# --- Faculty ---
from lfm.faculty.model import LanguageFaculty
from lfm.morphology.config import MorphologyConfig
from lfm.phonology.config import PhonologyConfig
from lfm.quantization.config import QuantizationConfig
from lfm.sentence.config import SentenceConfig
from lfm.syntax.config import SyntaxConfig
from lfm.training.config import (
    OptimizerConfig,
    PhaseConfig,
    SchedulerConfig,
    TrainingConfig,
)

# --- Training ---
from lfm.training.loop import TrainingLoop
from lfm.training.phase import TrainingPhase

__all__ = [
    # Version
    "__version__",
    # Registry
    "create",
    "list_registered",
    "register",
    # Core
    "CompositeLoss",
    "LFMLoss",
    "LFMModule",
    # Configs
    "ChannelConfig",
    "DataConfig",
    "ExperimentConfig",
    "FacultyConfig",
    "LFMBaseConfig",
    "MorphologyConfig",
    "OptimizerConfig",
    "PhaseConfig",
    "PhonologyConfig",
    "QuantizationConfig",
    "SchedulerConfig",
    "SentenceConfig",
    "SyntaxConfig",
    "TrainingConfig",
    # Faculty
    "LanguageFaculty",
    # Training
    "TrainingLoop",
    "TrainingPhase",
]

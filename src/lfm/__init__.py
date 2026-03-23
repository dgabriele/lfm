"""LFM — Language Faculty Model.

A framework for giving neural agents a natural language faculty via a
pretrained multilingual VAE decoder that produces linguistically
structured IPA output from agent embeddings.

Quick start::

    from lfm import FacultyConfig, GeneratorConfig, LanguageFaculty

    faculty = LanguageFaculty(FacultyConfig(
        dim=384,
        generator=GeneratorConfig(
            pretrained_decoder_path="data/vae_decoder.pt",
        ),
    ))
"""

from __future__ import annotations

__version__ = "0.1.0"

# --- Core abstractions ---
from lfm._registry import create, list_registered, register
from lfm._types import TokenBridge, TokenBridgeOutput
from lfm.config.base import LFMBaseConfig
from lfm.config.experiment import ExperimentConfig
from lfm.core.loss import CompositeLoss, LFMLoss
from lfm.core.module import LFMModule
from lfm.data.config import DataConfig

# --- Faculty + Generator ---
from lfm.faculty.config import FacultyConfig
from lfm.faculty.model import LanguageFaculty
from lfm.generator.config import GeneratorConfig

# --- Training ---
from lfm.training.config import (
    OptimizerConfig,
    PhaseConfig,
    SchedulerConfig,
    TrainingConfig,
)
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
    "DataConfig",
    "ExperimentConfig",
    "FacultyConfig",
    "GeneratorConfig",
    "LFMBaseConfig",
    "OptimizerConfig",
    "PhaseConfig",
    "SchedulerConfig",
    "TrainingConfig",
    # Types
    "TokenBridge",
    "TokenBridgeOutput",
    # Faculty
    "LanguageFaculty",
    # Training
    "TrainingLoop",
    "TrainingPhase",
]

"""Top-level experiment configuration.

``ExperimentConfig`` composes all sub-configurations (faculty, training,
data) into a single object that fully specifies an LFM experiment.
"""

from __future__ import annotations

from lfm.config.base import LFMBaseConfig
from lfm.data.config import DataConfig
from lfm.faculty.config import FacultyConfig
from lfm.training.config import TrainingConfig


class ExperimentConfig(LFMBaseConfig):
    """Complete experiment configuration.

    This is the top-level entry point for configuring an LFM experiment.
    It composes the faculty architecture, training pipeline, and data
    loading into a single immutable config object.

    Attributes:
        faculty: Configuration for the LanguageFaculty pipeline.
        training: Configuration for the training loop and phases.
        data: Configuration for data loading and preprocessing.
        seed: Random seed for reproducibility.
        device: Device string (e.g. ``"cuda"``, ``"cpu"``).
    """

    faculty: FacultyConfig = FacultyConfig()
    training: TrainingConfig = TrainingConfig()
    data: DataConfig = DataConfig()
    seed: int = 42
    device: str = "cuda"

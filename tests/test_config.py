"""Tests for the configuration system."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from lfm.config.base import LFMBaseConfig
from lfm.config.experiment import ExperimentConfig
from lfm.faculty.config import FacultyConfig
from lfm.quantization.config import QuantizationConfig
from lfm.training.config import PhaseConfig, TrainingConfig


def test_base_config_is_frozen():
    """LFMBaseConfig instances are immutable."""
    fc = FacultyConfig()
    with pytest.raises(ValidationError):
        fc.dim = 999  # type: ignore[misc]


def test_base_config_forbids_extra():
    """Extra fields raise ValidationError."""

    class _StrictConfig(LFMBaseConfig):
        x: int = 1

    with pytest.raises(ValidationError):
        _StrictConfig(x=1, y=2)  # type: ignore[call-arg]


def test_faculty_config_defaults():
    """FacultyConfig has sensible defaults."""
    fc = FacultyConfig()
    assert fc.dim == 256
    assert fc.max_seq_len == 64
    assert fc.phonology is not None  # enabled by default
    assert fc.quantizer is None
    assert fc.morphology is None
    assert fc.syntax is None
    assert fc.sentence is None
    assert fc.channel is None


def test_faculty_config_with_quantizer():
    """FacultyConfig accepts nested sub-module configs."""
    fc = FacultyConfig(
        dim=128,
        quantizer=QuantizationConfig(codebook_size=256, codebook_dim=32),
    )
    assert fc.quantizer is not None
    assert fc.quantizer.codebook_size == 256
    assert fc.quantizer.codebook_dim == 32


def test_experiment_config_composition():
    """ExperimentConfig composes all sub-configs."""
    exp = ExperimentConfig(
        faculty=FacultyConfig(dim=64),
        training=TrainingConfig(
            phases=[
                PhaseConfig(
                    name="structural_priors",
                    steps=1000,
                    losses={"well_formedness": 1.0},
                ),
            ],
        ),
        seed=123,
    )
    assert exp.faculty.dim == 64
    assert len(exp.training.phases) == 1
    assert exp.training.phases[0].name == "structural_priors"
    assert exp.seed == 123


def test_phase_config_defaults():
    """PhaseConfig has correct defaults for optional fields."""
    pc = PhaseConfig(name="test", steps=100, losses={"x": 1.0})
    assert pc.modules_frozen == []
    assert pc.lr_scale == 1.0
    assert pc.data_source == "corpus"

"""Tests for LanguageFaculty instantiation and config wiring."""

from __future__ import annotations

from lfm import list_registered
from lfm.faculty.config import FacultyConfig
from lfm.faculty.model import LanguageFaculty
from lfm.quantization.config import QuantizationConfig


def test_faculty_minimal():
    """LanguageFaculty with all modules disabled."""
    config = FacultyConfig(phonology=None)  # disable the default phonology
    faculty = LanguageFaculty(config)
    assert faculty.quantizer is None
    assert faculty.phonology is None
    assert faculty.morphology is None
    assert faculty.syntax is None
    assert faculty.sentence is None
    assert faculty.channel is None


def test_faculty_with_quantizer():
    """LanguageFaculty instantiates a quantizer from config."""
    config = FacultyConfig(
        dim=128,
        quantizer=QuantizationConfig(
            name="vqvae",
            codebook_size=256,
            codebook_dim=64,
            input_dim=128,
        ),
        phonology=None,
    )
    faculty = LanguageFaculty(config)
    assert faculty.quantizer is not None
    assert type(faculty.quantizer).__name__ == "VQVAEQuantizer"


def test_faculty_creates_projections():
    """Dim mismatch between quantizer and faculty creates a projection."""
    config = FacultyConfig(
        dim=128,
        quantizer=QuantizationConfig(
            name="vqvae",
            codebook_size=256,
            codebook_dim=64,  # != 128
            input_dim=128,
        ),
        phonology=None,
    )
    faculty = LanguageFaculty(config)
    assert "quantizer_to_faculty" in faculty.projections


def test_faculty_no_projection_when_dims_match():
    """No projection is created when dims already match."""
    config = FacultyConfig(
        dim=64,
        quantizer=QuantizationConfig(
            name="vqvae",
            codebook_size=256,
            codebook_dim=64,  # == 64
            input_dim=64,
        ),
        phonology=None,
    )
    faculty = LanguageFaculty(config)
    assert "quantizer_to_faculty" not in faculty.projections


def test_registry_populated_after_faculty():
    """Importing LanguageFaculty populates the registry with concrete types."""
    # Force registry population
    LanguageFaculty(FacultyConfig(phonology=None))
    assert "vqvae" in list_registered("quantizer")
    assert "fsq" in list_registered("quantizer")
    assert "lfq" in list_registered("quantizer")
    assert "morphological_well_formedness" in list_registered("loss")
    assert "morpheme_reuse" in list_registered("loss")
    assert "pronounceable" in list_registered("phonology")

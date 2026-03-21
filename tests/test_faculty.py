"""Tests for LanguageFaculty instantiation and config wiring."""

from __future__ import annotations

import pytest
import torch

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
    assert "surface" in list_registered("phonology")


# -- Pre-tokenized path tests -----------------------------------------------


def test_faculty_pretokenized_forward():
    """Pre-tokenized (tokens, embeddings) bypass quantizer."""
    fc = FacultyConfig(dim=64, quantizer=None, phonology=None)
    faculty = LanguageFaculty(fc)
    tokens = torch.randint(0, 100, (2, 10))
    embeddings = torch.randn(2, 10, 64)
    out = faculty(tokens=tokens, embeddings=embeddings)
    assert "pretokenized.tokens" in out
    assert "pretokenized.embeddings" in out
    assert "quantization.tokens" not in out


def test_faculty_pretokenized_dim_projection():
    """Pre-tokenized embeddings projected when pretokenized_dim set."""
    fc = FacultyConfig(dim=64, pretokenized_dim=32, quantizer=None, phonology=None)
    faculty = LanguageFaculty(fc)
    tokens = torch.randint(0, 100, (2, 10))
    embeddings = torch.randn(2, 10, 32)
    out = faculty(tokens=tokens, embeddings=embeddings)
    assert "pretokenized.embeddings" in out


def test_faculty_pretokenized_rejects_both():
    """ValueError when both agent_state and tokens provided."""
    fc = FacultyConfig(dim=64, quantizer=None, phonology=None)
    faculty = LanguageFaculty(fc)
    with pytest.raises(ValueError, match="not both"):
        faculty(
            agent_state=torch.randn(2, 64),
            tokens=torch.randint(0, 10, (2, 5)),
            embeddings=torch.randn(2, 5, 64),
        )


def test_faculty_pretokenized_rejects_partial():
    """ValueError when only tokens or only embeddings provided."""
    fc = FacultyConfig(dim=64, quantizer=None, phonology=None)
    faculty = LanguageFaculty(fc)
    with pytest.raises(ValueError, match="both tokens and embeddings"):
        faculty(tokens=torch.randint(0, 10, (2, 5)))


def test_faculty_pretokenized_rejects_dim_mismatch():
    """ValueError on dim mismatch without pretokenized_dim configured."""
    fc = FacultyConfig(dim=64, quantizer=None, phonology=None)
    faculty = LanguageFaculty(fc)
    with pytest.raises(ValueError, match="pretokenized_dim"):
        faculty(tokens=torch.randint(0, 10, (2, 5)), embeddings=torch.randn(2, 5, 32))


def test_faculty_pretokenized_with_phonology():
    """Pre-tokenized input flows through phonology."""
    fc = FacultyConfig(dim=64, quantizer=None)  # phonology on by default
    faculty = LanguageFaculty(fc)
    tokens = torch.randint(0, 100, (2, 10))
    embeddings = torch.randn(2, 10, 64)
    out = faculty(tokens=tokens, embeddings=embeddings)
    assert "phonology.surface_forms" in out
    assert "phonology.pronounceability_score" in out


def test_faculty_agent_state_still_works():
    """Existing agent_state path unaffected by new signature."""
    fc = FacultyConfig(dim=64, pretokenized_dim=32, quantizer=None)
    faculty = LanguageFaculty(fc)
    out = faculty(torch.randn(2, 64))  # positional, as before
    assert "phonology.surface_forms" in out

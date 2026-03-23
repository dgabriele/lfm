"""Tests for LanguageFaculty instantiation and config wiring."""

from __future__ import annotations

import pytest
import torch

from lfm import list_registered
from lfm.faculty.config import FacultyConfig
from lfm.faculty.model import LanguageFaculty


def test_faculty_minimal():
    """LanguageFaculty with no generator."""
    config = FacultyConfig()
    faculty = LanguageFaculty(config)
    assert faculty.generator is None


def test_faculty_agent_state_passthrough():
    """Agent state passes through without generator."""
    fc = FacultyConfig(dim=64)
    faculty = LanguageFaculty(fc)
    out = faculty(torch.randn(2, 64))
    assert "extra_losses" in out


def test_faculty_pretokenized_forward():
    """Pre-tokenized (tokens, embeddings) input works."""
    fc = FacultyConfig(dim=64)
    faculty = LanguageFaculty(fc)
    tokens = torch.randint(0, 100, (2, 10))
    embeddings = torch.randn(2, 10, 64)
    out = faculty(tokens=tokens, embeddings=embeddings)
    assert "pretokenized.tokens" in out
    assert "pretokenized.embeddings" in out


def test_faculty_pretokenized_dim_projection():
    """Pre-tokenized embeddings projected when pretokenized_dim set."""
    fc = FacultyConfig(dim=64, pretokenized_dim=32)
    faculty = LanguageFaculty(fc)
    tokens = torch.randint(0, 100, (2, 10))
    embeddings = torch.randn(2, 10, 32)
    out = faculty(tokens=tokens, embeddings=embeddings)
    assert "pretokenized.embeddings" in out


def test_faculty_pretokenized_rejects_both():
    """ValueError when both agent_state and tokens provided."""
    fc = FacultyConfig(dim=64)
    faculty = LanguageFaculty(fc)
    with pytest.raises(ValueError, match="not both"):
        faculty(
            agent_state=torch.randn(2, 64),
            tokens=torch.randint(0, 10, (2, 5)),
            embeddings=torch.randn(2, 5, 64),
        )


def test_faculty_pretokenized_rejects_partial():
    """ValueError when only tokens or only embeddings provided."""
    fc = FacultyConfig(dim=64)
    faculty = LanguageFaculty(fc)
    with pytest.raises(ValueError, match="both tokens and embeddings"):
        faculty(tokens=torch.randint(0, 10, (2, 5)))


def test_registry_populated_after_faculty():
    """Importing LanguageFaculty populates the generator registry."""
    LanguageFaculty(FacultyConfig())
    assert "multilingual_vae" in list_registered("generator")
    assert "leipzig" in list_registered("corpus_loader")

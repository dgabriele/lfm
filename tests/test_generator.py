"""Tests for the generator module and its faculty integration."""

from __future__ import annotations

import pytest
import torch

from lfm.faculty.config import FacultyConfig
from lfm.faculty.model import LanguageFaculty
from lfm.generator.config import GeneratorConfig

# -- Shared helpers ----------------------------------------------------------

def _small_config(**overrides: object) -> GeneratorConfig:
    """Create a small generator config for fast tests."""
    defaults = dict(
        latent_dim=32,
        vocab_size=50,
        max_output_len=8,
        decoder_hidden_dim=64,
        decoder_num_layers=2,
        decoder_num_heads=2,
        decoder_dropout=0.0,
        kl_weight=0.1,
        freeze_decoder=False,
        attention_head_windows=(3, 0),
        attention_global_every=4,
        use_rope=True,
        share_decoder_layers=True,
    )
    defaults.update(overrides)
    return GeneratorConfig(**defaults)


def _make_generator(**overrides: object) -> LanguageFaculty:
    """Create a LanguageFaculty with only a generator (no other stages)."""
    return LanguageFaculty(FacultyConfig(
        dim=64,
        generator=_small_config(**overrides),
        quantizer=None,
        phonology=None,
    ))


# -- Config tests ------------------------------------------------------------

def test_generator_config_frozen():
    """GeneratorConfig is immutable."""
    cfg = _small_config()
    with pytest.raises(Exception):
        cfg.latent_dim = 999


def test_generator_config_forbids_extra():
    """GeneratorConfig rejects unknown fields."""
    with pytest.raises(Exception):
        GeneratorConfig(nonexistent_field=42)


# -- Generator forward tests -------------------------------------------------

def test_generator_forward_basic():
    """Generator produces correct output keys and shapes."""
    faculty = _make_generator()
    embeddings = torch.randn(2, 5, 64)

    out = faculty(tokens=torch.zeros(2, 5, dtype=torch.long), embeddings=embeddings)

    # Generator outputs should be namespaced
    assert "generator.tokens" in out
    assert "generator.token_probs" in out
    assert "generator.embeddings" in out
    assert "generator.lengths" in out
    assert "generator.mask" in out
    assert "generator.mu" in out
    assert "generator.logvar" in out

    # Shape checks
    assert out["generator.tokens"].shape == (2, 8)  # max_output_len=8
    assert out["generator.token_probs"].shape == (2, 8, 52)  # vocab_size+2
    assert out["generator.embeddings"].shape == (2, 8, 64)  # decoder_hidden_dim
    assert out["generator.lengths"].shape == (2,)
    assert out["generator.mask"].shape == (2, 8)
    assert out["generator.mu"].shape == (2, 32)  # latent_dim
    assert out["generator.logvar"].shape == (2, 32)


def test_generator_forward_agent_state():
    """Generator works with raw agent_state input (no quantizer)."""
    faculty = _make_generator()
    out = faculty(torch.randn(2, 64))

    assert "generator.tokens" in out
    assert out["generator.tokens"].shape[0] == 2


def test_generator_forward_variable_length():
    """Output mask reflects computed lengths."""
    faculty = _make_generator()
    out = faculty(torch.randn(3, 64))

    mask = out["generator.mask"]
    lengths = out["generator.lengths"]

    # Each row of mask should have exactly `length` True values
    for i in range(3):
        assert mask[i].sum().item() == lengths[i].item()


def test_generator_extra_losses_kl():
    """KL divergence appears in extra_losses after forward."""
    faculty = _make_generator()
    out = faculty(torch.randn(2, 64))

    extra = out["extra_losses"]
    assert "generator.kl_divergence" in extra
    assert extra["generator.kl_divergence"].shape == ()  # scalar
    assert extra["generator.kl_divergence"].item() >= 0


def test_generator_freeze_decoder():
    """When freeze_decoder=True, decoder params are frozen but input proj is trainable."""
    faculty = LanguageFaculty(FacultyConfig(
        dim=64,
        generator=_small_config(freeze_decoder=True),
        quantizer=None,
        phonology=None,
    ))

    gen = faculty.generator
    assert gen is not None

    # Trigger lazy init of _input_proj
    faculty(torch.randn(2, 64))

    # Decoder params should be frozen
    for param in gen.decoder.parameters():
        assert not param.requires_grad
    for param in gen.output_head.parameters():
        assert not param.requires_grad
    for param in gen.token_embedding.parameters():
        assert not param.requires_grad

    # Input projection should be trainable
    assert gen._input_proj is not None
    for param in gen._input_proj.parameters():
        assert param.requires_grad


def test_generator_dim_projection():
    """Decoder hidden_dim != faculty dim creates a projection layer."""
    faculty = LanguageFaculty(FacultyConfig(
        dim=128,  # different from decoder_hidden_dim=64
        generator=_small_config(decoder_hidden_dim=64),
        quantizer=None,
        phonology=None,
    ))
    assert "generator_to_faculty" in faculty.projections


def test_generator_no_projection_when_dims_match():
    """No projection when decoder_hidden_dim == faculty dim."""
    faculty = LanguageFaculty(FacultyConfig(
        dim=64,
        generator=_small_config(decoder_hidden_dim=64),
        quantizer=None,
        phonology=None,
    ))
    assert "generator_to_faculty" not in faculty.projections


# -- Faculty integration tests -----------------------------------------------

def test_faculty_with_generator():
    """Full faculty forward with generator produces correct namespaces."""
    faculty = _make_generator()
    out = faculty(torch.randn(2, 64))

    # Generator outputs present
    assert "generator.tokens" in out

    # No quantization outputs (quantizer=None)
    assert "quantization.tokens" not in out

    # No phonology outputs (phonology=None)
    assert "phonology.surface_forms" not in out


def test_faculty_generator_with_phonology():
    """Generator output flows through phonology when both configured."""
    faculty_with_phon = LanguageFaculty(FacultyConfig(
        dim=64,
        generator=_small_config(decoder_hidden_dim=64),
        quantizer=None,
        # phonology enabled by default
    ))
    out = faculty_with_phon(torch.randn(2, 64))

    # Both generator and phonology outputs should be present
    assert "generator.tokens" in out
    assert "phonology.surface_forms" in out
    assert "phonology.pronounceability_score" in out


def test_faculty_generator_agent_state_compat():
    """Existing agent_state path works when generator is NOT configured."""
    faculty = LanguageFaculty(FacultyConfig(
        dim=64,
        generator=None,
        quantizer=None,
    ))
    out = faculty(torch.randn(2, 64))

    # Should have phonology (default) but no generator
    assert "phonology.surface_forms" in out
    assert "generator.tokens" not in out


def test_generator_eval_deterministic():
    """Generator is deterministic in eval mode (no reparameterization noise)."""
    faculty = _make_generator()
    faculty.eval()

    x = torch.randn(2, 64)
    torch.manual_seed(42)
    out1 = faculty(x)
    torch.manual_seed(42)
    out2 = faculty(x)

    # mu should be identical (deterministic path)
    assert torch.equal(out1["generator.mu"], out2["generator.mu"])


def test_generator_registered():
    """MultilingualVAEGenerator is registered in the global registry."""
    from lfm import list_registered

    # Force registry population
    LanguageFaculty(FacultyConfig(phonology=None))
    assert "multilingual_vae" in list_registered("generator")

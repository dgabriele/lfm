"""Tests for phonotactic structural priors."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
from torch import nn

from lfm.phonology.config import PhonologyConfig
from lfm.phonology.priors import PANPHON_DIM, PhonotacticDataset, PhonotacticPriorConfig
from lfm.phonology.surface import SurfacePhonology

# Check panphon availability for conditional tests
try:
    import panphon  # noqa: F401

    _has_panphon = True
except ImportError:
    _has_panphon = False


# ------------------------------------------------------------------
# Backward compatibility
# ------------------------------------------------------------------


def test_phonology_config_backward_compat():
    """PhonologyConfig() still works with pretrained_smoothness_path=None."""
    config = PhonologyConfig()
    assert config.pretrained_smoothness_path is None
    assert config.surface_dim == 12
    assert config.smoothness_hidden_dim == 32


def test_surface_phonology_no_pretrained():
    """SurfacePhonology(PhonologyConfig()) works unchanged."""
    config = PhonologyConfig()
    module = SurfacePhonology(config)
    assert module.surface_dim == 12
    # Should work without any pretrained path
    tokens = torch.zeros(2, 4, dtype=torch.long)
    embeddings = torch.randn(2, 4, 32)
    out = module(tokens, embeddings)
    assert "surface_forms" in out
    assert "pronounceability_score" in out


# ------------------------------------------------------------------
# Pretrained weight loading
# ------------------------------------------------------------------


def _make_checkpoint(
    tmp_path: Path,
    surface_dim: int = 12,
    smoothness_hidden_dim: int = 32,
) -> Path:
    """Create a mock checkpoint with matching dimensions."""
    gru = nn.GRU(surface_dim, smoothness_hidden_dim, batch_first=True)
    head = nn.Linear(smoothness_hidden_dim, surface_dim)
    proj = nn.Linear(PANPHON_DIM, surface_dim)

    path = tmp_path / "mock_prior.pt"
    torch.save(
        {
            "surface_dim": surface_dim,
            "smoothness_hidden_dim": smoothness_hidden_dim,
            "panphon_dim": PANPHON_DIM,
            "smoothness_gru": gru.state_dict(),
            "smoothness_head": head.state_dict(),
            "feature_proj": proj.state_dict(),
            "train_loss": 0.5,
            "val_loss": 0.6,
            "num_languages": 10,
            "num_samples": 1000,
        },
        path,
    )
    return path


def test_surface_phonology_load_pretrained(tmp_path: Path):
    """Mock checkpoint with matching dims loads correctly."""
    ckpt_path = _make_checkpoint(tmp_path, surface_dim=12, smoothness_hidden_dim=32)

    # Load with pretrained weights
    config = PhonologyConfig(pretrained_smoothness_path=str(ckpt_path))
    module = SurfacePhonology(config)

    # Weights should be loaded from checkpoint (different from random init)
    # Load the checkpoint to verify weights match
    checkpoint = torch.load(ckpt_path, weights_only=True)
    for key, value in checkpoint["smoothness_gru"].items():
        param = dict(module.smoothness_gru.named_parameters())[key]
        assert torch.equal(param.data, value), f"GRU parameter {key} not loaded correctly"

    for key, value in checkpoint["smoothness_head"].items():
        param = dict(module.smoothness_head.named_parameters())[key]
        assert torch.equal(param.data, value), f"Head parameter {key} not loaded correctly"


def test_surface_phonology_dim_mismatch(tmp_path: Path):
    """Clear error when checkpoint surface_dim doesn't match config."""
    ckpt_path = _make_checkpoint(tmp_path, surface_dim=24, smoothness_hidden_dim=32)

    config = PhonologyConfig(
        surface_dim=12,
        pretrained_smoothness_path=str(ckpt_path),
    )
    with pytest.raises(ValueError, match="surface_dim"):
        SurfacePhonology(config)


def test_surface_phonology_hidden_dim_mismatch(tmp_path: Path):
    """Clear error when checkpoint smoothness_hidden_dim doesn't match config."""
    ckpt_path = _make_checkpoint(tmp_path, surface_dim=12, smoothness_hidden_dim=64)

    config = PhonologyConfig(
        surface_dim=12,
        smoothness_hidden_dim=32,
        pretrained_smoothness_path=str(ckpt_path),
    )
    with pytest.raises(ValueError, match="smoothness_hidden_dim"):
        SurfacePhonology(config)


def test_surface_phonology_file_not_found():
    """FileNotFoundError when pretrained path doesn't exist."""
    config = PhonologyConfig(pretrained_smoothness_path="nonexistent.pt")
    with pytest.raises(FileNotFoundError):
        SurfacePhonology(config)


# ------------------------------------------------------------------
# Dataset
# ------------------------------------------------------------------


def test_phonotactic_dataset():
    """PhonotacticDataset pads and returns correct shapes."""
    import numpy as np

    arrays = [
        np.random.randn(5, PANPHON_DIM).astype(np.float32),
        np.random.randn(3, PANPHON_DIM).astype(np.float32),
        np.random.randn(10, PANPHON_DIM).astype(np.float32),
    ]
    dataset = PhonotacticDataset(arrays, max_word_length=8)
    assert len(dataset) == 3

    features, length = dataset[0]
    assert features.shape == (8, PANPHON_DIM)
    assert length == 5

    features, length = dataset[1]
    assert features.shape == (8, PANPHON_DIM)
    assert length == 3

    features, length = dataset[2]
    assert features.shape == (8, PANPHON_DIM)
    assert length == 8  # Truncated from 10 to 8


# ------------------------------------------------------------------
# Panphon-dependent tests
# ------------------------------------------------------------------


@pytest.mark.skipif(not _has_panphon, reason="panphon not installed")
def test_feature_converter():
    """Known IPA segments produce correct shape and values."""
    from lfm.phonology.priors import FeatureConverter

    converter = FeatureConverter()

    # Common IPA segments that panphon should recognize
    result = converter.segments_to_features(["p", "a", "t"])
    assert result is not None
    assert result.shape == (3, PANPHON_DIM)
    # Values should be in {-1, 0, +1}
    unique_vals = set(result.flatten().tolist())
    assert unique_vals.issubset({-1.0, 0.0, 1.0})


@pytest.mark.skipif(not _has_panphon, reason="panphon not installed")
def test_feature_converter_unrecognized():
    """Mostly unrecognized segments return None."""
    from lfm.phonology.priors import FeatureConverter

    converter = FeatureConverter()
    # Garbage segments — more than 50% should fail
    result = converter.segments_to_features(["###", "???", "!!!", "p"])
    assert result is None


@pytest.mark.skipif(not _has_panphon, reason="panphon not installed")
def test_wikipron_loader(tmp_path: Path):
    """Synthetic TSV parsing works correctly."""
    from lfm.phonology.priors import WikiPronLoader

    # Create a synthetic WikiPron TSV
    tsv_dir = tmp_path / "tsv"
    tsv_dir.mkdir()

    broad_file = tsv_dir / "eng_latn_broad.tsv"
    broad_file.write_text(
        "hello\th ɛ l oʊ\n"
        "world\tw ɜː l d\n"
        "a\tæ\n"  # Too short (1 segment, min is 2)
        "cat\tk æ t\n"
    )

    # Also create a narrow file that should be deprioritized
    narrow_file = tsv_dir / "eng_latn_narrow.tsv"
    narrow_file.write_text("hello\thɛˈloʊ\n")

    loader = WikiPronLoader(
        wikipron_dir=str(tsv_dir),
        max_samples_per_language=10,
        min_word_length=2,
    )
    samples = loader.load()

    # Should have 3 samples (hello, world, cat — 'a' is too short)
    assert len(samples) == 3
    assert all(lang == "eng" for lang, _ in samples)
    # First sample should be "hello" → ["h", "ɛ", "l", "oʊ"]
    assert samples[0][1] == ["h", "ɛ", "l", "oʊ"]


@pytest.mark.skipif(not _has_panphon, reason="panphon not installed")
def test_pretrain_end_to_end(tmp_path: Path):
    """Small synthetic dataset trains and produces loadable checkpoint."""
    from lfm.phonology.priors import pretrain_phonotactic_prior

    # Create synthetic WikiPron data
    tsv_dir = tmp_path / "tsv"
    tsv_dir.mkdir()

    # Generate enough samples for a meaningful split
    lines = []
    words = [
        ("hello", "h ɛ l oʊ"),
        ("world", "w ɜ l d"),
        ("cat", "k æ t"),
        ("dog", "d ɔ ɡ"),
        ("fish", "f ɪ ʃ"),
        ("bird", "b ɜ d"),
        ("tree", "t ɹ iː"),
        ("book", "b ʊ k"),
        ("hand", "h æ n d"),
        ("foot", "f ʊ t"),
        ("water", "w ɔ t ɜ"),
        ("stone", "s t oʊ n"),
    ]
    for word, ipa in words:
        lines.append(f"{word}\t{ipa}")

    (tsv_dir / "eng_latn_broad.tsv").write_text("\n".join(lines) + "\n")

    output_path = tmp_path / "prior.pt"
    config = PhonotacticPriorConfig(
        wikipron_dir=str(tsv_dir),
        max_samples_per_language=100,
        min_word_length=2,
        max_word_length=10,
        surface_dim=8,
        smoothness_hidden_dim=16,
        batch_size=4,
        lr=1e-3,
        num_epochs=2,
        val_fraction=0.2,
        seed=42,
        device="cpu",
        output_path=str(output_path),
    )

    metrics = pretrain_phonotactic_prior(config)

    assert "train_loss" in metrics
    assert "val_loss" in metrics
    assert metrics["num_samples"] > 0
    assert output_path.exists()

    # Verify checkpoint is loadable into SurfacePhonology
    phon_config = PhonologyConfig(
        surface_dim=8,
        smoothness_hidden_dim=16,
        pretrained_smoothness_path=str(output_path),
    )
    module = SurfacePhonology(phon_config)

    # Verify it runs
    tokens = torch.zeros(2, 3, dtype=torch.long)
    embeddings = torch.randn(2, 3, 32)
    out = module(tokens, embeddings)
    assert "surface_forms" in out

"""Tests for the translator module."""

from __future__ import annotations

import json
import tempfile

import pytest
from pydantic import ValidationError

from lfm.translator.config import PairGenerationConfig, TranslatorConfig

# ── Config tests ──────────────────────────────────────────────────


def test_pair_generation_config_defaults():
    """PairGenerationConfig has sensible defaults."""
    cfg = PairGenerationConfig()
    assert cfg.max_phrases == 16
    assert cfg.device == "cuda"


def test_pair_generation_config_is_frozen():
    """PairGenerationConfig is immutable."""
    cfg = PairGenerationConfig()
    with pytest.raises(ValidationError):
        cfg.max_phrases = 100  # type: ignore[misc]


def test_pair_generation_config_forbids_extra():
    """Extra fields raise ValidationError."""
    with pytest.raises(ValidationError):
        PairGenerationConfig(unknown_field="x")  # type: ignore[call-arg]


def test_translator_config_defaults():
    """TranslatorConfig has sensible defaults."""
    cfg = TranslatorConfig()
    assert cfg.model_name == "Qwen/Qwen2.5-0.5B"
    assert cfg.use_lora is False
    assert cfg.epochs == 3
    assert cfg.lr == 2e-5
    assert cfg.batch_size == 8
    assert cfg.gradient_accumulation_steps == 4
    assert cfg.use_amp is True
    assert cfg.warmup_fraction == 0.1
    assert cfg.max_grad_norm == 1.0
    assert cfg.val_fraction == 0.1


def test_translator_config_lora_fields():
    """TranslatorConfig LoRA fields are accessible."""
    cfg = TranslatorConfig(use_lora=True)
    assert cfg.use_lora is True
    assert cfg.lora_r == 16
    assert cfg.lora_alpha == 32
    assert cfg.lora_target_modules == ["q_proj", "v_proj"]


def test_translator_config_is_frozen():
    """TranslatorConfig is immutable."""
    cfg = TranslatorConfig()
    with pytest.raises(ValidationError):
        cfg.epochs = 10  # type: ignore[misc]


def test_translator_config_forbids_extra():
    """Extra fields raise ValidationError."""
    with pytest.raises(ValidationError):
        TranslatorConfig(unknown_field="x")  # type: ignore[call-arg]


def test_translator_config_serialization():
    """Config round-trips through model_dump."""
    cfg = TranslatorConfig(epochs=5, use_lora=True)
    d = cfg.model_dump()
    assert d["epochs"] == 5
    assert d["use_lora"] is True
    cfg2 = TranslatorConfig(**d)
    assert cfg2 == cfg


# ── Dataset tests ─────────────────────────────────────────────────


def test_dataset_from_jsonl():
    """IPATranslationDataset.from_jsonl loads pairs correctly."""
    from lfm.translator.dataset import IPATranslationDataset

    with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
        for i in range(5):
            f.write(json.dumps({"ipa": f"ipa_{i}", "english": f"eng_{i}"}) + "\n")
        f.flush()

        # Minimal mock tokenizer
        class MockTokenizer:
            def __call__(self, text, **kwargs):
                import torch
                n = kwargs.get("max_length", 32)
                return {
                    "input_ids": torch.ones(1, n, dtype=torch.long),
                    "attention_mask": torch.ones(1, n, dtype=torch.long),
                }

        ds = IPATranslationDataset.from_jsonl(f.name, MockTokenizer(), max_len=32)
        assert len(ds) == 5
        item = ds[0]
        assert "input_ids" in item
        assert "attention_mask" in item
        assert "labels" in item


def test_dataset_label_masking():
    """Labels correctly mask the prompt portion with -100."""
    import torch

    from lfm.translator.dataset import IPATranslationDataset

    class SimpleTokenizer:
        """Tokenizer that returns position-based IDs for testing."""
        def __call__(self, text, **kwargs):
            n = kwargs.get("max_length", 32)
            # Simulate: each char is a token, padded to max_length
            ids = list(range(1, min(len(text) + 1, n + 1)))
            pad_len = n - len(ids)
            attn = [1] * len(ids) + [0] * pad_len
            ids = ids + [0] * pad_len
            return {
                "input_ids": torch.tensor([ids], dtype=torch.long),
                "attention_mask": torch.tensor([attn], dtype=torch.long),
            }

    ds = IPATranslationDataset(
        ["hello"], ["world"], SimpleTokenizer(), max_len=64
    )
    item = ds[0]
    labels = item["labels"]

    # The prompt portion should be -100
    prompt = "<ipa> hello </ipa> <eng>"
    prompt_len = min(len(prompt), 64)
    assert (labels[:prompt_len] == -100).all()

    # Padding should also be -100
    attn = item["attention_mask"]
    assert (labels[attn == 0] == -100).all()


# ── BLEU tests ────────────────────────────────────────────────────


def test_bleu_perfect():
    """Perfect match gives high BLEU scores."""
    from lfm.translator.evaluator import compute_bleu

    refs = ["the cat sat on the mat"]
    hyps = ["the cat sat on the mat"]
    bleu = compute_bleu(refs, hyps)
    assert bleu["bleu_1"] > 0.99
    assert bleu["bleu_4_geometric"] > 0.99


def test_bleu_empty():
    """Empty hypothesis gives zero BLEU."""
    from lfm.translator.evaluator import compute_bleu

    refs = ["the cat sat on the mat"]
    hyps = [""]
    bleu = compute_bleu(refs, hyps)
    assert bleu["bleu_1"] == 0.0
    assert bleu["bleu_4_geometric"] == 0.0


def test_bleu_partial():
    """Partial match gives intermediate BLEU."""
    from lfm.translator.evaluator import compute_bleu

    refs = ["the cat sat on the mat"]
    hyps = ["the cat sat"]
    bleu = compute_bleu(refs, hyps)
    assert 0.0 < bleu["bleu_1"] < 1.0
    # BLEU-1 should be higher than BLEU-4
    assert bleu["bleu_1"] > bleu["bleu_4_geometric"]

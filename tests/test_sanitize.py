"""Tests for the sanitization module."""

from __future__ import annotations

import pytest

from lfm.data.sanitize import (
    SanitizeConfig,
    _apply_number_policy,
    _apply_symbol_policy,
    _check_foreign_script_ratio,
    _check_repetition,
    sanitize_one,
    sanitize_samples,
)


class TestSanitizeOne:
    """Tests for sanitize_one()."""

    def _cfg(self, **overrides) -> SanitizeConfig:
        return SanitizeConfig(**overrides)

    def test_accepts_clean_text(self):
        cfg = self._cfg(require_terminal_punctuation=False)
        result = sanitize_one(("eng", "The quick brown fox jumps over the lazy dog"), cfg)
        assert result is not None
        assert result[0] == "eng"
        assert "quick" in result[1]

    def test_rejects_short_text(self):
        cfg = self._cfg(min_line_length=20)
        result = sanitize_one(("eng", "Too short"), cfg)
        assert result is None

    def test_rejects_low_alpha_ratio(self):
        cfg = self._cfg(alpha_ratio_min=0.7, require_terminal_punctuation=False)
        result = sanitize_one(("eng", "### $$$ @@@ !!! ???"), cfg)
        assert result is None

    def test_rejects_digits_with_reject_policy(self):
        cfg = self._cfg(number_policy="reject", require_terminal_punctuation=False)
        result = sanitize_one(("eng", "The 42 quick brown foxes jump over the lazy dog"), cfg)
        assert result is None

    def test_keeps_digits_with_keep_policy(self):
        cfg = self._cfg(number_policy="keep", max_digit_ratio=1.0, require_terminal_punctuation=False)
        result = sanitize_one(("eng", "The 42 quick brown foxes jump over the lazy dog"), cfg)
        assert result is not None
        assert "42" in result[1]

    def test_strips_digits_with_strip_policy(self):
        cfg = self._cfg(number_policy="strip", require_terminal_punctuation=False)
        result = sanitize_one(("eng", "The 42 quick brown foxes jump over the lazy dog"), cfg)
        assert result is not None
        assert "42" not in result[1]

    def test_rejects_too_few_words(self):
        cfg = self._cfg(min_word_count=3, min_line_length=5, require_terminal_punctuation=False)
        result = sanitize_one(("eng", "Just two"), cfg)
        assert result is None

    def test_terminal_punctuation_required(self):
        cfg = self._cfg(require_terminal_punctuation=True)
        result = sanitize_one(("eng", "This sentence has no punctuation at the end"), cfg)
        assert result is None

    def test_terminal_punctuation_accepted(self):
        cfg = self._cfg(require_terminal_punctuation=True)
        result = sanitize_one(("eng", "This sentence ends properly."), cfg)
        assert result is not None

    def test_strips_urls(self):
        cfg = self._cfg(strip_urls=True, require_terminal_punctuation=False)
        result = sanitize_one(
            ("eng", "Visit the website at the location for more information about the world"),
            cfg,
        )
        assert result is not None

    def test_max_line_length_truncation(self):
        cfg = self._cfg(max_line_length=50, require_terminal_punctuation=False)
        long_text = "a " * 100  # 200 chars
        result = sanitize_one(("eng", long_text), cfg)
        if result is not None:
            assert len(result[1]) <= 50


class TestNumberPolicy:
    def test_reject_with_digits(self):
        assert _apply_number_policy("hello 42 world", "reject", "en") is None

    def test_reject_no_digits(self):
        assert _apply_number_policy("hello world", "reject", "en") == "hello world"

    def test_keep(self):
        assert _apply_number_policy("hello 42 world", "keep", "en") == "hello 42 world"

    def test_strip(self):
        result = _apply_number_policy("hello 42 world", "strip", "en")
        assert result is not None
        assert "42" not in result
        assert "hello" in result


class TestSymbolPolicy:
    def test_reject_with_symbols(self):
        assert _apply_symbol_policy("the α value", "reject") is None

    def test_keep(self):
        assert _apply_symbol_policy("the α value", "keep") == "the α value"

    def test_strip(self):
        result = _apply_symbol_policy("the α value", "strip")
        assert result is not None
        assert "α" not in result

    def test_spell_out(self):
        result = _apply_symbol_policy("the α value", "spell_out")
        assert result is not None
        assert "alpha" in result

    def test_no_symbols_passthrough(self):
        assert _apply_symbol_policy("hello world", "reject") == "hello world"


class TestForeignScriptRatio:
    def test_english_passes(self):
        assert _check_foreign_script_ratio("The quick brown fox", "eng", 0.3)

    def test_unknown_language_passes(self):
        assert _check_foreign_script_ratio("anything", "xyz", 0.3)

    def test_pure_foreign_fails(self):
        # Cyrillic text labeled as English
        assert not _check_foreign_script_ratio("Привет мир", "eng", 0.3)


class TestRepetition:
    def test_normal_text_passes(self):
        assert _check_repetition("the quick brown fox jumps over the lazy dog", 0.5, 0.4)

    def test_highly_repetitive_fails(self):
        assert not _check_repetition("the the the the the the the the", 0.5, 0.4)

    def test_short_text_passes(self):
        assert _check_repetition("hi hi", 0.5, 0.4)


class TestSanitizeSamples:
    def test_filters_batch(self):
        cfg = SanitizeConfig(require_terminal_punctuation=False)
        samples = [
            ("eng", "The quick brown fox jumps over the lazy dog near the forest"),
            ("eng", "x"),  # too short
            ("eng", "Another valid sentence about something important in the world"),
        ]
        result = sanitize_samples(samples, cfg, num_workers=1)
        assert len(result) == 2

    def test_empty_input(self):
        result = sanitize_samples([], SanitizeConfig(), num_workers=1)
        assert result == []

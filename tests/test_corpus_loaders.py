"""Tests for the modular corpus loader system."""

from __future__ import annotations

from pathlib import Path

import pytest

from lfm.data.loaders.leipzig import (
    LeipzigCorpusConfig,
    LeipzigCorpusLoader,
    _extract_lang_code,
)

# -- Language code extraction ------------------------------------------------

def test_extract_lang_code_standard():
    """Standard Leipzig filename: eng_news_2023_100K-sentences."""
    assert _extract_lang_code("eng_news_2023_100K-sentences") == "eng"


def test_extract_lang_code_no_suffix():
    """Filename without -sentences suffix."""
    assert _extract_lang_code("tur_web_2020_300K") == "tur"


def test_extract_lang_code_invalid():
    """Non-matching filename returns None."""
    assert _extract_lang_code("README") is None
    assert _extract_lang_code("12_foo_bar") is None


# -- Leipzig loader ----------------------------------------------------------

@pytest.fixture()
def leipzig_dir(tmp_path: Path) -> Path:
    """Create a temporary Leipzig-format data directory."""
    # English sentences
    eng_file = tmp_path / "eng_news_2023_100K-sentences.txt"
    eng_file.write_text(
        "1\tThis is an English sentence.\n"
        "2\tAnother English sentence for testing.\n"
        "3\tShort.\n"  # will be filtered by min_line_length
        "4\tA third English sentence with more words.\n"
    )

    # Turkish sentences
    tur_file = tmp_path / "tur_web_2022_100K-sentences.txt"
    tur_file.write_text(
        "1\tBu bir Türkçe cümledir.\n"
        "2\tBaşka bir Türkçe cümle.\n"
    )

    # Nested layout: deu_news_2023_100K/deu_news_2023_100K-sentences.txt
    deu_dir = tmp_path / "deu_news_2023_100K"
    deu_dir.mkdir()
    deu_file = deu_dir / "deu_news_2023_100K-sentences.txt"
    deu_file.write_text(
        "1\tDies ist ein deutscher Satz.\n"
        "2\tNoch ein deutscher Satz zum Testen.\n"
    )

    return tmp_path


def test_leipzig_loader_basic(leipzig_dir: Path):
    """Leipzig loader finds and parses all sentence files."""
    cfg = LeipzigCorpusConfig(data_dir=str(leipzig_dir))
    loader = LeipzigCorpusLoader(cfg)
    samples = loader.load()

    # Should have eng + tur + deu sentences
    langs = {lang for lang, _ in samples}
    assert "eng" in langs
    assert "tur" in langs
    assert "deu" in langs


def test_leipzig_loader_language_filter(leipzig_dir: Path):
    """Leipzig loader filters by language code."""
    cfg = LeipzigCorpusConfig(
        data_dir=str(leipzig_dir),
        languages=["tur"],
    )
    loader = LeipzigCorpusLoader(cfg)
    samples = loader.load()

    langs = {lang for lang, _ in samples}
    assert langs == {"tur"}


def test_leipzig_loader_min_line_length(leipzig_dir: Path):
    """Short lines are filtered out."""
    cfg = LeipzigCorpusConfig(
        data_dir=str(leipzig_dir),
        min_line_length=15,  # "Short." is only 6 chars
    )
    loader = LeipzigCorpusLoader(cfg)
    samples = loader.load()

    texts = [text for _, text in samples if text == "Short."]
    assert len(texts) == 0


def test_leipzig_loader_per_language_cap(leipzig_dir: Path):
    """Per-language sample cap is respected."""
    cfg = LeipzigCorpusConfig(
        data_dir=str(leipzig_dir),
        max_samples_per_language=1,
    )
    loader = LeipzigCorpusLoader(cfg)
    samples = loader.load()

    # Each language should have at most 1 sample
    from collections import Counter
    counts = Counter(lang for lang, _ in samples)
    for count in counts.values():
        assert count <= 1


def test_leipzig_loader_missing_dir():
    """FileNotFoundError when data_dir doesn't exist."""
    cfg = LeipzigCorpusConfig(data_dir="/nonexistent/path")
    loader = LeipzigCorpusLoader(cfg)
    with pytest.raises(FileNotFoundError):
        loader.load()


def test_leipzig_loader_nested_layout(leipzig_dir: Path):
    """Nested directory layout is discovered."""
    cfg = LeipzigCorpusConfig(
        data_dir=str(leipzig_dir),
        languages=["deu"],
    )
    loader = LeipzigCorpusLoader(cfg)
    samples = loader.load()

    assert len(samples) > 0
    assert all(lang == "deu" for lang, _ in samples)


# -- Registry integration ---------------------------------------------------

def test_leipzig_registered():
    """LeipzigCorpusLoader is registered in the global registry."""
    from lfm import list_registered
    from lfm.faculty.config import FacultyConfig
    from lfm.faculty.model import LanguageFaculty

    # Force registry population
    LanguageFaculty(FacultyConfig(phonology=None))
    assert "leipzig" in list_registered("corpus_loader")

"""Abstract base for corpus loaders.

Defines the ``CorpusLoader`` protocol, ``CorpusLoaderBase`` ABC, and
``CorpusLoaderConfig`` base config for the modular corpus loading system.
Concrete loaders (Leipzig, OPUS, etc.) register via
``@register("corpus_loader", name)`` and produce balanced
``(language_code, text_line)`` tuples.
"""

from __future__ import annotations

from typing import NamedTuple, Protocol

from lfm.config.base import LFMBaseConfig


class RawSample(NamedTuple):
    """A single raw text sample with full provenance metadata.

    Used by the dataset generation pipeline to track where each sample
    came from through the entire load → sanitize → IPA → HDF5 pipeline.
    """

    language: str       # ISO 639-3
    text: str           # Raw text
    source: str         # Corpus source name (e.g. "leipzig")
    source_file: str    # Original filename (e.g. "eng_news_2023_100K-sentences.txt")


class CorpusLoaderConfig(LFMBaseConfig):
    """Base configuration for corpus loaders.

    Attributes:
        name: Registry name of the corpus loader implementation.
        max_samples_per_language: Per-language sample cap for typological
            balance.  Languages with more samples are randomly subsampled.
        min_line_length: Minimum character count per line (shorter lines
            are skipped).
        max_line_length: Maximum character count per line (longer lines
            are truncated).
    """

    name: str
    max_samples_per_language: int = 50000
    min_line_length: int = 10
    max_line_length: int = 500


class CorpusLoader(Protocol):
    """Structural type for corpus loaders.

    All corpus loaders must implement ``load()`` which returns a balanced
    list of ``(language_code, text_line)`` tuples.  Language codes should
    be ISO 639-3 (3-letter) where possible.
    """

    def load(self) -> list[tuple[str, str]]:
        """Load and balance corpus data.

        Returns:
            List of ``(language_code, text_line)`` tuples, balanced
            across languages according to the loader's configuration.
        """
        ...


class CorpusLoaderBase:
    """Base class for corpus loaders with ``load_detailed()`` support.

    Provides a default ``load_detailed()`` that wraps ``load()`` output
    into ``RawSample`` namedtuples.  Subclasses can override to provide
    richer metadata (e.g. source filenames).
    """

    config: CorpusLoaderConfig

    def load(self) -> list[tuple[str, str]]:
        """Load and balance corpus data (abstract — subclasses must implement)."""
        raise NotImplementedError

    def load_detailed(self) -> list[RawSample]:
        """Load corpus data with full provenance metadata.

        Default implementation wraps ``load()`` output with the config
        name as source and empty source_file.  Override in subclasses
        to preserve actual filenames.

        Returns:
            List of ``RawSample`` namedtuples.
        """
        return [
            RawSample(lang, text, self.config.name, "")
            for lang, text in self.load()
        ]

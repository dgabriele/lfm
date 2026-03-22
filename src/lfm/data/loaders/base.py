"""Abstract base for corpus loaders.

Defines the ``CorpusLoader`` protocol and ``CorpusLoaderConfig`` base config
for the modular corpus loading system.  Concrete loaders (Leipzig, OPUS, etc.)
register via ``@register("corpus_loader", name)`` and produce balanced
``(language_code, text_line)`` tuples.
"""

from __future__ import annotations

from typing import Protocol

from lfm.config.base import LFMBaseConfig


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

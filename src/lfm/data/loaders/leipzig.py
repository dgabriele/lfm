"""Leipzig Corpora Collection loader.

Parses the Leipzig Corpora Collection sentence file format (tab-separated
``sentence_id<TAB>sentence_text``) with automatic language code extraction
from filenames, per-language sample caps for typological balance, and support
for both flat and nested directory layouts.
"""

from __future__ import annotations

import logging
import random
from collections import defaultdict
from pathlib import Path

from lfm._registry import register
from lfm.data.loaders.base import CorpusLoaderConfig

logger = logging.getLogger(__name__)


class LeipzigCorpusConfig(CorpusLoaderConfig):
    """Configuration for the Leipzig Corpora Collection loader.

    Attributes:
        name: Registry name (``"leipzig"``).
        data_dir: Path to directory containing Leipzig sentence files.
            Supports both flat layouts (all files in one directory) and
            nested layouts (``{name}/{name}-sentences.txt``).
        languages: ISO 639-3 language codes to include.  Empty list means
            include all languages found in ``data_dir``.
        seed: Random seed for reproducible subsampling.
    """

    name: str = "leipzig"
    data_dir: str = "data/leipzig"
    languages: list[str] = []
    seed: int = 42


@register("corpus_loader", "leipzig")
class LeipzigCorpusLoader:
    """Load and balance data from the Leipzig Corpora Collection.

    Leipzig sentence files use tab-separated format::

        1	This is the first sentence.
        2	This is the second sentence.

    Language codes are extracted from filenames, which follow the convention
    ``{lang}_{source}_{year}_{size}-sentences.txt`` (e.g.,
    ``eng_news_2023_100K-sentences.txt``).

    Args:
        config: Leipzig loader configuration.
    """

    def __init__(self, config: LeipzigCorpusConfig) -> None:
        self.config = config

    def load(self) -> list[tuple[str, str]]:
        """Load Leipzig data with per-language balancing.

        Returns:
            List of ``(language_code, text_line)`` tuples.

        Raises:
            FileNotFoundError: If ``data_dir`` does not exist.
        """
        cfg = self.config
        data_dir = Path(cfg.data_dir)

        if not data_dir.is_dir():
            raise FileNotFoundError(
                f"Leipzig data directory not found: {data_dir}"
            )

        lang_filter = set(cfg.languages) if cfg.languages else None

        # Discover sentence files
        sentence_files = self._discover_files(data_dir)
        if not sentence_files:
            logger.warning("No Leipzig sentence files found in %s", data_dir)
            return []

        # Load per-language, applying caps
        per_lang: dict[str, list[str]] = defaultdict(list)

        for lang_code, filepath in sentence_files:
            if lang_filter is not None and lang_code not in lang_filter:
                continue

            lines = self._parse_sentence_file(filepath)
            # Apply length filters
            lines = [
                ln[:cfg.max_line_length]
                for ln in lines
                if len(ln) >= cfg.min_line_length
            ]
            per_lang[lang_code].extend(lines)

        # Apply per-language cap with random subsampling
        rng = random.Random(cfg.seed)
        samples: list[tuple[str, str]] = []

        for lang_code, lines in sorted(per_lang.items()):
            if len(lines) > cfg.max_samples_per_language:
                lines = rng.sample(lines, cfg.max_samples_per_language)
            for line in lines:
                samples.append((lang_code, line))

        logger.info(
            "Loaded %d samples from %d languages",
            len(samples),
            len(per_lang),
        )
        return samples

    @staticmethod
    def _discover_files(data_dir: Path) -> list[tuple[str, Path]]:
        """Discover Leipzig sentence files and extract language codes.

        Handles both flat layouts (all ``*-sentences.txt`` in one dir) and
        nested layouts (``{name}/{name}-sentences.txt``).

        Returns:
            List of ``(language_code, filepath)`` tuples.
        """
        results: list[tuple[str, Path]] = []

        # Flat: data_dir/*-sentences.txt
        for f in sorted(data_dir.glob("*-sentences.txt")):
            lang = _extract_lang_code(f.stem)
            if lang:
                results.append((lang, f))

        # Nested: data_dir/{name}/{name}-sentences.txt
        for subdir in sorted(data_dir.iterdir()):
            if not subdir.is_dir():
                continue
            for f in sorted(subdir.glob("*-sentences.txt")):
                lang = _extract_lang_code(f.stem)
                if lang and (lang, f) not in results:
                    results.append((lang, f))

        return results

    @staticmethod
    def _parse_sentence_file(filepath: Path) -> list[str]:
        """Parse a Leipzig tab-separated sentence file.

        Args:
            filepath: Path to the sentence file.

        Returns:
            List of sentence text strings (IDs stripped).
        """
        lines: list[str] = []
        with open(filepath, encoding="utf-8", errors="replace") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                # Tab-separated: sentence_id <TAB> sentence_text
                parts = line.split("\t", maxsplit=1)
                if len(parts) == 2:
                    lines.append(parts[1])
                elif not parts[0][0].isdigit():
                    # No ID prefix — treat entire line as text
                    lines.append(parts[0])
        return lines


def _extract_lang_code(stem: str) -> str | None:
    """Extract ISO 639-3 language code from a Leipzig filename stem.

    Leipzig filenames follow ``{lang}_{source}_{year}_{size}-sentences``
    where ``{lang}`` is a 3-letter ISO 639-3 code.

    Args:
        stem: Filename stem (without extension), e.g.
            ``"eng_news_2023_100K-sentences"``.

    Returns:
        3-letter language code, or ``None`` if extraction fails.
    """
    # Remove the "-sentences" suffix if present
    if stem.endswith("-sentences"):
        stem = stem[: -len("-sentences")]

    # First segment before underscore should be the lang code
    parts = stem.split("_")
    if parts and len(parts[0]) == 3 and parts[0].isalpha():
        return parts[0].lower()

    return None

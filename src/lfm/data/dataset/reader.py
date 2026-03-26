"""Dataset reader: load pre-generated HDF5 datasets for pretraining.

Provides ``DatasetReader`` which reads HDF5 + manifest and returns
data in formats compatible with the pretraining pipeline.
"""

from __future__ import annotations

import logging
from collections.abc import Iterator
from pathlib import Path

from lfm.data.dataset.config import ProcessedSample
from lfm.data.dataset.manifest import DatasetManifest

logger = logging.getLogger(__name__)


class DatasetReader:
    """Read pre-generated HDF5 datasets.

    Loads ``samples.h5`` and ``manifest.yaml`` from a dataset directory
    and provides convenient access patterns for pretraining integration.

    Args:
        dataset_dir: Path to the dataset directory (containing
            ``samples.h5`` and ``manifest.yaml``).
    """

    def __init__(self, dataset_dir: str | Path) -> None:
        self._dir = Path(dataset_dir)
        self._manifest: DatasetManifest | None = None

        if not self._dir.is_dir():
            raise FileNotFoundError(f"Dataset directory not found: {self._dir}")

        h5_path = self._dir / "samples.h5"
        if not h5_path.is_file():
            raise FileNotFoundError(f"samples.h5 not found in {self._dir}")

    @property
    def manifest(self) -> DatasetManifest:
        """Load and cache the dataset manifest."""
        if self._manifest is None:
            manifest_path = self._dir / "manifest.yaml"
            if not manifest_path.is_file():
                raise FileNotFoundError(f"manifest.yaml not found in {self._dir}")
            self._manifest = DatasetManifest.load(manifest_path)
        return self._manifest

    def load_ipa_tuples(
        self,
        languages: list[str] | None = None,
        max_samples_per_language: int | None = None,
    ) -> list[tuple[str, str]]:
        """Load ``(lang, ipa_text)`` tuples — drop-in for pretrain pipeline.

        Args:
            languages: Filter to these ISO 639-3 codes. ``None`` = all.
            max_samples_per_language: Per-language cap. ``None`` = no cap.

        Returns:
            List of ``(language_code, ipa_text)`` tuples.
        """
        import random
        from collections import defaultdict

        import h5py

        h5_path = self._dir / "samples.h5"
        lang_filter = set(languages) if languages else None

        with h5py.File(h5_path, "r") as f:
            grp = f["samples"]
            langs = [x.decode("utf-8") if isinstance(x, bytes) else x for x in grp["language"][:]]
            ipas = [x.decode("utf-8") if isinstance(x, bytes) else x for x in grp["ipa"][:]]

        # Group by language
        by_lang: dict[str, list[str]] = defaultdict(list)
        for lang, ipa in zip(langs, ipas):
            if lang_filter is not None and lang not in lang_filter:
                continue
            by_lang[lang].append(ipa)

        # Apply per-language cap
        result: list[tuple[str, str]] = []
        rng = random.Random(42)

        for lang in sorted(by_lang.keys()):
            items = by_lang[lang]
            if max_samples_per_language is not None and len(items) > max_samples_per_language:
                items = rng.sample(items, max_samples_per_language)
            for ipa in items:
                result.append((lang, ipa))

        logger.info(
            "Loaded %d IPA tuples from %s (%d languages)",
            len(result), self._dir.name, len(by_lang),
        )
        return result

    def iter_samples(self) -> Iterator[ProcessedSample]:
        """Streaming iterator over all samples.

        Yields ``ProcessedSample`` dicts one at a time, suitable for
        large datasets that don't fit in memory.
        """
        import h5py

        h5_path = self._dir / "samples.h5"

        with h5py.File(h5_path, "r") as f:
            grp = f["samples"]
            n = len(grp["seq"])

            for i in range(n):
                yield ProcessedSample(
                    seq=int(grp["seq"][i]),
                    language=_decode(grp["language"][i]),
                    source=_decode(grp["source"][i]),
                    source_file=_decode(grp["source_file"][i]),
                    raw=_decode(grp["raw"][i]),
                    ipa=_decode(grp["ipa"][i]),
                    ipa_length=int(grp["ipa_length"][i]),
                )

    def languages(self) -> list[str]:
        """Return sorted list of language codes in the dataset."""
        return sorted(self.manifest.languages.keys())

    def __len__(self) -> int:
        """Return total number of samples."""
        return self.manifest.total_samples

    def __repr__(self) -> str:
        try:
            m = self.manifest
            return (
                f"DatasetReader({self._dir.name!r}, "
                f"{m.total_samples} samples, "
                f"{len(m.languages)} languages)"
            )
        except FileNotFoundError:
            return f"DatasetReader({self._dir!r})"


def _decode(value: bytes | str) -> str:
    """Decode HDF5 byte strings to str."""
    return value.decode("utf-8") if isinstance(value, bytes) else value

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

    def has_constituents(self) -> bool:
        """Check if this dataset has phase 2 constituent fields."""
        import h5py

        with h5py.File(self._dir / "samples.h5", "r") as f:
            return "parent_seq" in f["samples"]

    def load_constituent_tuples(
        self,
        languages: list[str] | None = None,
        max_per_language: int | None = None,
        balance_by_length: bool = True,
        length_buckets: tuple[int, ...] = (20, 50),
        seed: int = 42,
    ) -> tuple[list[tuple[str, str]], list[tuple[str, str, int, str]]]:
        """Load phase 2 constituent data: full sentences + paired constituents.

        Args:
            languages: Filter to these ISO 639-3 codes. None = all.
            max_per_language: Max constituents per language. None = no cap.
            balance_by_length: If True, subsample to equalize across
                length buckets (short/medium/long) within each language.
            length_buckets: Character-length boundaries for buckets.
                Default (20, 50) gives: short(<20), medium(20-50), long(>50).
            seed: Random seed for reproducible subsampling.

        Returns two lists:
          - sentences: list of (lang, ipa) for all full sentences (label="S")
          - constituents: list of (lang, ipa, parent_seq, label) for
            extracted constituents, balanced across languages and lengths.
        """
        import random
        from collections import defaultdict

        import h5py

        h5_path = self._dir / "samples.h5"
        lang_filter = set(languages) if languages else None

        with h5py.File(h5_path, "r") as f:
            grp = f["samples"]
            langs = [_decode(x) for x in grp["language"][:]]
            ipas = [_decode(x) for x in grp["ipa"][:]]
            parent_seqs = grp["parent_seq"][:].tolist()
            labels = [_decode(x) for x in grp["constituent_label"][:]]

        sentences: list[tuple[str, str]] = []
        constituents_raw: list[tuple[str, str, int, str]] = []
        seq_to_sent_idx: dict[int, int] = {}

        for i, (lang, ipa, pseq, label) in enumerate(
            zip(langs, ipas, parent_seqs, labels),
        ):
            if lang_filter is not None and lang not in lang_filter:
                continue
            if label == "S" or pseq == -1:
                seq_to_sent_idx[i] = len(sentences)
                sentences.append((lang, ipa))
            else:
                constituents_raw.append((lang, ipa, pseq, label))

        # Remap parent_seq from global index to sentences list index
        remapped: list[tuple[str, str, int, str]] = []
        for lang, ipa, pseq, label in constituents_raw:
            sent_idx = seq_to_sent_idx.get(pseq, -1)
            if sent_idx >= 0:
                remapped.append((lang, ipa, sent_idx, label))

        # Balanced subsampling: across languages and length buckets
        rng = random.Random(seed)

        if max_per_language is not None or balance_by_length:
            def _bucket(ipa: str) -> int:
                length = len(ipa)
                for bi, boundary in enumerate(length_buckets):
                    if length < boundary:
                        return bi
                return len(length_buckets)

            # Group by (language, length_bucket)
            by_lang_bucket: dict[tuple[str, int], list[tuple[str, str, int, str]]] = defaultdict(list)
            for item in remapped:
                lang = item[0]
                bkt = _bucket(item[1])
                by_lang_bucket[(lang, bkt)].append(item)

            # Determine per-bucket cap
            if balance_by_length and max_per_language is not None:
                n_buckets = len(length_buckets) + 1
                per_bucket = max_per_language // n_buckets
            elif max_per_language is not None:
                per_bucket = max_per_language
            else:
                # balance_by_length without max_per_language:
                # equalize to the smallest bucket per language
                per_lang_min: dict[str, int] = {}
                all_langs = set(lang for lang, _ in by_lang_bucket.keys())
                for lang in all_langs:
                    bucket_sizes = [
                        len(by_lang_bucket.get((lang, bi), []))
                        for bi in range(len(length_buckets) + 1)
                    ]
                    non_empty = [s for s in bucket_sizes if s > 0]
                    per_lang_min[lang] = min(non_empty) if non_empty else 0
                per_bucket = None  # per-language min used below

            balanced: list[tuple[str, str, int, str]] = []
            for (lang, bkt), items in sorted(by_lang_bucket.items()):
                cap = per_bucket if per_bucket is not None else per_lang_min.get(lang, len(items))
                if len(items) > cap:
                    items = rng.sample(items, cap)
                balanced.extend(items)

            rng.shuffle(balanced)
            remapped = balanced

        logger.info(
            "Loaded %d sentences + %d constituents from %s",
            len(sentences), len(remapped), self._dir.name,
        )
        return sentences, remapped

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

"""Dataset generator: load → sanitize → LLM gate → IPA → balance → HDF5.

Orchestrates the full pipeline from raw corpus text to a pre-generated
HDF5 dataset ready for pretraining consumption.
"""

from __future__ import annotations

import importlib
import logging
import random
from collections import Counter, defaultdict
from pathlib import Path

from lfm._registry import create
from lfm.data.dataset.config import DatasetGenerateConfig, ProcessedSample
from lfm.data.dataset.llm_gate import LLMGatekeeper
from lfm.data.dataset.manifest import DatasetManifest
from lfm.data.loaders.base import CorpusLoaderConfig, RawSample
from lfm.data.sanitize import _init_worker, _sanitize_one_worker

logger = logging.getLogger(__name__)

# Source descriptions for manifest
_SOURCE_DESCRIPTIONS: dict[str, str] = {
    "leipzig": "Leipzig Corpora Collection — news text from 16 languages",
}


class DatasetGenerator:
    """Generate HDF5 datasets from corpus sources.

    Pipeline stages:
        1. Load raw samples via corpus loader (with provenance)
        2. Sanitize via configurable rule-based filters
        3. LLM quality gate on sanitized text (optional)
        4. Convert to IPA transcription
        5. Balance per-language sample counts
        6. Write HDF5 + rejected.h5 + manifest.yaml
    """

    def __init__(self, config: DatasetGenerateConfig) -> None:
        self.config = config
        self._output_dir = Path(config.output or f"data/datasets/{config.source}")

    def generate(self) -> Path:
        """Run the full generation pipeline.

        Returns:
            Path to the output directory containing samples.h5 and manifest.yaml.
        """
        cfg = self.config

        # 1. Load
        raw_samples = self._load()
        logger.info("Loaded %d raw samples", len(raw_samples))

        # 2. Sanitize
        sanitized, rejected_sanitize = self._sanitize(raw_samples)
        logger.info(
            "Sanitized: %d accepted, %d rejected",
            len(sanitized), len(rejected_sanitize),
        )

        # 3. LLM gate (optional, on sanitized raw text)
        rejected_gate: list[tuple[str, str, str]] = []
        if cfg.llm_gate.enabled:
            sanitized, rejected_gate = self._llm_gate(sanitized)
            logger.info(
                "LLM gate: %d accepted, %d rejected",
                len(sanitized), len(rejected_gate),
            )

        # 3b. Constituency extraction (optional — augments with sub-phrases)
        if cfg.extract_constituents:
            sanitized = self._extract_constituents(sanitized)
            logger.info(
                "After constituency extraction: %d samples",
                len(sanitized),
            )

        # 4. IPA conversion
        processed = self._convert_ipa(sanitized, raw_samples)
        logger.info("IPA conversion: %d processed samples", len(processed))

        # 5. Balance
        processed = self._balance(processed)
        logger.info("After balancing: %d samples", len(processed))

        # 6. Write
        self._write_h5(processed, rejected_sanitize, rejected_gate)

        return self._output_dir

    def _load(self) -> list[RawSample]:
        """Load raw samples from the configured corpus source."""
        cfg = self.config

        # Registry-based loader lookup
        from lfm.data.loaders.leipzig import LeipzigCorpusConfig

        loader_registry: dict[str, tuple[str, type[CorpusLoaderConfig]]] = {
            "leipzig": ("lfm.data.loaders.leipzig", LeipzigCorpusConfig),
        }

        if cfg.source not in loader_registry:
            raise KeyError(
                f"Unknown source {cfg.source!r}. "
                f"Available: {sorted(loader_registry.keys())}"
            )

        mod_path, config_cls = loader_registry[cfg.source]
        importlib.import_module(mod_path)

        # Build loader config: merge source_config with our settings
        loader_kwargs: dict = {
            "name": cfg.source,
            "max_samples_per_language": cfg.max_samples,
            **cfg.source_config,
        }
        if cfg.languages:
            loader_kwargs["languages"] = cfg.languages

        loader_cfg = config_cls(**loader_kwargs)
        loader = create("corpus_loader", cfg.source, loader_cfg)

        # Use load_detailed() if available, else wrap load()
        if hasattr(loader, "load_detailed"):
            return loader.load_detailed()
        return [
            RawSample(lang, text, cfg.source, "")
            for lang, text in loader.load()
        ]

    def _extract_constituents(
        self,
        sanitized: list[tuple[RawSample, str]],
    ) -> list[tuple[RawSample, str]]:
        """Augment samples with phrase constituents via constituency parsing.

        Runs per-language parsing in parallel (one Stanza pipeline per
        worker process on CPU).  Each constituent becomes a separate
        training sample alongside the full sentence.
        """
        from lfm.data.constituents import extract_constituents_parallel

        cfg = self.config

        # Build (lang, text) pairs for the parallel extractor
        pairs = [(raw.language, text) for raw, text in sanitized]
        all_results = extract_constituents_parallel(
            pairs,
            min_length=cfg.min_constituent_length,
            num_workers=cfg.num_workers,
        )

        # Map results back to RawSample tuples.
        # The first len(sanitized) results are the original sentences (label="S").
        # Additional results are extracted constituents.
        augmented: list[tuple[RawSample, str]] = []

        # Build a lookup for original RawSamples
        raw_by_lang_text: dict[tuple[str, str], RawSample] = {}
        for raw, text in sanitized:
            raw_by_lang_text[(raw.language, text)] = raw

        for lang, text, label in all_results:
            existing = raw_by_lang_text.get((lang, text))
            if existing is not None:
                augmented.append((existing, text))
            else:
                # Constituent — create a new RawSample
                # Find any RawSample for this language for source metadata
                source = "unknown"
                source_file = ""
                for raw, _ in sanitized:
                    if raw.language == lang:
                        source = raw.source
                        source_file = raw.source_file
                        break
                aug_raw = RawSample(lang, text, source, source_file)
                augmented.append((aug_raw, text))

        return augmented

    def _sanitize(
        self, samples: list[RawSample],
    ) -> tuple[list[tuple[RawSample, str]], list[tuple[str, str, str]]]:
        """Sanitize samples with rejection tracking.

        Uses multiprocessing via ``sanitize_samples``, then aligns
        results back to original ``RawSample`` metadata in O(n).

        Returns:
            Tuple of:
              - accepted: list of (original_raw_sample, sanitized_text) pairs
              - rejected: list of (language, text, reason) tuples
        """
        import multiprocessing as mp
        import os

        cfg = self.config.sanitize
        tuples = [(s.language, s.text) for s in samples]
        num_workers = max(1, int(os.cpu_count() * 0.9))
        with mp.Pool(num_workers, initializer=_init_worker, initargs=(cfg,)) as pool:
            results = pool.map(_sanitize_one_worker, tuples, chunksize=1000)

        accepted: list[tuple[RawSample, str]] = []
        rejected_tuples: list[tuple[str, str, str]] = []

        for raw, result in zip(samples, results):
            if result is not None:
                accepted.append((raw, result[1]))
            else:
                rejected_tuples.append((raw.language, raw.text, "sanitize_filter"))

        return accepted, rejected_tuples

    def _llm_gate(
        self, samples: list[tuple[RawSample, str]],
    ) -> tuple[list[tuple[RawSample, str]], list[tuple[str, str, str]]]:
        """Run LLM quality gate on sanitized text.

        The LLM sees the sanitized natural language text (not IPA).
        """
        cfg = self.config.llm_gate
        gatekeeper = LLMGatekeeper(cfg)

        try:
            eval_input = [(raw.language, text) for raw, text in samples]
            results = gatekeeper.evaluate(eval_input)

            accepted: list[tuple[RawSample, str]] = []
            rejected: list[tuple[str, str, str]] = []

            for (raw, text), result in zip(samples, results):
                if result.verdict == "accept":
                    accepted.append((raw, text))
                elif result.verdict == "fix":
                    accepted.append((raw, result.text))
                else:
                    rejected.append((raw.language, text, f"llm_gate: {result.reason}"))

            return accepted, rejected
        finally:
            gatekeeper.unload()

    def _convert_ipa(
        self,
        sanitized: list[tuple[RawSample, str]],
        all_raw: list[RawSample],
    ) -> list[ProcessedSample]:
        """Convert sanitized text to IPA, preserving metadata."""
        from lfm.data.loaders.ipa import convert_corpus_to_ipa_labeled

        # Build (lang, sanitized_text) pairs for IPA conversion
        pairs = [(raw.language, text) for raw, text in sanitized]

        ipa_results = convert_corpus_to_ipa_labeled(
            pairs, num_workers=self.config.num_workers,
        )

        # Align IPA results back with metadata
        # convert_corpus_to_ipa_labeled drops failed samples, so we need
        # to track which succeeded
        processed: list[ProcessedSample] = []
        ipa_idx = 0

        for (raw, text), (lang, _) in zip(sanitized, pairs):
            if ipa_idx < len(ipa_results) and ipa_results[ipa_idx][0] == lang:
                _, ipa = ipa_results[ipa_idx]
                ipa_idx += 1

                # IPA quality check
                cfg = self.config.sanitize
                if len(ipa) < cfg.min_ipa_length:
                    continue
                non_alpha = sum(1 for c in ipa if not c.isalpha() and not c.isspace())
                if len(ipa) > 0 and non_alpha / len(ipa) > cfg.max_ipa_non_alpha_ratio:
                    continue

                processed.append(ProcessedSample(
                    seq=len(processed),
                    language=raw.language,
                    source=raw.source,
                    source_file=raw.source_file,
                    raw=raw.text,
                    ipa=ipa,
                    ipa_length=len(ipa),
                ))

        return processed

    def _balance(self, samples: list[ProcessedSample]) -> list[ProcessedSample]:
        """Apply per-language min/max balancing."""
        cfg = self.config
        rng = random.Random(cfg.seed)

        # Group by language
        by_lang: dict[str, list[ProcessedSample]] = defaultdict(list)
        for s in samples:
            by_lang[s["language"]].append(s)

        balanced: list[ProcessedSample] = []
        for lang in sorted(by_lang.keys()):
            lang_samples = by_lang[lang]

            # Drop languages below minimum
            if len(lang_samples) < cfg.min_samples:
                logger.info(
                    "Dropping %s: only %d samples (min=%d)",
                    lang, len(lang_samples), cfg.min_samples,
                )
                continue

            # Cap at max_samples
            if len(lang_samples) > cfg.max_samples:
                lang_samples = rng.sample(lang_samples, cfg.max_samples)

            balanced.extend(lang_samples)

        # Re-sequence
        for i, s in enumerate(balanced):
            s["seq"] = i

        return balanced

    def _write_h5(
        self,
        samples: list[ProcessedSample],
        rejected_sanitize: list[tuple[str, str, str]],
        rejected_gate: list[tuple[str, str, str]],
    ) -> None:
        """Write samples to HDF5 and manifest to YAML."""
        import h5py

        out = self._output_dir
        out.mkdir(parents=True, exist_ok=True)

        # Write samples.h5
        h5_path = out / "samples.h5"
        n = len(samples)

        if n == 0:
            logger.warning("No samples to write!")
            return

        str_dt = h5py.string_dtype()

        with h5py.File(h5_path, "w") as f:
            grp = f.create_group("samples")

            # Numeric datasets with gzip compression
            grp.create_dataset(
                "seq", data=[s["seq"] for s in samples],
                dtype="int64", compression="gzip", compression_opts=4,
            )
            grp.create_dataset(
                "ipa_length", data=[s["ipa_length"] for s in samples],
                dtype="int32", compression="gzip", compression_opts=4,
            )

            # String datasets with LZF compression
            for field in ("language", "source", "source_file", "raw", "ipa"):
                grp.create_dataset(
                    field,
                    data=[s[field].encode("utf-8") for s in samples],
                    dtype=str_dt,
                    compression="lzf",
                )

        logger.info("Wrote %d samples to %s", n, h5_path)

        # Write rejected.h5
        all_rejected = [
            *[(lang, text, reason) for lang, text, reason in rejected_sanitize],
            *rejected_gate,
        ]
        if all_rejected:
            rej_path = out / "rejected.h5"
            with h5py.File(rej_path, "w") as f:
                grp = f.create_group("rejected")
                grp.create_dataset(
                    "language",
                    data=[r[0].encode("utf-8") for r in all_rejected],
                    dtype=str_dt, compression="lzf",
                )
                grp.create_dataset(
                    "text",
                    data=[r[1].encode("utf-8") for r in all_rejected],
                    dtype=str_dt, compression="lzf",
                )
                grp.create_dataset(
                    "reason",
                    data=[r[2].encode("utf-8") for r in all_rejected],
                    dtype=str_dt, compression="lzf",
                )
            logger.info("Wrote %d rejected samples to %s", len(all_rejected), rej_path)

        # Language counts
        lang_counts = dict(Counter(s["language"] for s in samples).most_common())

        # Write manifest
        manifest = DatasetManifest.create(
            name=self.config.source,
            description=_SOURCE_DESCRIPTIONS.get(self.config.source, ""),
            sources=[self.config.source],
            languages=lang_counts,
            total_samples=n,
            rejected_samples=len(all_rejected),
            sanitize_config=self.config.sanitize.model_dump(),
            generate_config={
                "source": self.config.source,
                "max_samples": self.config.max_samples,
                "min_samples": self.config.min_samples,
                "seed": self.config.seed,
                "languages": self.config.languages,
            },
        )
        manifest.save(out / "manifest.yaml")

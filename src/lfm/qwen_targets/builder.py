"""End-to-end builder for a Qwen-latent :class:`EmbeddingStore`.

Composes :mod:`lfm.qwen_targets.corpora`, :mod:`extractor`,
:mod:`density`, and :mod:`cluster` into a single pipeline that turns
a mixed text corpus into a dialogue-game-compatible embedding store.

Pipeline stages:

  1. **Load corpora.**  A :class:`MixedCorpusLoader` interleaves
     examples from one or more :class:`CorpusSource` instances,
     respecting per-source weights and caps.
  2. **Extract hidden states.**  :class:`HiddenStateExtractor` pushes
     text through a frozen causal LM and returns unit-normalized
     pooled embeddings, streamed batch-by-batch.
  3. **Density-aware resampling.**  :class:`DensityReweighter`
     optionally resamples the population to correct for natural
     corpus density, interpolating between the raw distribution
     (temperature=0) and uniform on-manifold coverage (temperature=1).
  4. **Cluster.**  MiniBatchKMeans into
     ``config.cluster.num_clusters`` groups — matches the existing
     dialogue-game hard-negative sampling convention.
  5. **Save.**  Writes a drop-in-replacement
     :class:`~lfm.embeddings.store.EmbeddingStore`, including a
     ``metadata.json`` with full reproducibility info and a
     ``passages.jsonl`` listing the text and source provenance for
     every surviving example.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path

import numpy as np
import torch

from lfm.embeddings.store import EmbeddingStore
from lfm.qwen_targets.cluster import run_minibatch_kmeans
from lfm.qwen_targets.config import QwenTargetsConfig
from lfm.qwen_targets.corpora import (
    CorpusSource,
    CorpusText,
    HFStreamingCorpusSource,
    JSONLCorpusSource,
    MixedCorpusLoader,
    PlainTextCorpusSource,
)
from lfm.qwen_targets.density import DensityReweighter, ResamplingResult
from lfm.qwen_targets.extractor import HiddenStateExtractor

logger = logging.getLogger(__name__)


class QwenEmbeddingStoreBuilder:
    """Top-level orchestrator for Qwen-latent target construction.

    Usage::

        cfg = QwenTargetsConfig(...)
        builder = QwenEmbeddingStoreBuilder(cfg)
        store = builder.build()

    The returned :class:`EmbeddingStore` is already loaded and ready
    for immediate use by the dialogue game.  The same directory can
    be reopened by future runs via ``EmbeddingStore(path).load()``.

    Args:
        config: End-to-end configuration.
    """

    def __init__(self, config: QwenTargetsConfig) -> None:
        self.config = config

    # -- stage 1: corpora ---------------------------------------------------

    def _build_sources(self) -> list[CorpusSource]:
        """Instantiate :class:`CorpusSource` objects from config entries.

        Dispatches between local files, HF streaming, and HF prefetch
        caches based on which fields the config entry sets and whether
        a prefetched JSONL already exists on disk.  Preference order:

        1. Local ``path`` if set.
        2. Prefetch cache at ``prefetch_dir/<name>.jsonl`` if it exists
           (avoids network I/O on repeat runs).
        3. Live HuggingFace streaming as the fallback.

        Each entry must set exactly one of ``hf_dataset`` or ``path``.
        """
        from lfm.qwen_targets.prefetch import cache_path_for, is_cached

        sources: list[CorpusSource] = []
        prefetch_dir = Path(self.config.prefetch_dir)
        for entry in self.config.sources:
            if entry.hf_dataset is not None and entry.path is not None:
                raise ValueError(
                    f"Source '{entry.name}' sets both hf_dataset and path — "
                    "use exactly one.",
                )

            if entry.path is not None:
                path = Path(entry.path)
                if entry.text_field is not None and (
                    path.suffix.lower() in (".jsonl", ".json")
                    or path.name.endswith(".jsonl")
                ):
                    src: CorpusSource = JSONLCorpusSource(
                        path=path,
                        text_field=entry.text_field,
                        name=entry.name or path.stem,
                        max_samples=entry.max_samples,
                    )
                else:
                    src = PlainTextCorpusSource(
                        path=path,
                        name=entry.name or path.stem,
                        max_samples=entry.max_samples,
                    )
            elif entry.hf_dataset is not None:
                if is_cached(entry, prefetch_dir):
                    cache_path = cache_path_for(entry, prefetch_dir)
                    logger.info(
                        "Using prefetched cache for '%s': %s",
                        entry.name, cache_path,
                    )
                    src = JSONLCorpusSource(
                        path=cache_path,
                        text_field="text",
                        name=entry.name or cache_path.stem,
                        max_samples=entry.max_samples,
                    )
                else:
                    logger.info(
                        "No cache for '%s'; streaming live from HF "
                        "(consider running `lfm qwen-targets prefetch`)",
                        entry.name,
                    )
                    src = HFStreamingCorpusSource(
                        dataset_name=entry.hf_dataset,
                        config_name=entry.hf_config,
                        split=entry.hf_split,
                        text_field=entry.text_field or "text",
                        name=entry.name or entry.hf_dataset.replace("/", "-"),
                        max_samples=entry.max_samples,
                        min_length=entry.min_length,
                        max_length=entry.max_length,
                        trust_remote_code=entry.hf_trust_remote_code,
                    )
            else:
                raise ValueError(
                    f"Source '{entry.name}' must set either hf_dataset or path.",
                )
            sources.append(src)
        return sources

    def _build_mixer(self) -> MixedCorpusLoader:
        sources = self._build_sources()
        weights = [entry.weight for entry in self.config.sources]
        return MixedCorpusLoader(
            sources=sources,
            weights=weights,
            total_limit=self.config.max_extracted,
            seed=self.config.shuffle_seed,
        )

    # -- stage 2: extraction ------------------------------------------------

    def _extract(
        self, mixer: MixedCorpusLoader,
    ) -> tuple[np.ndarray, list[CorpusText]]:
        """Run the mixed corpus through the extractor.

        Returns the full (N, D) embedding matrix as a CPU float32
        ndarray, and the matching list of :class:`CorpusText` entries
        (same order).
        """
        device = torch.device(
            self.config.device if torch.cuda.is_available() else "cpu",
        )
        extractor = HiddenStateExtractor(self.config.extractor, device=device)

        # We iterate the mixer twice in parallel: once producing the
        # text stream for the extractor, once accumulating the
        # provenance records.  To avoid running the corpus twice we
        # cache records as we go in a side buffer.
        records: list[CorpusText] = []

        def text_iter():
            for rec in mixer:
                records.append(rec)
                yield rec.text

        logger.info("Extracting hidden states via %s", self.config.extractor.model_name)
        start = time.time()
        all_embs: list[torch.Tensor] = []
        for batch in extractor.encode_stream(text_iter()):
            all_embs.append(batch)
        if not all_embs:
            raise RuntimeError("No embeddings extracted — check corpora")
        embeddings = torch.cat(all_embs, dim=0).numpy().astype(np.float32)
        elapsed = time.time() - start
        logger.info(
            "Extracted %d embeddings in %.1fs (%.0f texts/sec), shape=%s",
            embeddings.shape[0], elapsed,
            embeddings.shape[0] / max(elapsed, 1e-6),
            embeddings.shape,
        )

        # Release the extractor's GPU memory now that we have the matrix.
        del extractor
        torch.cuda.empty_cache()

        if len(records) != embeddings.shape[0]:
            logger.warning(
                "Record count (%d) does not match embedding count (%d); "
                "truncating records to embedding count",
                len(records), embeddings.shape[0],
            )
            records = records[: embeddings.shape[0]]

        return embeddings, records

    # -- stage 3: density resampling ----------------------------------------

    def _density_resample(
        self, embeddings: np.ndarray, records: list[CorpusText],
    ) -> tuple[np.ndarray, list[CorpusText], ResamplingResult | None]:
        """Apply density-aware resampling if configured."""
        cfg = self.config.density
        if not cfg.enabled:
            logger.info(
                "Density resampling disabled; using %d raw embeddings",
                embeddings.shape[0],
            )
            return embeddings, records, None

        rng = np.random.default_rng(self.config.shuffle_seed)
        reweighter = DensityReweighter(knn_k=cfg.knn_k, temperature=cfg.temperature)
        result = reweighter.resample(
            embeddings=embeddings,
            target_size=cfg.target_size,
            rng=rng,
        )
        chosen = result.indices
        new_records = [records[i] for i in chosen]
        return result.embeddings.astype(np.float32), new_records, result

    # -- stage 4: clustering ------------------------------------------------

    def _cluster(self, embeddings: np.ndarray) -> np.ndarray:
        return run_minibatch_kmeans(embeddings, self.config.cluster)

    # -- stage 5: persist ---------------------------------------------------

    def _save(
        self,
        embeddings: np.ndarray,
        cluster_labels: np.ndarray,
        records: list[CorpusText],
        density_result: ResamplingResult | None,
    ) -> EmbeddingStore:
        metadata: dict = {
            "pipeline": "lfm.qwen_targets",
            "model_name": self.config.extractor.model_name,
            "layer": self.config.extractor.layer,
            "pooling": self.config.extractor.pooling,
            "density_enabled": self.config.density.enabled,
            "density_temperature": self.config.density.temperature,
            "density_knn_k": self.config.density.knn_k,
            "num_sources": len(self.config.sources),
            "source_names": [e.name or Path(e.path).stem for e in self.config.sources],
        }
        if density_result is not None:
            metadata["density_knn_stats"] = {
                "mean": float(density_result.knn_distances.mean()),
                "std": float(density_result.knn_distances.std()),
                "min": float(density_result.knn_distances.min()),
                "max": float(density_result.knn_distances.max()),
            }

        passages = [
            {
                "text": rec.text,
                "source": rec.source_name,
                "source_index": rec.source_index,
            }
            for rec in records
        ]

        return EmbeddingStore.create(
            store_dir=self.config.output_dir,
            embeddings=embeddings,
            cluster_labels=cluster_labels,
            metadata=metadata,
            passages=passages,
        )

    # -- top-level ----------------------------------------------------------

    def build(self) -> EmbeddingStore:
        """Run all stages and return the loaded store.

        If a store already exists at the configured output_dir, it is
        loaded and returned without re-running.  Delete the directory
        to force a fresh build.
        """
        out_dir = Path(self.config.output_dir)
        if (out_dir / "embeddings.npy").exists() and (out_dir / "metadata.json").exists():
            logger.info(
                "Existing store at %s — loading instead of rebuilding",
                out_dir,
            )
            store = EmbeddingStore(out_dir)
            store.load()
            return store

        logger.info("Building Qwen-latent target store at %s", out_dir)
        mixer = self._build_mixer()
        embeddings, records = self._extract(mixer)
        embeddings, records, density_result = self._density_resample(embeddings, records)
        cluster_labels = self._cluster(embeddings)
        store = self._save(embeddings, cluster_labels, records, density_result)
        logger.info("Qwen-latent store ready: %s", out_dir)
        return store

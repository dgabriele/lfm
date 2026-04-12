"""HuggingFace ``datasets`` streaming corpus source.

Streams text examples from a HuggingFace hub dataset without
downloading the whole archive, using ``load_dataset(..., streaming=True)``.
Supports per-source subset (``config_name``), split selection, text
field mapping, length filtering, optional content truncation, and
optional text filters that drop records before they reach the cache
(e.g. license headers, Gutenberg preambles, exact duplicates).

The streaming backend returns an :class:`~datasets.IterableDataset`,
so iteration cost is bounded by the examples actually pulled.  A
``max_samples`` cap is strictly honoured — the dataset is closed as
soon as the cap is hit.
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Iterator

from lfm.qwen_targets.corpora.base import CorpusSource, CorpusText

logger = logging.getLogger(__name__)


TextPredicate = Callable[[str], bool]


class HFStreamingCorpusSource(CorpusSource):
    """Stream :class:`CorpusText` records from a HuggingFace dataset.

    Args:
        dataset_name: HF dataset identifier, e.g.
            ``cerebras/SlimPajama-627B``.
        config_name: Optional dataset config/subset (e.g. the language
            code on multilingual datasets).
        split: Which split to stream; defaults to ``"train"``.
        text_field: The record key that holds the text content.
        name: Human-readable label used for provenance and logging.
        max_samples: Optional cap on emitted examples.
        min_length: Minimum text length in characters; shorter records
            are skipped (removes boilerplate and empty lines).
        max_length: Maximum text length in characters; longer records
            are truncated to this prefix.  Protects the extractor from
            pathological inputs without dropping them entirely.
        filter_fn: Optional predicate on the raw *record dict*.  Records
            for which it returns ``False`` are skipped before any text
            extraction or length filtering.
        text_filters: Optional list of predicates on the *extracted
            text* (after length filtering / truncation).  Each filter
            returns ``True`` to keep the text, ``False`` to drop it.
            Dropped records never reach the downstream cache.  Use
            :mod:`lfm.qwen_targets.filters` for reusable predicates.
        trust_remote_code: Forwarded to ``load_dataset``.  Required for
            some datasets with custom loading scripts.
    """

    def __init__(
        self,
        dataset_name: str,
        *,
        config_name: str | None = None,
        split: str = "train",
        text_field: str = "text",
        name: str,
        max_samples: int | None = None,
        min_length: int = 20,
        max_length: int = 4000,
        filter_fn: Callable[[dict[str, Any]], bool] | None = None,
        text_filters: list[TextPredicate] | None = None,
        trust_remote_code: bool = False,
    ) -> None:
        super().__init__(name=name, max_samples=max_samples)
        self.dataset_name = dataset_name
        self.config_name = config_name
        self.split = split
        self.text_field = text_field
        self.min_length = min_length
        self.max_length = max_length
        self.filter_fn = filter_fn
        self.text_filters = list(text_filters) if text_filters else []
        self.trust_remote_code = trust_remote_code

    def iterate(self) -> Iterator[CorpusText]:
        from datasets import load_dataset

        logger.info(
            "Streaming HF dataset %s (config=%s, split=%s, text_field=%s, cap=%s)",
            self.dataset_name, self.config_name, self.split, self.text_field,
            self.max_samples,
        )

        load_kwargs: dict[str, Any] = {
            "split": self.split,
            "streaming": True,
        }
        if self.config_name is not None:
            load_kwargs["name"] = self.config_name
        if self.trust_remote_code:
            load_kwargs["trust_remote_code"] = True

        ds = load_dataset(self.dataset_name, **load_kwargs)

        emitted = 0
        raw_index = 0
        filtered_out = 0
        for record in ds:
            raw_index += 1
            if self.filter_fn is not None and not self.filter_fn(record):
                continue
            text = record.get(self.text_field) if isinstance(record, dict) else None
            if not isinstance(text, str):
                continue
            text = text.strip()
            if len(text) < self.min_length:
                continue
            if len(text) > self.max_length:
                text = text[: self.max_length]
            # Text-level filters: drop boilerplate, duplicates, etc.
            if self.text_filters and not all(f(text) for f in self.text_filters):
                filtered_out += 1
                continue
            yield CorpusText(
                text=text,
                source_name=self.name,
                source_index=raw_index,
            )
            emitted += 1
            if self.max_samples is not None and emitted >= self.max_samples:
                logger.info(
                    "%s: reached max_samples=%d after %d raw records "
                    "(text_filters dropped %d)",
                    self.name, self.max_samples, raw_index, filtered_out,
                )
                return

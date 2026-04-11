"""Corpus source wrapper that splits documents into sentence-aware chunks.

Wraps any :class:`CorpusSource` and applies a
:class:`~lfm.qwen_targets.chunking.SentenceAwareChunker` to each
yielded document, emitting ``0..max_chunks_per_doc`` :class:`CorpusText`
records per input.  Documents are buffered in batches so the Rust-
backed tokenizer can amortize its per-call overhead across many texts.
"""

from __future__ import annotations

import logging
from typing import Iterator

from lfm.qwen_targets.chunking import SentenceAwareChunker
from lfm.qwen_targets.corpora.base import CorpusSource, CorpusText

logger = logging.getLogger(__name__)


class ChunkedCorpusSource(CorpusSource):
    """Wrap a :class:`CorpusSource` and chunk each yielded document.

    Inherits ``max_samples`` from the wrapped source — which caps the
    *input* document count, not the output chunk count.  The number of
    emitted chunks depends on each document's length.

    Args:
        source: The underlying corpus source.
        chunker: A configured :class:`SentenceAwareChunker`.
        buffer_size: How many documents to buffer before calling the
            tokenizer.  Larger values amortize Rust-call overhead but
            cost more memory; 1024 is a reasonable default for text.
    """

    def __init__(
        self,
        source: CorpusSource,
        chunker: SentenceAwareChunker,
        buffer_size: int = 1024,
    ) -> None:
        super().__init__(name=source.name, max_samples=source.max_samples)
        self.source = source
        self.chunker = chunker
        self.buffer_size = buffer_size

    def iterate(self) -> Iterator[CorpusText]:
        buffer: list[CorpusText] = []
        for record in self.source.iterate():
            buffer.append(record)
            if len(buffer) >= self.buffer_size:
                yield from self._flush(buffer)
                buffer = []
        if buffer:
            yield from self._flush(buffer)

    def _flush(self, buffer: list[CorpusText]) -> Iterator[CorpusText]:
        texts = [r.text for r in buffer]
        chunk_lists = self.chunker.chunk_batch(texts)
        for record, chunks in zip(buffer, chunk_lists):
            for chunk in chunks:
                yield CorpusText(
                    text=chunk,
                    source_name=record.source_name,
                    source_index=record.source_index,
                )

"""Sentence-aware document chunking for coherent-unit target extraction.

The downstream extractor pools the last token of each input sequence
to produce a single target embedding.  That embedding is only
semantically meaningful if the sequence ends at a natural boundary —
pooling mid-sentence gives a hidden state representing "half a
thought," which is not a point we want the dialogue agent to learn
to produce.

This module splits documents into contiguous chunks of one or more
whole sentences, each sized to fit within a token budget matching the
extractor's ``max_len``.  Short documents pass through as a single
chunk.  Long documents yield up to ``max_chunks_per_doc`` chunks, each
a complete multi-sentence span.

Implementation notes:
  * Uses the HuggingFace *fast* tokenizer's ``offset_mapping`` to
    locate sentence-boundary character positions in token space
    without re-tokenizing per sentence.  This keeps per-document
    overhead dominated by a single Rust-backed tokenizer call and
    lets us batch many documents per call.
  * Sentence boundaries are detected by a small multilingual regex
    covering Latin (. ! ?) and CJK (。！？) punctuation plus
    paragraph breaks.  Abbreviation edge cases ("Mr. Smith") are
    intentionally not handled — a false split produces two
    slightly-shorter chunks rather than a broken one, and the cost is
    negligible for our purpose.
"""

from __future__ import annotations

import bisect
import logging
import re

logger = logging.getLogger(__name__)


# Sentence-end pattern: any run of Latin or CJK sentence-ending
# punctuation, or a paragraph break (two or more newlines).
_SENT_END_RE = re.compile(r"[.!?。！？]+|\n{2,}")


class SentenceAwareChunker:
    """Split documents into coherent chunks bounded by a token budget.

    Args:
        tokenizer_name: HuggingFace model name whose tokenizer to use.
            Must match the extractor's ``model_name`` so chunk token
            counts align with the downstream ``max_len``.
        max_tokens: Maximum tokens per chunk.  Should equal (or sit
            slightly below) the extractor's ``max_len``.
        max_chunks_per_doc: Cap on chunks emitted per input document.
            Prevents a single very long source from dominating output.
        min_chunk_tokens: Drop chunks shorter than this.  Avoids
            emitting single-word fragments as separate samples.
    """

    def __init__(
        self,
        tokenizer_name: str,
        max_tokens: int,
        max_chunks_per_doc: int = 3,
        min_chunk_tokens: int = 30,
    ) -> None:
        from transformers import AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_name, use_fast=True,
        )
        if not self.tokenizer.is_fast:
            raise RuntimeError(
                f"{tokenizer_name} has no fast tokenizer; offset_mapping "
                "is unavailable and chunking would be prohibitively slow.",
            )
        self.max_tokens = max_tokens
        self.max_chunks_per_doc = max_chunks_per_doc
        self.min_chunk_tokens = min_chunk_tokens
        logger.info(
            "SentenceAwareChunker: model=%s max_tokens=%d max_chunks=%d min_tokens=%d",
            tokenizer_name, max_tokens, max_chunks_per_doc, min_chunk_tokens,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def chunk(self, text: str) -> list[str]:
        """Chunk a single text.  Convenience wrapper over :meth:`chunk_batch`."""
        return self.chunk_batch([text])[0]

    def chunk_batch(self, texts: list[str]) -> list[list[str]]:
        """Chunk a batch of texts; returns one chunk list per input.

        Empty input texts return empty chunk lists.  Batching cuts
        per-document tokenizer overhead by 10-20× vs one-at-a-time.
        """
        if not texts:
            return []
        enc = self.tokenizer(
            texts,
            add_special_tokens=False,
            return_offsets_mapping=True,
            truncation=False,
        )
        return [
            self._chunk_one(text, ids, offsets)
            for text, ids, offsets in zip(
                texts, enc["input_ids"], enc["offset_mapping"],
            )
        ]

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _chunk_one(
        self,
        text: str,
        token_ids: list[int],
        offsets: list[tuple[int, int]],
    ) -> list[str]:
        n_tokens = len(token_ids)
        if n_tokens == 0:
            return []

        stripped = text.strip()

        # Fast path: short-enough documents pass through as a single
        # chunk (minus the whitespace trim), if they meet the floor.
        if n_tokens <= self.max_tokens:
            if n_tokens >= self.min_chunk_tokens and stripped:
                return [stripped]
            return []

        # Locate sentence boundaries in the original text and project
        # them onto token-index positions via offset_mapping.  A
        # "boundary" is the token index 1 + (last token whose end char
        # offset is ≤ the boundary character position).
        token_ends = [o[1] for o in offsets]
        boundaries: list[int] = []
        for match in _SENT_END_RE.finditer(text):
            idx = bisect.bisect_right(token_ends, match.end())
            if 0 < idx <= n_tokens and (not boundaries or idx > boundaries[-1]):
                boundaries.append(idx)
        # Ensure the final token index is a candidate boundary so the
        # tail of the document can flush into a chunk.
        if not boundaries or boundaries[-1] != n_tokens:
            boundaries.append(n_tokens)

        # Greedy pack: walk boundaries and emit a chunk whenever
        # adding the next boundary would exceed the token budget.
        chunks: list[str] = []
        chunk_start = 0
        last_valid_end = 0
        for bdry in boundaries:
            if bdry - chunk_start <= self.max_tokens:
                last_valid_end = bdry
                continue
            # Overflow: flush whatever we had and start a fresh chunk.
            self._maybe_emit(chunks, text, offsets, chunk_start, last_valid_end)
            if len(chunks) >= self.max_chunks_per_doc:
                return chunks
            # Restart from the previous accepted boundary.  If that
            # boundary is still far from ``bdry`` (a single
            # super-long "sentence"), skip forward so we don't
            # livelock; the unrepresentable span is dropped.
            chunk_start = last_valid_end
            if bdry - chunk_start <= self.max_tokens:
                last_valid_end = bdry
            else:
                chunk_start = bdry
                last_valid_end = bdry

        # Flush any trailing chunk
        self._maybe_emit(chunks, text, offsets, chunk_start, last_valid_end)
        return chunks[: self.max_chunks_per_doc]

    def _maybe_emit(
        self,
        chunks: list[str],
        text: str,
        offsets: list[tuple[int, int]],
        start_token: int,
        end_token: int,
    ) -> None:
        """Append a chunk if it meets the minimum-token floor."""
        if end_token - start_token < self.min_chunk_tokens:
            return
        start_char = offsets[start_token][0]
        end_char = offsets[end_token - 1][1]
        chunk_text = text[start_char:end_char].strip()
        if chunk_text:
            chunks.append(chunk_text)

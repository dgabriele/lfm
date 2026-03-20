"""Sliding-window text chunker for the embeddings pipeline.

Splits raw text into overlapping passages of bounded token length, suitable
for encoding by a frozen text encoder.  Respects token boundaries by
tokenizing first and then decoding each window back to text.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from lfm.embeddings.config import ChunkerConfig
from lfm.utils.logging import get_logger

logger = get_logger(__name__)


class TextChunker:
    """Tokenize-then-window text chunker.

    Uses the encoder's own tokenizer to split text into overlapping windows of
    ``max_tokens`` tokens with a stride of ``max_tokens - overlap_tokens``.
    Windows shorter than ``min_tokens`` are discarded.

    Args:
        config: Chunker configuration.
        tokenizer: A HuggingFace-compatible tokenizer exposing ``encode`` and
            ``decode`` methods.
    """

    def __init__(self, config: ChunkerConfig, tokenizer: Any) -> None:
        self.config = config
        self._tokenizer = tokenizer

    def chunk_text(self, text: str, source: str = "") -> list[dict[str, Any]]:
        """Chunk a single text into overlapping token windows.

        Args:
            text: The full document text to chunk.
            source: Optional provenance identifier (e.g. file path) stored in
                each chunk's metadata.

        Returns:
            List of chunk dicts, each containing:
                - ``"text"``: decoded passage string
                - ``"source"``: provenance identifier
                - ``"start_token"``: starting token index in the document
                - ``"end_token"``: ending token index (exclusive)
        """
        text = text.strip()
        if not text:
            return []

        token_ids: list[int] = self._tokenizer.encode(text, add_special_tokens=False)
        total_tokens = len(token_ids)

        if total_tokens < self.config.min_tokens:
            return []

        stride = self.config.max_tokens - self.config.overlap_tokens
        if stride <= 0:
            raise ValueError(
                f"Stride must be positive: max_tokens ({self.config.max_tokens}) "
                f"- overlap_tokens ({self.config.overlap_tokens}) = {stride}"
            )

        chunks: list[dict[str, Any]] = []
        start = 0
        while start < total_tokens:
            end = min(start + self.config.max_tokens, total_tokens)
            window_len = end - start

            if window_len < self.config.min_tokens:
                break

            window_ids = token_ids[start:end]
            passage_text = self._tokenizer.decode(window_ids, skip_special_tokens=True)

            chunks.append(
                {
                    "text": passage_text,
                    "source": source,
                    "start_token": start,
                    "end_token": end,
                }
            )
            start += stride

        return chunks

    def chunk_file(self, path: str) -> list[dict[str, Any]]:
        """Read a file from disk and chunk its contents.

        Args:
            path: Path to a plain text file.

        Returns:
            List of chunk dicts (see :meth:`chunk_text`).
        """
        file_path = Path(path)
        text = file_path.read_text(encoding="utf-8")
        return self.chunk_text(text, source=str(file_path))

    def chunk_corpus(self, paths: list[str], corpus_format: str = "text") -> list[dict[str, Any]]:
        """Chunk an entire corpus from one or more paths.

        For each path:
          - If it is a directory, all ``.txt`` files are recursively discovered
            and chunked.
          - If ``corpus_format`` is ``"text"``, the file is read as plain text.
          - If ``corpus_format`` is ``"jsonl"``, each line is parsed as JSON and
            the ``"text"`` field is extracted and chunked independently.

        Args:
            paths: File or directory paths comprising the corpus.
            corpus_format: Either ``"text"`` or ``"jsonl"``.

        Returns:
            Concatenated list of chunk dicts from all files.
        """
        all_chunks: list[dict[str, Any]] = []

        resolved_files = self._resolve_paths(paths, corpus_format)
        logger.info("Resolved %d files from %d paths", len(resolved_files), len(paths))

        for file_path in resolved_files:
            try:
                if corpus_format == "jsonl":
                    file_chunks = self._chunk_jsonl_file(file_path)
                else:
                    file_chunks = self.chunk_file(str(file_path))
                all_chunks.extend(file_chunks)
                logger.info(
                    "Chunked %s -> %d passages (total: %d)",
                    file_path.name,
                    len(file_chunks),
                    len(all_chunks),
                )
            except Exception:
                logger.exception("Failed to chunk file: %s", file_path)

        logger.info(
            "Corpus chunking complete: %d passages from %d files",
            len(all_chunks),
            len(resolved_files),
        )
        return all_chunks

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _resolve_paths(self, paths: list[str], corpus_format: str) -> list[Path]:
        """Expand directories into individual files."""
        extension = ".jsonl" if corpus_format == "jsonl" else ".txt"
        resolved: list[Path] = []

        for p in paths:
            path = Path(p)
            if path.is_dir():
                resolved.extend(sorted(path.rglob(f"*{extension}")))
            elif path.is_file():
                resolved.append(path)
            else:
                logger.warning("Path does not exist, skipping: %s", path)

        return resolved

    def _chunk_jsonl_file(self, file_path: Path) -> list[dict[str, Any]]:
        """Chunk a JSONL file where each line has a ``"text"`` field."""
        chunks: list[dict[str, Any]] = []
        source = str(file_path)
        with file_path.open("r", encoding="utf-8") as fh:
            for line_no, line in enumerate(fh, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    logger.warning("Skipping malformed JSON at %s:%d", file_path, line_no)
                    continue
                text = record.get("text", "")
                if text:
                    line_source = f"{source}:{line_no}"
                    chunks.extend(self.chunk_text(text, source=line_source))
        return chunks

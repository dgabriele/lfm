"""JSONL and plain-text corpus source implementations."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterator

from lfm.qwen_targets.corpora.base import CorpusSource, CorpusText


class JSONLCorpusSource(CorpusSource):
    """Iterate over a JSONL file, pulling a text field from each line.

    Args:
        path: Path to the JSONL file.
        text_field: Key of the text field within each JSON object.
        name: Human-readable label.
        max_samples: Optional cap on number of examples yielded.
    """

    def __init__(
        self,
        path: str | Path,
        text_field: str = "text",
        name: str | None = None,
        max_samples: int | None = None,
    ) -> None:
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"JSONL corpus not found: {path}")
        super().__init__(name=name or path.stem, max_samples=max_samples)
        self.path = path
        self.text_field = text_field

    def iterate(self) -> Iterator[CorpusText]:
        emitted = 0
        with open(self.path, encoding="utf-8") as fh:
            for idx, line in enumerate(fh):
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue
                text = record.get(self.text_field)
                if not isinstance(text, str) or not text.strip():
                    continue
                yield CorpusText(
                    text=text.strip(),
                    source_name=self.name,
                    source_index=idx,
                )
                emitted += 1
                if self.max_samples is not None and emitted >= self.max_samples:
                    return


class PlainTextCorpusSource(CorpusSource):
    """Iterate over a plain-text file, one example per non-empty line.

    Args:
        path: Path to the text file.
        name: Human-readable label.
        max_samples: Optional cap on number of examples yielded.
    """

    def __init__(
        self,
        path: str | Path,
        name: str | None = None,
        max_samples: int | None = None,
    ) -> None:
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Text corpus not found: {path}")
        super().__init__(name=name or path.stem, max_samples=max_samples)
        self.path = path

    def iterate(self) -> Iterator[CorpusText]:
        emitted = 0
        with open(self.path, encoding="utf-8") as fh:
            for idx, line in enumerate(fh):
                text = line.strip()
                if not text:
                    continue
                yield CorpusText(
                    text=text,
                    source_name=self.name,
                    source_index=idx,
                )
                emitted += 1
                if self.max_samples is not None and emitted >= self.max_samples:
                    return

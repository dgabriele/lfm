"""Abstract base class for corpus sources.

A :class:`CorpusSource` is any object that can lazily iterate over text
examples.  Concrete implementations include plain-text files, JSONL
files, and (in the future) HuggingFace datasets.  The abstraction lets
us plug different corpora into the extraction pipeline without the
builder knowing about file formats.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Iterator


@dataclass(frozen=True)
class CorpusText:
    """One text example from a corpus source.

    Attributes:
        text: The actual text content.
        source_name: Label of the originating source (for provenance).
        source_index: Index of this text within its originating source.
    """

    text: str
    source_name: str
    source_index: int


class CorpusSource(ABC):
    """Lazy iterable over text examples from one corpus.

    Subclasses implement :meth:`iterate` to yield :class:`CorpusText`
    objects one at a time, without loading the entire corpus into
    memory.  Implementations should respect ``max_samples`` to cap
    output from oversized sources.

    Args:
        name: Human-readable label for logging and provenance.
        max_samples: Hard cap on number of examples yielded. ``None``
            means no cap.
    """

    def __init__(self, name: str, max_samples: int | None = None) -> None:
        self.name = name
        self.max_samples = max_samples

    @abstractmethod
    def iterate(self) -> Iterator[CorpusText]:
        """Yield :class:`CorpusText` instances lazily."""

    def __iter__(self) -> Iterator[CorpusText]:
        return self.iterate()

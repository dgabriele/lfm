"""Base protocol for constituency parsing backends."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol


@dataclass
class ParseTree:
    """A node in a constituency parse tree.

    Mirrors the interface of Stanza's Tree objects but is
    backend-independent.
    """

    label: str
    children: list[ParseTree] = field(default_factory=list)

    @property
    def is_leaf(self) -> bool:
        return not self.children

    def leaf_text(self) -> str:
        """Collect leaf labels (terminal words) left-to-right."""
        if self.is_leaf:
            return self.label
        return " ".join(c.leaf_text() for c in self.children)


class ConstituencyBackend(Protocol):
    """Protocol for constituency parsing backends.

    Each backend handles a set of languages and returns ParseTree
    objects from raw text.
    """

    def parse(self, sentences: list[str]) -> list[ParseTree | None]:
        """Parse sentences into constituency trees.

        Args:
            sentences: Raw text sentences.

        Returns:
            List of ParseTree roots, one per sentence.
            None for sentences that failed to parse.
        """
        ...

    def supports(self, lang_iso3: str) -> bool:
        """Check if this backend supports the given language."""
        ...

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
    position: int = -1  # sentence position (0-based) for leaf nodes

    @property
    def is_leaf(self) -> bool:
        return not self.children

    def leaf_text(self) -> str:
        """Collect leaf labels in sentence order (by position)."""
        leaves = self._collect_leaves()
        leaves.sort(key=lambda lp: lp[1])
        return " ".join(text for text, _ in leaves)

    def _collect_leaves(self) -> list[tuple[str, int]]:
        """Collect (text, position) for all leaves."""
        if self.is_leaf:
            return [(self.label, self.position)]
        result = []
        for c in self.children:
            result.extend(c._collect_leaves())
        return result

    def min_position(self) -> int:
        """Leftmost word position in this subtree."""
        if self.is_leaf:
            return self.position if self.position >= 0 else 999999
        return min((c.min_position() for c in self.children), default=999999)


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

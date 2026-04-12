"""Text filters for discarding low-information boilerplate from a corpus.

These filters are designed to catch content that either:

1. **Dominates unfairly** — license headers copy-pasted across thousands
   of code files, Project Gutenberg front matter repeated across many
   book records, and similar template boilerplate.  Such content would
   cluster together in k-means output and crowd out actual semantic
   variety.

2. **Near-duplicates** — identical or extremely similar text that would
   produce identical embeddings, collapsing cluster density in a
   single spot and wasting budget.

The filters are small, pure-function predicates with a uniform
signature — ``(text: str) -> bool`` returning ``True`` to **keep** the
record — so they can be composed in a list and applied at prefetch
time (to keep the cache clean) or post-hoc (to clean an existing
built store before re-clustering).
"""

from __future__ import annotations

import hashlib
import logging
import re

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Project Gutenberg preamble / license boilerplate
# ---------------------------------------------------------------------------

_GUTENBERG_MARKERS = (
    "Project Gutenberg eBook",
    "Project Gutenberg EBook",
    "The Project Gutenberg",
    "Project Gutenberg's",
    "This eBook is for the use of anyone anywhere",
    "Release Date:",
    "Character set encoding",
    "START OF THIS PROJECT GUTENBERG",
    "START OF THE PROJECT GUTENBERG",
    "END OF THIS PROJECT GUTENBERG",
    "END OF THE PROJECT GUTENBERG",
    "www.gutenberg.org",
    "Updated editions will replace",
    "Title: ",
    "Author: ",
)
_GUTENBERG_RE = re.compile("|".join(re.escape(m) for m in _GUTENBERG_MARKERS))


def is_gutenberg_boilerplate(text: str, min_hits: int = 2) -> bool:
    """True if the text reads like Gutenberg front matter.

    Requires ``min_hits`` distinct marker phrases so a passing mention
    of "The Project Gutenberg" inside a fictional passage doesn't get
    falsely flagged.  The actual front matter has *many* markers.
    """
    hits = set(_GUTENBERG_RE.findall(text))
    return len(hits) >= min_hits


def gutenberg_filter(text: str) -> bool:
    """Keep iff the text is NOT Gutenberg front matter."""
    return not is_gutenberg_boilerplate(text)


# ---------------------------------------------------------------------------
# Software license headers
# ---------------------------------------------------------------------------

_LICENSE_MARKERS = (
    "Licensed under the Apache License",
    "Apache License, Version 2.0",
    "Licensed to the Apache Software Foundation",
    "GNU General Public License",
    "GNU Lesser General Public License",
    "The MIT License",
    "Permission is hereby granted, free of charge",
    "BSD 2-Clause",
    "BSD 3-Clause",
    "Redistribution and use in source and binary forms",
    "THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS",
    '"AS IS" AND ANY EXPRESS OR IMPLIED',
    "WITHOUT WARRANTY OF ANY KIND",
    "EXPRESS OR IMPLIED WARRANTIES",
    "NO EVENT SHALL",
    "WITHOUT LIMITATION THE RIGHTS",
    "above copyright notice",
    "Mozilla Public License",
    "GNU Affero General Public License",
    "SPDX-License-Identifier",
)
_LICENSE_RE = re.compile("|".join(re.escape(m) for m in _LICENSE_MARKERS))


def is_license_boilerplate(text: str, min_hits: int = 2) -> bool:
    """True if the text is dominated by software-license boilerplate.

    Requires ``min_hits`` distinct license phrases: one casual mention
    like ``"Apache License"`` inside a tutorial doesn't count, but a
    block with ``"Licensed under the Apache License"`` + ``"WITHOUT
    WARRANTY OF ANY KIND"`` almost certainly is a copied header.
    """
    hits = set(_LICENSE_RE.findall(text))
    return len(hits) >= min_hits


def license_filter(text: str) -> bool:
    """Keep iff the text is NOT dominated by license boilerplate."""
    return not is_license_boilerplate(text)


# ---------------------------------------------------------------------------
# Exact / near-duplicate detection (stateful)
# ---------------------------------------------------------------------------


class DuplicateFilter:
    """Stateful filter that drops repeat texts.

    Two records with identical text (after stripping whitespace) are
    collapsed to the first occurrence.  A SHA-256 hash of the
    normalized text is kept in a set, so this costs O(N) memory in
    the number of unique records seen.

    For 1-2M records the memory cost is ~64-128 MB of hash strings —
    fine for our scale.
    """

    def __init__(self) -> None:
        self._seen: set[str] = set()
        self.dropped = 0
        self.kept = 0

    def __call__(self, text: str) -> bool:
        key = hashlib.sha256(" ".join(text.split()).encode("utf-8")).hexdigest()
        if key in self._seen:
            self.dropped += 1
            return False
        self._seen.add(key)
        self.kept += 1
        return True


# ---------------------------------------------------------------------------
# Composition helper
# ---------------------------------------------------------------------------


def apply_filters(text: str, filters: list) -> bool:
    """Return True if ``text`` passes *every* filter in the list."""
    return all(f(text) for f in filters)

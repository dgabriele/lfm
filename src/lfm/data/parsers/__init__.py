"""Constituency parsing backends for phrase extraction.

Provides a unified interface across multiple parser backends:

- **StanzaBackend**: Stanza constituency parser (de, en, es, id, pt, tr, vi)
- **BeneparBackend**: Berkeley Neural Parser (ar, hu, ko, pl)
- **DepConBackend**: UD dependency → constituency conversion (cs, et, fi, hi, ru)

Usage::

    from lfm.data.parsers import get_backend

    backend = get_backend("deu")  # returns StanzaBackend
    trees = backend.parse(["Ein Satz.", "Noch einer."])
"""

from lfm.data.parsers.base import ConstituencyBackend, ParseTree
from lfm.data.parsers.registry import get_backend, supported_languages

__all__ = [
    "ConstituencyBackend",
    "ParseTree",
    "get_backend",
    "supported_languages",
]

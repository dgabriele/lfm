"""Syntax subsystem for the LFM framework.

Provides the abstract ``SyntaxModule`` base class and ``SyntaxConfig``
for inducing hierarchical constituency structure over token sequences.
"""

from __future__ import annotations

from lfm.syntax.base import SyntaxModule
from lfm.syntax.config import SyntaxConfig

__all__ = ["SyntaxConfig", "SyntaxModule"]

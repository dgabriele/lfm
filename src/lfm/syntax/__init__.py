"""Syntax subsystem for the LFM framework.

Provides the abstract ``SyntaxModule`` base class and ``SyntaxConfig``
for learning structural agreement and ordering constraints over token
sequences. Phrase structure emerges from morphological agreement and
information-theoretic ordering pressures rather than explicit grammar
induction.

Implementations:
    - ``agreement`` -- soft agreement constraints via multi-head attention
    - ``morphological_attention`` -- attention biased by feature similarity
    - ``ordering_pressure`` -- information-theoretic ordering scores
"""

from __future__ import annotations

from lfm.syntax.base import SyntaxModule
from lfm.syntax.config import SyntaxConfig

__all__ = ["SyntaxConfig", "SyntaxModule"]

"""Configuration for syntax modules.

Defines the ``SyntaxConfig`` used to parameterize syntactic parsing components
that impose hierarchical constituency structure on token sequences.
"""

from __future__ import annotations

from lfm.config.base import LFMBaseConfig


class SyntaxConfig(LFMBaseConfig):
    """Configuration for a syntax module.

    Attributes:
        name: Registry name of the syntax implementation to use.
        num_nonterminals: Number of nonterminal categories in the induced
            grammar.
        num_preterminals: Number of preterminal (part-of-speech-like)
            categories.
        latent_dim: Dimensionality of the latent syntactic representations.
    """

    name: str = "neural_pcfg"
    num_nonterminals: int = 30
    num_preterminals: int = 60
    latent_dim: int = 64

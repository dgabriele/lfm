"""Learnable structured expression generation via the LFM linguistic bottleneck.

This package provides modular components for learning compositional
tree-structured expressions through a frozen multilingual VAE decoder.
An agent produces a binary constituency tree where the topology is learned
and each leaf carries a latent z vector.  The leaves are decoded as one
continuous autoregressive IPA sequence with z-switching at phrase
boundaries — the KV cache carries across transitions, producing natural
coarticulation and prosodic coherence.

Architecture::

    agent embedding
      → ExpressionGenerator
          → learn tree topology (expand/leaf decisions via REINFORCE)
          → project leaf hidden states → (μ, σ) → sample z
          → continuous AR decode: z₁ → z₂ → z₃ (KV cache persists)
      → Expression (topology + decoded token sequence + phrase boundaries)
      → ExpressionEncoder
          → phrase-level pooling + tree-guided composition
          → fixed-size message vector for downstream use

Usage::

    from lfm.expression import Expression, ExpressionGenerator, ExpressionEncoder
    from lfm.faculty import LanguageFaculty

    faculty = LanguageFaculty(config)
    gen = ExpressionGenerator(
        generator=faculty.generator,
        input_dim=384, latent_dim=384, hidden_dim=512,
    )
    enc = ExpressionEncoder(hidden_dim=512, output_dim=384)

    expression = gen(agent_embedding)   # topology + continuous decode
    message = enc(expression)           # tree-composed message vector
"""

from lfm.expression.config import ExpressionConfig
from lfm.expression.encoder import ExpressionEncoder
from lfm.expression.expression import Expression
from lfm.expression.generator import ExpressionGenerator

__all__ = [
    "Expression",
    "ExpressionConfig",
    "ExpressionEncoder",
    "ExpressionGenerator",
]

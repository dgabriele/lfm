"""Generator subsystem for the LFM framework.

Provides the abstract ``GeneratorModule`` base class and ``GeneratorConfig``
for generating structurally well-formed subword token sequences from agent
embeddings via a pretrained multilingual VAE decoder.
"""

from __future__ import annotations

from lfm.generator.base import GeneratorModule
from lfm.generator.config import GeneratorConfig

__all__ = ["GeneratorConfig", "GeneratorModule"]

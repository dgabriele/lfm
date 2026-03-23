"""Configuration for the LanguageFaculty compositor.

Defines ``FacultyConfig``, which configures the generator-based
linguistic bottleneck pipeline.
"""

from __future__ import annotations

from lfm.config.base import LFMBaseConfig
from lfm.generator.config import GeneratorConfig


class FacultyConfig(LFMBaseConfig):
    """Configuration for the LanguageFaculty pipeline.

    The primary pipeline uses the **generator** (pretrained multilingual
    VAE decoder) to produce linguistically structured output from agent
    embeddings.  The generator is configured via ``GeneratorConfig``.

    Attributes:
        dim: Default internal embedding dimension.
        max_seq_len: Maximum sequence length for token sequences.
        pretokenized_dim: Embedding dim of pre-tokenized external input.
            Creates a projection to ``dim`` when set.
        generator: Generator config.  The recommended and default path.
    """

    dim: int = 256
    max_seq_len: int = 64
    pretokenized_dim: int | None = None
    generator: GeneratorConfig | None = None

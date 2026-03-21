"""Configuration for the LanguageFaculty compositor.

Defines ``FacultyConfig``, which aggregates the sub-module configs for every
stage of the LFM pipeline.  Each stage is optional: set a config to ``None``
to disable that stage entirely.
"""

from __future__ import annotations

from lfm.channel.config import ChannelConfig
from lfm.config.base import LFMBaseConfig
from lfm.morphology.config import MorphologyConfig
from lfm.phonology.config import PhonologyConfig
from lfm.quantization.config import QuantizationConfig
from lfm.sentence.config import SentenceConfig
from lfm.syntax.config import SyntaxConfig


class FacultyConfig(LFMBaseConfig):
    """Configuration for the complete LanguageFaculty pipeline.

    Attributes:
        dim: Default internal embedding dimension shared across stages.
        max_seq_len: Maximum sequence length for token sequences.
        quantizer: Quantization stage config, or ``None`` to skip.
        phonology: Phonology stage config, or ``None`` to skip.
            Enabled by default with learnable phonotactic constraints.
        morphology: Morphology stage config, or ``None`` to skip.
        syntax: Syntax stage config, or ``None`` to skip.
        sentence: Sentence-level stage config, or ``None`` to skip.
        channel: Communication channel stage config, or ``None`` to skip.
    """

    dim: int = 256
    max_seq_len: int = 64
    pretokenized_dim: int | None = None
    """Embedding dim of pre-tokenized external input. Creates a projection
    to ``dim`` when set. ``None`` (default) disables."""
    quantizer: QuantizationConfig | None = None
    phonology: PhonologyConfig | None = PhonologyConfig()
    morphology: MorphologyConfig | None = None
    syntax: SyntaxConfig | None = None
    sentence: SentenceConfig | None = None
    channel: ChannelConfig | None = None

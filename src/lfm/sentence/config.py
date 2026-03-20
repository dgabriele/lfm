"""Configuration for sentence modules.

Defines the ``SentenceConfig`` used to parameterize sentence-level processing
components that classify sentence types and detect boundaries.
"""

from __future__ import annotations

from lfm.config.base import LFMBaseConfig


class SentenceConfig(LFMBaseConfig):
    """Configuration for a sentence module.

    Attributes:
        name: Registry name of the sentence implementation to use.
        num_sentence_types: Number of distinct sentence type categories
            (e.g. statement, question, imperative, exclamatory).
        boundary_threshold: Probability threshold for classifying a
            position as a sentence boundary.
    """

    name: str = "type_head"
    num_sentence_types: int = 4
    boundary_threshold: float = 0.5

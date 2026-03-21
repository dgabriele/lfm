"""Configuration for morphology modules.

Defines the ``MorphologyConfig`` used to parameterize morphological segmentation
and composition components that impose internal token structure.
"""

from __future__ import annotations

from lfm.config.base import LFMBaseConfig


class MorphologyConfig(LFMBaseConfig):
    """Configuration for a morphology module.

    Attributes:
        name: Registry name of the morphology implementation to use.
        max_morphemes: Maximum number of morpheme segments per token.
        morpheme_dim: Dimensionality of each morpheme embedding.
        min_morpheme_len: Minimum length of a single morpheme segment.
        max_morpheme_len: Maximum length of a single morpheme segment.
        num_grammatical_features: Number of learned latent grammatical feature
            dimensions.
    """

    name: str = "mdl_segmenter"
    max_morphemes: int = 6
    morpheme_dim: int = 32
    min_morpheme_len: int = 1
    max_morpheme_len: int = 8
    num_grammatical_features: int = 16

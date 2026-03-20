"""Configuration for phonology modules.

Defines the ``PhonologyConfig`` used to parameterize phonological processing
components that constrain token representations toward pronounceable forms.
"""

from __future__ import annotations

from lfm.config.base import LFMBaseConfig


class PhonologyConfig(LFMBaseConfig):
    """Configuration for a phonology module.

    Attributes:
        name: Registry name of the phonology implementation to use.
        phoneme_inventory: Identifier for the phoneme inventory to use
            (e.g. ``"english"``, ``"ipa_full"``).
        max_syllables_per_token: Maximum number of syllables allowed per
            discrete token.
        pronounceability_weight: Scalar weight for the pronounceability
            loss term.
    """

    name: str = "pronounceable"
    phoneme_inventory: str = "english"
    max_syllables_per_token: int = 4
    pronounceability_weight: float = 1.0

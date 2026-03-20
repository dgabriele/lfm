"""Phoneme inventory module.

Provides a configurable phoneme inventory with consonant/vowel categorisation,
sonority rankings, and cluster definitions.  The default inventory covers
English phonemes using IPA-like symbols.
"""

from __future__ import annotations

from typing import ClassVar

from torch import Tensor

from lfm._registry import register
from lfm._types import TokenIds
from lfm.phonology.base import PhonologyModule
from lfm.phonology.config import PhonologyConfig

# ------------------------------------------------------------------
# Default English phoneme inventory data
# ------------------------------------------------------------------

#: English vowel phonemes (monophthongs and common diphthongs).
_DEFAULT_VOWELS: tuple[str, ...] = (
    "i",
    "I",
    "e",
    "E",
    "ae",
    "a",
    "A",
    "o",
    "O",
    "u",
    "U",
    "aI",
    "aU",
    "OI",
    "eI",
    "oU",
)

#: English consonant phonemes.
_DEFAULT_CONSONANTS: tuple[str, ...] = (
    "p",
    "b",
    "t",
    "d",
    "k",
    "g",  # stops
    "f",
    "v",
    "T",
    "D",
    "s",
    "z",  # fricatives
    "S",
    "Z",
    "h",  # fricatives (cont.)
    "tS",
    "dZ",  # affricates
    "m",
    "n",
    "N",  # nasals
    "l",
    "r",  # liquids
    "w",
    "j",  # glides
)

#: Sonority ranking (higher = more sonorous).  Used for sonority sequencing.
_SONORITY: dict[str, int] = {
    # Stops
    "p": 1,
    "b": 1,
    "t": 1,
    "d": 1,
    "k": 1,
    "g": 1,
    # Fricatives
    "f": 2,
    "v": 2,
    "T": 2,
    "D": 2,
    "s": 2,
    "z": 2,
    "S": 2,
    "Z": 2,
    "h": 2,
    # Affricates
    "tS": 2,
    "dZ": 2,
    # Nasals
    "m": 3,
    "n": 3,
    "N": 3,
    # Liquids
    "l": 4,
    "r": 4,
    # Glides
    "w": 5,
    "j": 5,
    # Vowels — most sonorous
    **{v: 6 for v in _DEFAULT_VOWELS},
}

#: Common English onset clusters.
_DEFAULT_ONSET_CLUSTERS: tuple[str, ...] = (
    "pl",
    "pr",
    "bl",
    "br",
    "tr",
    "dr",
    "kl",
    "kr",
    "gl",
    "gr",
    "fl",
    "fr",
    "sl",
    "sm",
    "sn",
    "sp",
    "st",
    "sk",
    "sw",
    "spl",
    "spr",
    "str",
    "skr",
)


@register("phonology", "inventory")
class PhonemeInventory(PhonologyModule):
    """Configurable phoneme inventory with consonant/vowel categorization.

    Maintains a phoneme-to-index mapping, consonant/vowel splits, sonority
    rankings, and valid onset/coda clusters.  The default English inventory
    can be overridden by subclassing or passing a custom inventory
    specification through config.

    Args:
        config: Phonology configuration specifying inventory and constraints.
    """

    #: Default vowels — overridable via subclass.
    DEFAULT_VOWELS: ClassVar[tuple[str, ...]] = _DEFAULT_VOWELS

    #: Default consonants — overridable via subclass.
    DEFAULT_CONSONANTS: ClassVar[tuple[str, ...]] = _DEFAULT_CONSONANTS

    #: Default sonority map — overridable via subclass.
    DEFAULT_SONORITY: ClassVar[dict[str, int]] = _SONORITY

    #: Default onset clusters — overridable via subclass.
    DEFAULT_ONSET_CLUSTERS: ClassVar[tuple[str, ...]] = _DEFAULT_ONSET_CLUSTERS

    def __init__(self, config: PhonologyConfig) -> None:
        super().__init__(config)

        self.vowels = list(self.DEFAULT_VOWELS)
        self.consonants = list(self.DEFAULT_CONSONANTS)
        self.sonority = dict(self.DEFAULT_SONORITY)
        self.onset_clusters = list(self.DEFAULT_ONSET_CLUSTERS)

        # Build phoneme-to-index mapping (0 reserved for padding)
        all_phonemes = self.consonants + self.vowels
        self.phoneme_to_idx: dict[str, int] = {p: i + 1 for i, p in enumerate(all_phonemes)}
        self.idx_to_phoneme: dict[int, str] = {v: k for k, v in self.phoneme_to_idx.items()}
        self.num_phonemes = len(all_phonemes) + 1  # +1 for padding

    # ------------------------------------------------------------------
    # Abstract method implementations
    # ------------------------------------------------------------------

    def to_phonemes(self, tokens: TokenIds) -> Tensor:
        """Map discrete tokens to phoneme sequences.

        Args:
            tokens: Integer token indices of shape ``(batch, seq_len)``.

        Returns:
            Phoneme index tensor of shape ``(batch, seq_len, max_phonemes)``.
        """
        raise NotImplementedError("PhonemeInventory.to_phonemes() not yet implemented")

    def forward(self, tokens: TokenIds, embeddings: Tensor) -> dict[str, Tensor]:
        """Run phonological analysis using the phoneme inventory.

        Args:
            tokens: Integer token indices of shape ``(batch, seq_len)``.
            embeddings: Dense token embeddings of shape
                ``(batch, seq_len, dim)``.

        Returns:
            Dictionary with phoneme sequences, syllable structure,
            pronounceability scores, and enriched embeddings.
        """
        raise NotImplementedError("PhonemeInventory.forward() not yet implemented")

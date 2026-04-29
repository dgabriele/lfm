"""Alien vocabulary: deterministic CV/CVC syllable token set.

All tokens use standard Latin characters and diacritics only
(no phonetic notation). The vocabulary is fully reproducible from
(vocab_size, seed) and is saved/loaded as JSON.
"""

from __future__ import annotations

import json
import random
from pathlib import Path

from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.processors import TemplateProcessing
from tokenizers.trainers import BpeTrainer
from transformers import PreTrainedTokenizerFast

_CONSONANTS = list("bdfghjklmnprstvwz")

# Standard Latin-extended diacritics only — no phonetic symbols.
_VOWEL_VARIANTS: dict[str, list[str]] = {
    "a": ["a", "à", "á", "â", "ã", "ä"],
    "e": ["e", "è", "é", "ê", "ë"],
    "i": ["i", "ì", "í", "î", "ï"],
    "o": ["o", "ò", "ó", "ô", "õ", "ö"],
    "u": ["u", "ù", "ú", "û", "ü"],
}
_VOWELS = [v for variants in _VOWEL_VARIANTS.values() for v in variants]

SPECIAL_TOKENS = ["[PAD]", "[UNK]", "[BOS]", "[EOS]", "[SEP]", "[MASK]"]
PUNCT_TOKENS = list('.,!?;:"-()[]{}')

PAD_ID  = 0
UNK_ID  = 1
BOS_ID  = 2
EOS_ID  = 3
SEP_ID  = 4
MASK_ID = 5


class AlienVocab:
    """Deterministic alien syllable vocabulary.

    Args:
        vocab_size: Total number of tokens (including specials and punct).
        seed: Random seed for syllable ordering.
    """

    def __init__(self, vocab_size: int = 8000, seed: int = 42) -> None:
        self.vocab_size = vocab_size
        self.seed = seed
        self._syllables: list[str] = []
        self._build()

    def _build(self) -> None:
        # Generate syllables in (consonant, vowel) order — leading characters are
        # contiguous in the syllable list. Combined with the cluster-pool cipher,
        # this gives same-cluster English words visually similar alien-word first
        # syllables (sharing leading consonant and vowel), which BPE then captures
        # as shared subword tokens.
        cv  = [c + v for c in _CONSONANTS for v in _VOWELS]
        cvc = [c + v + c2 for c in _CONSONANTS for v in _VOWELS for c2 in _CONSONANTS]
        # Stable order: CV then CVC, both sorted by leading consonant, then vowel,
        # then ending consonant for CVC.
        pool = list(dict.fromkeys(cv + cvc))
        n = self.vocab_size - len(SPECIAL_TOKENS) - len(PUNCT_TOKENS)
        self._syllables = pool[:n]

    @property
    def syllables(self) -> list[str]:
        return list(self._syllables)

    def build_tokenizer(
        self,
        encoded_sentences: list[str],
        vocab_size: int = 32_000,
    ) -> PreTrainedTokenizerFast:
        """Train a BPE tokenizer on cipher-encoded sentences.

        Args:
            encoded_sentences: Cipher-encoded alien text (one sentence per entry),
                               with words already concatenated (e.g. 'hámzog sâznãrùz').
            vocab_size: BPE vocabulary size (controls alien_emb / alien_head dimensions).
        """
        tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
        tokenizer.pre_tokenizer = Whitespace()
        trainer = BpeTrainer(
            vocab_size=vocab_size,
            special_tokens=SPECIAL_TOKENS,
            show_progress=True,
        )
        tokenizer.train_from_iterator(encoded_sentences, trainer=trainer)

        # Remap special token IDs to match our fixed constants.
        # BpeTrainer assigns them in order of the special_tokens list → matches PAD=0..EOS=3.
        tokenizer.post_processor = TemplateProcessing(
            single="$A [EOS]",
            special_tokens=[("[EOS]", tokenizer.token_to_id("[EOS]"))],
        )

        return PreTrainedTokenizerFast(
            tokenizer_object=tokenizer,
            pad_token="[PAD]",
            unk_token="[UNK]",
            bos_token="[BOS]",
            eos_token="[EOS]",
            sep_token="[SEP]",
            mask_token="[MASK]",
        )

    def save(self, path: Path) -> None:
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        data = {"seed": self.seed, "vocab_size": self.vocab_size, "syllables": self._syllables}
        (path / "alien_vocab.json").write_text(json.dumps(data))

    @classmethod
    def load(cls, path: Path) -> AlienVocab:
        data = json.loads((Path(path) / "alien_vocab.json").read_text())
        obj = cls.__new__(cls)
        obj.seed = data["seed"]
        obj.vocab_size = data["vocab_size"]
        obj._syllables = data["syllables"]
        return obj

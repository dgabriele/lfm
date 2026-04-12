"""VAE generator over the Qwen-BPE-stable phoneme alphabet.

Sibling of :class:`~lfm.generator.multilingual_vae.MultilingualVAEGenerator`
that produces surface forms composed of our 50 rare-multilingual-Latin
phonemes (see ``docs/phoneme-alphabet-design.md``) instead of IPA
subwords.  The phoneme alphabet guarantees deterministic tokenization
by Qwen's BPE, preserving the topographic structure of the VAE's
latent space through the downstream interpreter round-trip.

Architecture is inherited unchanged from
:class:`~lfm.generator.vae_base.BaseVAEGenerator` — this module
contributes only the tokenizer wiring and ``decode_to_text``.
"""

from __future__ import annotations

import logging

from torch import Tensor

from lfm._registry import register
from lfm.generator.config import GeneratorConfig
from lfm.generator.phoneme_tokenizer import PhonemeTokenizer
from lfm.generator.vae_base import BaseVAEGenerator

logger = logging.getLogger(__name__)


@register("generator", "phoneme_vae")
class PhonemeVAEGenerator(BaseVAEGenerator):
    """VAE generator with a phoneme-level output alphabet.

    Produces sequences over a 50-phoneme vocabulary drawn from
    non-dominant Latin-script languages (Turkish, Czech, Hungarian,
    Finnish, Polish, Indonesian, Estonian).  Each phoneme is a
    single deterministic Qwen BPE token when space-prefixed, so
    concatenated Neuroglot output survives LLM tokenization intact.
    """

    def _init_tokenizer(self, config: GeneratorConfig) -> None:
        """Install a phoneme tokenizer.

        ``config.spm_model_path`` is repurposed to point at the
        phoneme-alphabet JSON artifact produced by
        ``scripts/design_phoneme_alphabet_multi.py``.  Surface formatting
        follows ``config.phoneme_word_boundary`` and
        ``config.phoneme_word_size``.  When the alphabet path is unset,
        ``decode_to_text`` raises.
        """
        self._tokenizer: PhonemeTokenizer | None = None
        self._phoneme_word_size = config.phoneme_word_size
        if config.spm_model_path is not None:
            self._tokenizer = PhonemeTokenizer(
                config.spm_model_path,
                word_boundary=config.phoneme_word_boundary,
            )
            if self._tokenizer.vocab_size != config.vocab_size:
                logger.warning(
                    "Phoneme alphabet size (%d) != config.vocab_size (%d). "
                    "This is usually a config mismatch.",
                    self._tokenizer.vocab_size, config.vocab_size,
                )

    def decode_to_text(self, token_ids: Tensor) -> list[str]:
        """Decode integer token ids to Neuroglot surface strings.

        Surface format is governed by ``config.phoneme_word_boundary``
        (joiner within a word, e.g. "" for concatenated words or "-" for
        hyphenated display) and ``config.phoneme_word_size`` (phonemes
        per word).

        Args:
            token_ids: Integer tensor of shape ``(batch, seq_len)``.

        Returns:
            List of Neuroglot strings, one per batch element.

        Raises:
            RuntimeError: If no phoneme alphabet was configured.
        """
        if self._tokenizer is None:
            raise RuntimeError(
                "Cannot decode to text: no phoneme alphabet path configured "
                "in GeneratorConfig.spm_model_path.",
            )
        return self._tokenizer.batch_decode(
            token_ids, word_size=self._phoneme_word_size,
        )

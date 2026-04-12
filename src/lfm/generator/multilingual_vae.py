"""Multilingual VAE generator over an IPA/SPM alphabet.

The original LFM phrase-VAE: decodes latent z into IPA phoneme subword
tokens via a sentencepiece vocabulary trained on 12 typologically
diverse human languages.  Produces human-pronounceable output that
looks like a natural phonetic transcription.

All VAE/decoder architecture is inherited from
:class:`~lfm.generator.vae_base.BaseVAEGenerator`; this module
contributes only the SPM tokenizer wiring and ``decode_to_text``.

The sibling :class:`~lfm.generator.phoneme_vae.PhonemeVAEGenerator`
(forthcoming) uses the same architecture with a phoneme-level ASCII
alphabet optimized for stable tokenization by downstream LLMs.
"""

from __future__ import annotations

import logging

from torch import Tensor

from lfm._registry import register
from lfm.generator.config import GeneratorConfig
from lfm.generator.tokenizer import SubwordTokenizer
from lfm.generator.vae_base import BaseVAEGenerator

logger = logging.getLogger(__name__)


@register("generator", "multilingual_vae")
class MultilingualVAEGenerator(BaseVAEGenerator):
    """VAE generator with sentencepiece-over-IPA output alphabet.

    Produces phonetically structured subword token sequences.  Thin
    subclass of :class:`BaseVAEGenerator` — overrides only the
    tokenizer wiring and ``decode_to_text``.
    """

    def _init_tokenizer(self, config: GeneratorConfig) -> None:
        """Install an SPM tokenizer over IPA subwords.

        When ``config.spm_model_path`` is provided, loads the sentencepiece
        model produced during multilingual VAE pretraining.  Without it,
        ``decode_to_text`` will raise — callers that only need the
        architecture (e.g. some tests, raw-token consumers) can omit it.
        """
        self._tokenizer: SubwordTokenizer | None = None
        if config.spm_model_path is not None:
            self._tokenizer = SubwordTokenizer(
                config.spm_model_path, config.vocab_size,
            )

    def decode_to_text(self, token_ids: Tensor) -> list[str]:
        """Decode integer token ids to IPA strings.

        Args:
            token_ids: Integer tensor of shape ``(batch, seq_len)``.

        Returns:
            List of IPA strings, one per batch element.

        Raises:
            RuntimeError: If no sentencepiece model was configured.
        """
        if self._tokenizer is None:
            raise RuntimeError(
                "Cannot decode to text: no spm_model_path configured in "
                "GeneratorConfig.",
            )
        return self._tokenizer.batch_decode(token_ids)

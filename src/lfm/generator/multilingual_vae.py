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

    def render_surface(
        self,
        token_ids: Tensor,
        mask: Tensor | None = None,
        eos_id: int | None = None,
        output_mode: str | None = None,
    ) -> list[str]:
        """Render surface text with optional IPA-specific formatting.

        In addition to the base BOS/EOS/mask handling, applies
        alphabet-specific post-formatting per ``output_mode``:

        * ``"hyphenated_ipa"`` (default): syllable-hyphenate the IPA text,
          exposing phonotactic structure to downstream LLM tokenizers.
        * ``"romanized"``: lossy ASCII romanization (legacy mode).
        * ``"hyphenated_romanized"``: romanized with syllable hyphens.
        * ``"romanized_iso"``: ISO-romanized.
        * ``"raw"`` or ``None``: no post-formatting (raw SPM decode output).

        The translator/corpus generators pass their ``output_mode`` config
        through here so the corpus layer stays alphabet-agnostic.
        """
        raw = super().render_surface(
            token_ids, mask=mask, eos_id=eos_id, output_mode=output_mode,
        )
        if not output_mode or output_mode == "raw":
            return raw
        from lfm.translator.romanize import (
            romanize,
            syllable_hyphenate,
        )
        out: list[str] = []
        for ipa in raw:
            if not ipa:
                out.append(ipa)
                continue
            if output_mode == "hyphenated_ipa":
                out.append(syllable_hyphenate(ipa))
            elif output_mode == "hyphenated_romanized":
                out.append(romanize(syllable_hyphenate(ipa)))
            elif output_mode == "romanized":
                out.append(romanize(ipa))
            elif output_mode == "romanized_iso":
                from lfm.translator.romanize import romanize_iso
                out.append(romanize_iso(ipa))
            else:
                out.append(ipa)
        return out

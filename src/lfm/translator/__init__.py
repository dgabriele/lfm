"""Self-supervised IPA to English translation module.

Trains a small causal LM to translate emergent IPA output back into English,
closing the interpretability loop: agent -> LFM -> IPA -> LLM -> English.

Public API::

    from lfm.translator import (
        PairGenerationConfig,
        TranslatorConfig,
        IPATranslationDataset,
        PairGenerator,
        TranslatorTrainer,
        TranslatorEvaluator,
    )
"""

from lfm.translator.config import PairGenerationConfig, TranslatorConfig
from lfm.translator.dataset import IPATranslationDataset
from lfm.translator.evaluator import TranslatorEvaluator
from lfm.translator.pairs import PairGenerator
from lfm.translator.trainer import TranslatorTrainer

__all__ = [
    "PairGenerationConfig",
    "TranslatorConfig",
    "IPATranslationDataset",
    "PairGenerator",
    "TranslatorTrainer",
    "TranslatorEvaluator",
]

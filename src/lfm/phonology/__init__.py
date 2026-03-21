"""Phonology subsystem for the LFM framework.

Provides the abstract ``PhonologyModule`` base class, ``PhonologyConfig``,
and the ``SurfacePhonology`` implementation which achieves phonotactic
constraint via implicit surface-form smoothness pressures rather than
explicit phonological categories.

The smoothness GRU can optionally be pre-trained on cross-linguistic IPA data
via ``lfm.phonology.priors.pretrain_phonotactic_prior()`` and loaded at init
time by setting ``PhonologyConfig.pretrained_smoothness_path``.
"""

from __future__ import annotations

from lfm.phonology.base import PhonologyModule
from lfm.phonology.config import PhonologyConfig

__all__ = ["PhonologyConfig", "PhonologyModule"]

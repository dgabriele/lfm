"""Parser backend registry — automatic backend selection per language."""

from __future__ import annotations

import logging

from lfm.data.parsers.base import ConstituencyBackend
from lfm.data.parsers.benepar_backend import BENEPAR_MODELS
from lfm.data.parsers.depcon_backend import DEPCON_LANGS
from lfm.data.parsers.stanza_backend import STANZA_LANGS

logger = logging.getLogger(__name__)

# Unified dep→con conversion for all languages. Consistency across all
# 16 training languages is more valuable than higher recall on 7.
# Stanza dependency parsers cover 14/16 languages; swa and tgl
# fall back to full-sentence-only.
_BACKEND_PRIORITY: list[tuple[str, dict[str, str]]] = [
    ("depcon", DEPCON_LANGS),
]


def supported_languages() -> dict[str, str]:
    """Return all supported languages and their backend names.

    Returns:
        Dict of iso3_code → backend_name.
    """
    langs: dict[str, str] = {}
    for backend_name, lang_map in _BACKEND_PRIORITY:
        for iso3 in lang_map:
            if iso3 not in langs:
                langs[iso3] = backend_name
    return langs


def get_backend(
    lang_iso3: str,
    use_gpu: bool = True,
) -> ConstituencyBackend:
    """Get the best available constituency backend for a language.

    Selection priority: Stanza constituency > benepar > dep→con.

    Args:
        lang_iso3: ISO 639-3 language code.
        use_gpu: Whether to use GPU for parsing.

    Returns:
        A ConstituencyBackend instance.

    Raises:
        KeyError: If no backend supports the language.
    """
    # Dispatch based on priority list (not hardcoded order)
    for backend_name, lang_map in _BACKEND_PRIORITY:
        if lang_iso3 in lang_map:
            if backend_name == "stanza":
                from lfm.data.parsers.stanza_backend import StanzaBackend

                return StanzaBackend(lang_iso3, use_gpu=use_gpu)
            elif backend_name == "benepar":
                from lfm.data.parsers.benepar_backend import BeneparBackend

                return BeneparBackend(lang_iso3)
            elif backend_name == "depcon":
                from lfm.data.parsers.depcon_backend import DepConBackend

                return DepConBackend(lang_iso3, use_gpu=use_gpu)

    all_langs = supported_languages()
    raise KeyError(
        f"No constituency backend for {lang_iso3!r}. "
        f"Supported: {sorted(all_langs.keys())}"
    )

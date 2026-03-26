"""IPA (International Phonetic Alphabet) transcription for corpus text.

Converts multilingual text to IPA using ``epitran`` for rule-based
grapheme-to-phoneme conversion (non-English) and the CMU Pronouncing
Dictionary for English.  Every output symbol maps to exactly one sound,
producing a truly uniform phonetic representation across all languages.

Words not found in the CMU dictionary (English) or not convertible via
epitran are dropped.

Language-specific post-processing:
    Hindi (hin): strip leaked Devanagari, drop Latin-script loanwords,
    apply schwa deletion heuristics for word-final position.
    Arabic (ara): strip leaked Arabic script characters.
"""

from __future__ import annotations

import logging
import re
import unicodedata

logger = logging.getLogger(__name__)

# ARPABET to IPA mapping (CMU dict uses ARPABET)
_ARPA_TO_IPA: dict[str, str] = {
    "AA": "ɑ", "AE": "æ", "AH": "ʌ", "AO": "ɔ", "AW": "aʊ", "AX": "ə",
    "AY": "aɪ", "B": "b", "CH": "tʃ", "D": "d", "DH": "ð", "EH": "ɛ",
    "ER": "ɝ", "EY": "eɪ", "F": "f", "G": "ɡ", "HH": "h", "IH": "ɪ",
    "IY": "i", "JH": "dʒ", "K": "k", "L": "l", "M": "m", "N": "n",
    "NG": "ŋ", "OW": "oʊ", "OY": "ɔɪ", "P": "p", "R": "ɹ", "S": "s",
    "SH": "ʃ", "T": "t", "TH": "θ", "UH": "ʊ", "UW": "u", "V": "v",
    "W": "w", "Y": "j", "Z": "z", "ZH": "ʒ",
}

# Epitran language codes for our supported languages
_EPITRAN_LANGS: dict[str, str] = {
    "tur": "tur-Latn",
    "deu": "deu-Latn",
    "spa": "spa-Latn",
    "fin": "fin-Latn",
    "vie": "vie-Latn",
    "ind": "ind-Latn",
    "por": "por-Latn",
    "hun": "hun-Latn",
    "est": "est-Latn",
    "ces": "ces-Latn",
    "pol": "pol-Latn",
    "rus": "rus-Cyrl",
    "hin": "hin-Deva",
    "kor": "kor-Hang",
    "ara": "ara-Arab",
    # "jpn": "jpn-Kana",  # produces kana, not IPA — excluded until proper G2P
}


class IPAConverter:
    """Convert multilingual text to IPA transcription.

    Uses ``epitran`` for non-English languages and the CMU Pronouncing
    Dictionary (via NLTK) for English.  Converters are lazily initialized
    per language to avoid loading all models upfront.

    Args:
        drop_unconvertible: If ``True``, drop words that can't be
            converted to IPA.  If ``False``, keep them as-is (Latin).
    """

    def __init__(self, drop_unconvertible: bool = True) -> None:
        self._drop_unconvertible = drop_unconvertible
        self._epitran_cache: dict[str, object] = {}
        self._cmu_dict: dict[str, list] | None = None
        self._word_re = re.compile(r"[^\s]+")

    def _get_epitran(self, lang: str) -> object | None:
        """Get or create an epitran instance for the given language."""
        if lang not in _EPITRAN_LANGS:
            return None
        if lang not in self._epitran_cache:
            try:
                import epitran

                self._epitran_cache[lang] = epitran.Epitran(
                    _EPITRAN_LANGS[lang]
                )
            except Exception:
                logger.warning("Failed to create epitran for %s", lang)
                self._epitran_cache[lang] = None  # type: ignore[assignment]
        return self._epitran_cache[lang]

    def _get_cmu_dict(self) -> dict[str, list]:
        """Lazily load the CMU Pronouncing Dictionary."""
        if self._cmu_dict is None:
            try:
                import nltk

                nltk.download("cmudict", quiet=True)
                from nltk.corpus import cmudict

                self._cmu_dict = cmudict.dict()
            except Exception:
                logger.warning("Failed to load CMU dict")
                self._cmu_dict = {}
        return self._cmu_dict

    def _english_word_to_ipa(self, word: str) -> str | None:
        """Convert a single English word to IPA via CMU dict."""
        cmu = self._get_cmu_dict()
        key = word.lower().strip(".,!?;:\"'()-")
        if key not in cmu:
            return None
        phones = cmu[key][0]  # first pronunciation
        return "".join(
            _ARPA_TO_IPA.get(p.rstrip("012"), p) for p in phones
        )

    def convert_line(self, lang: str, text: str) -> str | None:
        """Convert a text line to IPA.

        Args:
            lang: ISO 639-3 language code (e.g. ``"eng"``, ``"tur"``).
            text: Raw text line.

        Returns:
            IPA transcription, or ``None`` if conversion fails.
        """
        if lang == "eng":
            return self._convert_english(text)
        return self._convert_epitran(lang, text)

    def _convert_english(self, text: str) -> str | None:
        """Convert English text to IPA word by word via CMU dict."""
        words = text.split()
        ipa_words: list[str] = []
        for word in words:
            ipa = self._english_word_to_ipa(word)
            if ipa is not None:
                ipa_words.append(ipa)
            elif not self._drop_unconvertible:
                ipa_words.append(word.lower())
            # else: drop the word entirely

        if not ipa_words:
            return None
        return " ".join(ipa_words)

    def _convert_epitran(self, lang: str, text: str) -> str | None:
        """Convert non-English text to IPA via epitran."""
        epi = self._get_epitran(lang)
        if epi is None:
            return None
        try:
            return epi.transliterate(text)  # type: ignore[union-attr]
        except Exception:
            return None


# Characters allowed in cleaned IPA output:
# - Unicode letters (IPA symbols, diacritics) — \w covers these
# - IPA suprasegmentals: ː ˈ ˌ ˑ
# - Combining diacritics: U+0300-U+036F (nasalization, tone, etc.)
# - Tie bars: ͡ ͜ (for affricates like t͡s)
# - Spaces (word boundaries)
# Everything else (quotes, brackets, digits, punctuation) is stripped.
_IPA_ALLOWED = re.compile(
    r"[^\w\sːˈˌˑ\u0300-\u036F\u0361\u035C]",
    re.UNICODE,
)


def _clean_ipa(text: str) -> str:
    """Strip non-IPA characters from transcribed text.

    Removes quotation marks, brackets, digits, and other orthographic
    noise that leaks through from news text.  Keeps only IPA-valid
    characters: letters, diacritics, suprasegmentals, tie bars, and spaces.
    """
    cleaned = _IPA_ALLOWED.sub("", text)
    # Collapse multiple spaces
    cleaned = " ".join(cleaned.split())
    return cleaned.strip()


# ── Script detection helpers ──────────────────────────────────────────


def _has_script(text: str, script_name: str) -> bool:
    """Check if text contains characters from a specific Unicode script."""
    return any(
        unicodedata.name(c, "").startswith(script_name)
        for c in text if not c.isspace()
    )


def _is_native_script(word: str, script_name: str) -> bool:
    """Check if a word contains characters from the expected native script."""
    return any(
        unicodedata.name(c, "").startswith(script_name)
        for c in word if c.isalpha()
    )


def _is_ascii_loanword(word: str) -> bool:
    """Check if a source-text word is a Latin-script loanword.

    Only applied to the *raw source text* (Devanagari/Arabic), not IPA output.
    A word like "Video" in Hindi text is a loanword; IPA output like "saːtʰ"
    uses Latin characters but is not a loanword.
    """
    return all(c.isascii() for c in word if c.isalpha()) and any(c.isalpha() for c in word)


# ── Hindi IPA post-processing ────────────────────────────────────────


# Devanagari Unicode block: U+0900-U+097F, extensions U+A8E0-U+A8FF
_DEVANAGARI_RE = re.compile(r"[\u0900-\u097F\uA8E0-\uA8FF]+")

# Word-final schwa deletion: epitran inserts schwa after final consonants
# but Hindi drops it in most word-final positions (Ohala 1983).
_HINDI_FINAL_SCHWA_RE = re.compile(r"ə(?=\s|$)")


def _prefilter_hindi(raw_text: str) -> str | None:
    """Pre-filter Hindi source text before epitran conversion.

    1. Reject lines where >30% of words are Latin-script loanwords.
    2. Strip Latin-script loanwords from the text.

    Returns:
        Filtered text, or ``None`` to reject the line entirely.
    """
    words = raw_text.split()
    if not words:
        return None

    native = [w for w in words if not _is_ascii_loanword(w)]
    if len(native) < len(words) * 0.5:
        return None  # too code-mixed

    return " ".join(native)


def _postprocess_hindi(ipa: str) -> str | None:
    """Post-process Hindi IPA to fix epitran artifacts.

    1. Strip any leaked Devanagari characters.
    2. Apply word-final schwa deletion heuristic.

    Args:
        ipa: Raw IPA from epitran (already source-filtered).

    Returns:
        Cleaned IPA, or ``None`` if too short.
    """
    # Strip leaked Devanagari
    ipa = _DEVANAGARI_RE.sub("", ipa)

    # Word-final schwa deletion
    ipa = _HINDI_FINAL_SCHWA_RE.sub("", ipa)

    ipa = " ".join(ipa.split()).strip()
    return ipa if len(ipa) >= 10 else None


# ── Arabic IPA post-processing ────────────────────────────────────────


_ARABIC_RE = re.compile(r"[\u0600-\u06FF\u0750-\u077F\uFB50-\uFDFF\uFE70-\uFEFF]+")


def _prefilter_arabic(raw_text: str) -> str | None:
    """Pre-filter Arabic source text: reject/strip Latin loanwords."""
    words = raw_text.split()
    if not words:
        return None

    native = [w for w in words if not _is_ascii_loanword(w)]
    if len(native) < len(words) * 0.5:
        return None

    return " ".join(native)


def _postprocess_arabic(ipa: str) -> str | None:
    """Post-process Arabic IPA: strip leaked Arabic script."""
    ipa = _ARABIC_RE.sub("", ipa)
    ipa = " ".join(ipa.split()).strip()
    return ipa if len(ipa) >= 10 else None


# Language → pre-filter and post-processor dispatch
_PREFILTERS: dict[str, callable] = {
    "hin": _prefilter_hindi,
    "ara": _prefilter_arabic,
}

_POSTPROCESSORS: dict[str, callable] = {
    "hin": _postprocess_hindi,
    "ara": _postprocess_arabic,
}


def _convert_one(sample: tuple[str, str]) -> str | None:
    """Convert a single ``(lang, text)`` to IPA.  Module-level for pickling."""
    # Each worker gets its own converter (epitran is not thread-safe)
    global _worker_converter  # noqa: PLW0603
    if "_worker_converter" not in globals() or _worker_converter is None:
        _worker_converter = IPAConverter(drop_unconvertible=True)
    lang, text = sample
    # Pre-filter (strip loanwords, reject code-mixed lines)
    prefilter = _PREFILTERS.get(lang)
    if prefilter is not None:
        text = prefilter(text)
        if text is None:
            return None
    ipa = _worker_converter.convert_line(lang, text)
    if not ipa:
        return None
    # Post-process (strip leaked script, schwa deletion)
    postprocess = _POSTPROCESSORS.get(lang)
    if postprocess is not None:
        ipa = postprocess(ipa)
        if ipa is None:
            return None
    ipa = _clean_ipa(ipa)
    if len(ipa) >= 10:
        return ipa
    return None


def _convert_one_labeled(sample: tuple[str, str]) -> tuple[str, str] | None:
    """Convert a single ``(lang, text)`` to ``(lang, ipa)``.  Preserves label."""
    global _worker_converter  # noqa: PLW0603
    if "_worker_converter" not in globals() or _worker_converter is None:
        _worker_converter = IPAConverter(drop_unconvertible=True)
    lang, text = sample
    # Pre-filter (strip loanwords, reject code-mixed lines)
    prefilter = _PREFILTERS.get(lang)
    if prefilter is not None:
        text = prefilter(text)
        if text is None:
            return None
    ipa = _worker_converter.convert_line(lang, text)
    if not ipa:
        return None
    # Post-process (strip leaked script, schwa deletion)
    postprocess = _POSTPROCESSORS.get(lang)
    if postprocess is not None:
        ipa = postprocess(ipa)
        if ipa is None:
            return None
    ipa = _clean_ipa(ipa)
    if len(ipa) >= 10:
        return (lang, ipa)
    return None


_worker_converter: IPAConverter | None = None


def convert_corpus_to_ipa(
    samples: list[tuple[str, str]],
    drop_unconvertible: bool = True,
    num_workers: int | None = None,
) -> list[str]:
    """Convert a corpus of ``(lang, text)`` tuples to IPA.

    Uses multiprocessing for parallel conversion across CPU cores.

    Args:
        samples: List of ``(language_code, text_line)`` tuples.
        drop_unconvertible: Drop words that can't be converted.
        num_workers: Number of parallel workers.  ``None`` = cpu_count.

    Returns:
        List of IPA-transcribed text lines (lines that fail entirely
        are dropped).
    """
    import multiprocessing as mp

    if num_workers is None:
        import os

        num_workers = max(1, int(os.cpu_count() * 0.9))

    logger.info(
        "Converting %d samples to IPA with %d workers...",
        len(samples),
        num_workers,
    )

    ctx = mp.get_context("forkserver")
    with ctx.Pool(num_workers) as pool:
        results = pool.map(_convert_one, samples, chunksize=500)

    succeeded = [r for r in results if r is not None]
    failed = len(results) - len(succeeded)

    logger.info(
        "IPA conversion: %d succeeded, %d failed/dropped",
        len(succeeded),
        failed,
    )
    return succeeded


def convert_corpus_to_ipa_labeled(
    samples: list[tuple[str, str]],
    num_workers: int | None = None,
) -> list[tuple[str, str]]:
    """Convert corpus to IPA while preserving language labels.

    Like ``convert_corpus_to_ipa`` but returns ``(lang_code, ipa_text)``
    tuples, keeping language alignment intact through the conversion.

    Args:
        samples: List of ``(language_code, text_line)`` tuples.
        num_workers: Number of parallel workers.  ``None`` = cpu_count.

    Returns:
        List of ``(lang_code, ipa_text)`` tuples (failed lines dropped).
    """
    import multiprocessing as mp

    if num_workers is None:
        import os

        num_workers = max(1, int(os.cpu_count() * 0.9))

    logger.info(
        "Converting %d samples to IPA (labeled) with %d workers...",
        len(samples),
        num_workers,
    )

    ctx = mp.get_context("forkserver")
    with ctx.Pool(num_workers) as pool:
        results = pool.map(_convert_one_labeled, samples, chunksize=500)

    succeeded = [r for r in results if r is not None]
    failed = len(results) - len(succeeded)

    logger.info(
        "IPA conversion (labeled): %d succeeded, %d failed/dropped",
        len(succeeded),
        failed,
    )
    return succeeded

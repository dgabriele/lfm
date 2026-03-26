"""Text sanitization for multilingual corpus preprocessing.

Configurable rule-based filters for cleaning raw text before IPA conversion.
Based on established multilingual corpus preprocessing best practices
(CCNet, mC4, OSCAR, RedPajama).

Extracted from ``generator.pretrain`` — the pretraining pipeline imports
from here rather than defining sanitization inline.
"""

from __future__ import annotations

import logging
import multiprocessing as mp
import os
import re
import unicodedata
from typing import NamedTuple

from cleantext import clean

from lfm.config.base import LFMBaseConfig

logger = logging.getLogger(__name__)

# Greek/math symbols → English name mapping (standard TTS preprocessing)
_GREEK_NAMES: dict[str, str] = {
    "α": "alpha", "β": "beta", "γ": "gamma", "δ": "delta", "ε": "epsilon",
    "ζ": "zeta", "η": "eta", "θ": "theta", "ι": "iota", "κ": "kappa",
    "λ": "lambda", "μ": "mu", "ν": "nu", "ξ": "xi", "ο": "omicron",
    "π": "pi", "ρ": "rho", "σ": "sigma", "ς": "sigma", "τ": "tau",
    "υ": "upsilon", "φ": "phi", "χ": "chi", "ψ": "psi", "ω": "omega",
    "Α": "Alpha", "Β": "Beta", "Γ": "Gamma", "Δ": "Delta", "Ε": "Epsilon",
    "Ζ": "Zeta", "Η": "Eta", "Θ": "Theta", "Ι": "Iota", "Κ": "Kappa",
    "Λ": "Lambda", "Μ": "Mu", "Ν": "Nu", "Ξ": "Xi", "Ο": "Omicron",
    "Π": "Pi", "Ρ": "Rho", "Σ": "Sigma", "Τ": "Tau", "Υ": "Upsilon",
    "Φ": "Phi", "Χ": "Chi", "Ψ": "Psi", "Ω": "Omega",
    "∞": "infinity", "∑": "sum", "∏": "product", "√": "square root",
    "∂": "partial", "∫": "integral", "∇": "nabla", "±": "plus minus",
    "≈": "approximately", "≠": "not equal", "≤": "less or equal",
    "≥": "greater or equal",
}

# Regex for Greek/math symbols
_GREEK_RE = re.compile(
    "[" + re.escape("".join(_GREEK_NAMES.keys())) + "]"
)

# Terminal punctuation characters (multilingual)
_TERMINAL_PUNCT = set(".!?।؟。！？")

# Script detection: map ISO 639-3 → primary Unicode script name prefix.
# Used by max_foreign_script_ratio to detect code-switching.
_LANG_SCRIPTS: dict[str, str] = {
    "eng": "LATIN", "deu": "LATIN", "spa": "LATIN", "por": "LATIN",
    "tur": "LATIN", "fin": "LATIN", "hun": "LATIN", "est": "LATIN",
    "ces": "LATIN", "pol": "LATIN", "ind": "LATIN", "vie": "LATIN",
    "rus": "CYRILLIC", "hin": "DEVANAGARI", "ara": "ARABIC",
    "kor": "HANGUL",
}


class SanitizeConfig(LFMBaseConfig):
    """Configurable text sanitization filters.

    Based on established multilingual corpus preprocessing best practices
    (CCNet, mC4, OSCAR, RedPajama).
    """

    # Length bounds
    min_line_length: int = 20
    max_line_length: int = 500
    min_word_count: int = 3

    # Character composition
    alpha_ratio_min: float = 0.7
    max_digit_ratio: float = 0.0

    # Code-switching / script purity
    max_foreign_script_ratio: float = 0.3

    # Repetition detection (degenerate text)
    max_word_repetition_ratio: float = 0.5
    max_bigram_repetition_ratio: float = 0.4

    # Content stripping
    strip_urls: bool = True
    strip_emails: bool = True
    strip_phone_numbers: bool = True
    require_terminal_punctuation: bool = True

    # Number handling: "reject", "strip", "keep", "spell_out"
    number_policy: str = "spell_out"

    # Greek/math symbol handling: "reject", "strip", "keep", "spell_out"
    symbol_policy: str = "spell_out"

    # IPA post-conversion quality
    min_ipa_length: int = 10
    max_ipa_non_alpha_ratio: float = 0.3


class RejectedSample(NamedTuple):
    """A sample that was rejected during sanitization, with reason."""

    language: str
    text: str
    reason: str


def _spell_out_number(match: re.Match, lang: str = "en") -> str:
    """Convert a matched number to words via num2words."""
    try:
        from num2words import num2words
    except ImportError:
        return ""  # strip if num2words unavailable

    text = match.group(0)
    # Strip thousands separators
    text = text.replace(",", "")

    try:
        val = float(text)
    except ValueError:
        return ""

    # Cap: only spell out integers < 10000 and simple decimals
    if val != int(val) and len(text.split(".")[-1]) > 2:
        return ""  # complex decimal — strip
    if abs(val) >= 10000:
        return ""  # too large — strip

    try:
        return num2words(val, lang=lang)
    except (NotImplementedError, OverflowError):
        return ""


def _apply_number_policy(text: str, policy: str, lang: str = "en") -> str | None:
    """Apply the configured number policy to text.

    Returns None to reject, or the processed text.
    """
    if policy == "keep":
        return text

    has_digits = bool(re.search(r"\d", text))
    if not has_digits:
        return text

    if policy == "reject":
        return None

    if policy == "strip":
        # Remove digit sequences (and surrounding punctuation like 1,000 or 3.14)
        text = re.sub(r"\d[\d,.]*\d|\d", "", text)
        return " ".join(text.split()) or None

    if policy == "spell_out":
        # Map ISO 639-3 to num2words language codes
        lang_map = {
            "eng": "en", "deu": "de", "spa": "es", "por": "pt",
            "tur": "tr", "fin": "fi", "hun": "hu", "ces": "cz",
            "pol": "pl", "rus": "ru", "hin": "hi", "ara": "ar",
            "ind": "id", "vie": "vi", "est": "en", "kor": "ko",
        }
        n2w_lang = lang_map.get(lang, "en")
        text = re.sub(
            r"\d[\d,.]*\d|\d",
            lambda m: _spell_out_number(m, n2w_lang),
            text,
        )
        return " ".join(text.split()) or None

    return text  # unknown policy — pass through


def _apply_symbol_policy(text: str, policy: str) -> str | None:
    """Apply the configured Greek/math symbol policy."""
    if policy == "keep":
        return text

    has_symbols = bool(_GREEK_RE.search(text))
    if not has_symbols:
        return text

    if policy == "reject":
        return None

    if policy == "strip":
        text = _GREEK_RE.sub("", text)
        return " ".join(text.split()) or None

    if policy == "spell_out":
        text = _GREEK_RE.sub(lambda m: " " + _GREEK_NAMES.get(m.group(), "") + " ", text)
        return " ".join(text.split()) or None

    return text


def _check_foreign_script_ratio(text: str, lang: str, max_ratio: float) -> bool:
    """Return True if the text passes the foreign script ratio check."""
    script = _LANG_SCRIPTS.get(lang)
    if script is None:
        return True  # unknown language — skip check

    words = text.split()
    if not words:
        return False

    foreign = 0
    for word in words:
        # A word is "foreign" if it has alpha chars but none from the native script
        alpha_chars = [c for c in word if c.isalpha()]
        if alpha_chars and not any(
            unicodedata.name(c, "").startswith(script) for c in alpha_chars
        ):
            foreign += 1

    return (foreign / len(words)) <= max_ratio


def _check_repetition(text: str, max_word_ratio: float, max_bigram_ratio: float) -> bool:
    """Return True if text passes repetition checks."""
    words = text.lower().split()
    if len(words) < 4:
        return True  # too short to judge

    # Word repetition
    unique_words = set(words)
    word_rep_ratio = 1.0 - (len(unique_words) / len(words))
    if word_rep_ratio > max_word_ratio:
        return False

    # Bigram repetition
    if len(words) >= 2:
        bigrams = [(words[i], words[i + 1]) for i in range(len(words) - 1)]
        unique_bigrams = set(bigrams)
        bigram_rep_ratio = 1.0 - (len(unique_bigrams) / len(bigrams))
        if bigram_rep_ratio > max_bigram_ratio:
            return False

    return True


# Module-level config for multiprocessing workers
_worker_config: SanitizeConfig | None = None


def _init_worker(cfg: SanitizeConfig) -> None:
    """Initialize the worker-local sanitize config."""
    global _worker_config  # noqa: PLW0603
    _worker_config = cfg


def sanitize_one(
    sample: tuple[str, str],
    cfg: SanitizeConfig | None = None,
) -> tuple[str, str] | None:
    """Sanitize a single ``(lang, text)`` sample.

    Returns the cleaned ``(lang, text)`` or ``None`` to reject.
    Uses the module-level ``_worker_config`` when ``cfg`` is not provided
    (for multiprocessing compatibility).
    """
    config = cfg or _worker_config
    if config is None:
        raise RuntimeError("No SanitizeConfig provided and no worker config set")

    lang, line = sample

    # 1. Apply number policy
    result = _apply_number_policy(line, config.number_policy, lang)
    if result is None:
        return None
    line = result

    # 2. Apply symbol policy
    result = _apply_symbol_policy(line, config.symbol_policy)
    if result is None:
        return None
    line = result

    # 3. Clean with clean-text library
    line = clean(
        line,
        fix_unicode=True,
        to_ascii=False,
        lower=False,
        no_urls=config.strip_urls,
        no_emails=config.strip_emails,
        no_phone_numbers=config.strip_phone_numbers,
        no_numbers=False,
        no_digits=False,
        no_currency_symbols=True,
        no_punct=False,
    )

    # 4. Remove placeholder tokens
    line = line.replace("<NUMBER>", "").replace("<EMAIL>", "")
    line = line.replace("<URL>", "").replace("<PHONE>", "")
    line = " ".join(line.split())

    # 5. Length check
    if not line or len(line) < config.min_line_length:
        return None
    if len(line) > config.max_line_length:
        line = line[: config.max_line_length]

    # 6. Word count check
    words = line.split()
    if len(words) < config.min_word_count:
        return None

    # 7. Alpha ratio (letters + combining marks)
    ling_chars = sum(
        1 for c in line
        if c.isalpha() or unicodedata.category(c).startswith("M")
    )
    alpha_ratio = ling_chars / max(len(line), 1)
    if alpha_ratio < config.alpha_ratio_min:
        return None

    # 8. Digit ratio check
    if config.max_digit_ratio < 1.0:
        digit_count = sum(1 for c in line if c.isdigit())
        digit_ratio = digit_count / max(len(line), 1)
        if digit_ratio > config.max_digit_ratio:
            return None

    # 9. Foreign script ratio
    if not _check_foreign_script_ratio(line, lang, config.max_foreign_script_ratio):
        return None

    # 10. Repetition detection
    if not _check_repetition(
        line, config.max_word_repetition_ratio, config.max_bigram_repetition_ratio
    ):
        return None

    # 11. Terminal punctuation
    if config.require_terminal_punctuation:
        if not line or line[-1] not in _TERMINAL_PUNCT:
            return None

    return (lang, line)


def _sanitize_one_worker(sample: tuple[str, str]) -> tuple[str, str] | None:
    """Multiprocessing wrapper — uses module-level _worker_config."""
    return sanitize_one(sample)


def sanitize_samples(
    samples: list[tuple[str, str]],
    cfg: SanitizeConfig | None = None,
    num_workers: int | None = None,
) -> list[tuple[str, str]]:
    """Sanitize corpus samples using multiprocessing.

    Args:
        samples: List of ``(lang, text)`` tuples.
        cfg: Sanitization config. If ``None``, uses default config.
        num_workers: Number of parallel workers. ``None`` = auto (90% of cores).

    Returns:
        Filtered list of ``(lang, text)`` tuples.
    """
    if cfg is None:
        cfg = SanitizeConfig()

    if num_workers is None:
        num_workers = max(1, int(os.cpu_count() * 0.9))

    logger.info(
        "Sanitizing %d samples with %d workers (policy: numbers=%s, symbols=%s)...",
        len(samples), num_workers, cfg.number_policy, cfg.symbol_policy,
    )

    with mp.Pool(num_workers, initializer=_init_worker, initargs=(cfg,)) as pool:
        results = pool.map(_sanitize_one_worker, samples, chunksize=1000)

    accepted = [r for r in results if r is not None]
    logger.info(
        "Sanitization: %d accepted, %d rejected (of %d)",
        len(accepted), len(samples) - len(accepted), len(samples),
    )
    return accepted


def sanitize_samples_detailed(
    samples: list[tuple[str, str]],
    cfg: SanitizeConfig | None = None,
    num_workers: int | None = None,
) -> tuple[list[tuple[str, str]], list[RejectedSample]]:
    """Sanitize with rejection tracking using multiprocessing.

    Runs ``sanitize_one`` in parallel, then separates accepted from
    rejected samples.  Rejection reasons are coarse ("sanitize_filter")
    since the parallel path doesn't track which specific filter rejected.

    Returns:
        Tuple of (accepted samples, rejected samples with reasons).
    """
    if cfg is None:
        cfg = SanitizeConfig()

    if num_workers is None:
        num_workers = max(1, int(os.cpu_count() * 0.9))

    logger.info(
        "Sanitizing %d samples (detailed) with %d workers...",
        len(samples), num_workers,
    )

    with mp.Pool(num_workers, initializer=_init_worker, initargs=(cfg,)) as pool:
        results = pool.map(_sanitize_one_worker, samples, chunksize=1000)

    accepted: list[tuple[str, str]] = []
    rejected: list[RejectedSample] = []

    for sample, result in zip(samples, results):
        if result is not None:
            accepted.append(result)
        else:
            rejected.append(RejectedSample(sample[0], sample[1], "sanitize_filter"))

    logger.info(
        "Sanitization (detailed): %d accepted, %d rejected (of %d)",
        len(accepted), len(rejected), len(samples),
    )
    return accepted, rejected

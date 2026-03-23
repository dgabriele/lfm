"""Language metadata for the 16 Leipzig corpus languages.

Maps ISO 639-3 codes to display names, family, and morphological type.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class LanguageInfo:
    code: str
    name: str
    family: str
    morph_type: str  # fusional, agglutinative, isolating, introflexive


# The 16 languages used in VAE pretraining
LANGUAGES: dict[str, LanguageInfo] = {
    "ara": LanguageInfo("ara", "Arabic", "Afro-Asiatic", "introflexive"),
    "ces": LanguageInfo("ces", "Czech", "Indo-European", "fusional"),
    "deu": LanguageInfo("deu", "German", "Indo-European", "fusional"),
    "eng": LanguageInfo("eng", "English", "Indo-European", "fusional"),
    "est": LanguageInfo("est", "Estonian", "Uralic", "agglutinative"),
    "fin": LanguageInfo("fin", "Finnish", "Uralic", "agglutinative"),
    "hin": LanguageInfo("hin", "Hindi", "Indo-European", "fusional"),
    "hun": LanguageInfo("hun", "Hungarian", "Uralic", "agglutinative"),
    "ind": LanguageInfo("ind", "Indonesian", "Austronesian", "isolating"),
    "kor": LanguageInfo("kor", "Korean", "Koreanic", "agglutinative"),
    "pol": LanguageInfo("pol", "Polish", "Indo-European", "fusional"),
    "por": LanguageInfo("por", "Portuguese", "Indo-European", "fusional"),
    "rus": LanguageInfo("rus", "Russian", "Indo-European", "fusional"),
    "spa": LanguageInfo("spa", "Spanish", "Indo-European", "fusional"),
    "tur": LanguageInfo("tur", "Turkish", "Turkic", "agglutinative"),
    "vie": LanguageInfo("vie", "Vietnamese", "Austroasiatic", "isolating"),
}

# Unique families (for color mapping)
FAMILIES: list[str] = sorted({lang.family for lang in LANGUAGES.values()})

# Unique morphological types
MORPH_TYPES: list[str] = sorted({lang.morph_type for lang in LANGUAGES.values()})


def get_label(code: str, by: str = "language") -> str:
    """Get display label for a language code.

    Args:
        code: ISO 639-3 language code.
        by: One of ``"language"``, ``"family"``, ``"type"``.

    Returns:
        Human-readable label string.
    """
    info = LANGUAGES.get(code)
    if info is None:
        return code
    if by == "family":
        return info.family
    if by == "type":
        return info.morph_type
    return info.name


def get_color_groups(by: str = "language") -> dict[str, list[str]]:
    """Group language codes by the specified attribute.

    Args:
        by: One of ``"language"``, ``"family"``, ``"type"``.

    Returns:
        Dict mapping group label to list of language codes.
    """
    from collections import defaultdict

    groups: dict[str, list[str]] = defaultdict(list)
    for code, info in LANGUAGES.items():
        label = get_label(code, by)
        groups[label].append(code)
    return dict(groups)

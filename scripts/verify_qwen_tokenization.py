#!/usr/bin/env python
"""Verify Qwen tokenizes our phoneme-VAE output as designed.

Feeds actual diagnostic samples from the remote training log (both
originals and reconstructions) through Qwen's BPE tokenizer and checks:

  * Each phoneme resolves to a single BPE token (space-prefixed at word
    starts, bare elsewhere).
  * Word boundaries are preserved (space between words = expected token
    split).
  * Total tokens per sample matches the expected phoneme count within
    a small tolerance.

If any sample fragments unexpectedly, the design intent is broken —
this is a fast local check we can run without GPU.
"""

from __future__ import annotations

import json
from pathlib import Path

from transformers import AutoTokenizer

MODEL = "Qwen/Qwen2.5-0.5B"
ALPHABET_PATH = Path("data/phoneme_alphabet_multi.json")

# Actual samples pulled from remote training log (step ~5300).
# Two rendering variants of each: concatenated (no hyphens within words) and
# hyphenated (phonemes separated by '-' within words, spaces between words).
SAMPLES_CONCAT = [
    ("eng-orig",  "plyuyukugjak pszaby ilkottustjavukug wynozottell jakhudopr "
                  "fonuzellukugopropp uyivottopp wikuyabyell pszaby oproppuvuyop"),
    ("deu-orig",  "ottellochogrvak ilkjavogrustochogrvak oproppjanogrvakochogrith "
                  "ustochogrith oprbizjavuzwikuzugopr"),
]

# Hyphenated: same content, but phonemes separated by '-' within each word.
SAMPLES_HYPHEN = [
    ("eng-orig",  "ply-uy-uk-ug-jak psz-aby ilk-ott-ust-jav-uk-ug wyn-oz-ott-ell "
                  "jak-hud-opr fon-uz-ell-uk-ug-opr-opp uy-iv-ott-opp wik-uy-aby-ell "
                  "psz-aby opr-opp-uv-uy-op"),
    ("deu-orig",  "ott-ell-och-ogr-vak ilk-jav-ogr-ust-och-ogr-vak "
                  "opr-opp-jan-ogr-vak-och-ogr-ith ust-och-ogr-ith "
                  "opr-biz-jav-uz-wik-uz-ug-opr"),
]
# Try a few other separators too — curious which ones avoid Qwen's
# per-separator merge surprises.  Pipe, underscore, comma.
def _rejoin(concat_sample: str, sep: str) -> str:
    parts = []
    for word in concat_sample.split():
        # Greedy-match phonemes in concatenated word
        phs = split_into_phonemes(word, _PHONEMES_GLOBAL)
        parts.append(sep.join(phs) if phs else word)
    return " ".join(parts)


_PHONEMES_GLOBAL: list[str] = []


SAMPLES = [("CONCAT/" + n, s) for n, s in SAMPLES_CONCAT] + \
          [("HYPHEN/" + n, s) for n, s in SAMPLES_HYPHEN]


def split_into_phonemes(word: str, phonemes: list[str]) -> list[str] | None:
    """Greedy longest-match split. Returns list of phonemes or None if any
    char can't be matched."""
    phonemes_by_len = sorted(phonemes, key=len, reverse=True)
    out: list[str] = []
    i = 0
    while i < len(word):
        matched = False
        for p in phonemes_by_len:
            if word.startswith(p, i):
                out.append(p)
                i += len(p)
                matched = True
                break
        if not matched:
            return None
    return out


def main() -> None:
    print(f"Loading tokenizer: {MODEL}")
    tok = AutoTokenizer.from_pretrained(MODEL)
    alphabet = json.loads(ALPHABET_PATH.read_text())
    phonemes: list[str] = alphabet["phonemes"]
    phoneme_set = set(phonemes)
    global _PHONEMES_GLOBAL
    _PHONEMES_GLOBAL = phonemes

    # Add separator comparison for the deu sample
    deu_src = SAMPLES_CONCAT[1][1]
    for sep in ["_", "|", ".", ":"]:
        rejoined = _rejoin(deu_src, sep)
        ids = tok.encode(" " + rejoined, add_special_tokens=False)
        pieces = tok.convert_ids_to_tokens(ids)
        # count how many pieces are known phonemes (stripping Ġ and sep prefixes)
        clean_count = 0
        for p in pieces:
            core = p.lstrip("Ġ").lstrip(sep)
            if core in phoneme_set:
                clean_count += 1
        print(f"  SEP={sep!r}: {len(pieces)} tokens, {clean_count} clean phoneme tokens")
        print(f"    sample tokens: {pieces[:20]}")

    # Single-token ids for each phoneme (space-prefixed and bare forms).
    ph_sp_ids = {p: tok.encode(" " + p, add_special_tokens=False) for p in phonemes}
    ph_bare_ids = {p: tok.encode(p, add_special_tokens=False) for p in phonemes}
    all_single_sp = all(len(v) == 1 for v in ph_sp_ids.values())
    all_single_bare = all(len(v) == 1 for v in ph_bare_ids.values())
    print(f"  all phonemes single-token when space-prefixed: {all_single_sp}")
    print(f"  all phonemes single-token when bare:           {all_single_bare}")

    for name, sample in SAMPLES:
        print(f"\n=== {name} ===")
        print(f"  surface: {sample[:120]}{'...' if len(sample) > 120 else ''}")

        # Expected phoneme structure: if hyphenated, split by both spaces and '-';
        # if concatenated, greedy-match within each space-separated word.
        if "-" in sample:
            words = [w for w in sample.replace("-", " ").split() if w]
            expected_total_phonemes = len(words)
            unparseable = sum(1 for w in words if w not in phoneme_set)
            # In hyphenated mode, each '-' also becomes one or more Qwen tokens
            hyphen_count = sample.count("-")
            print(f"  phonemes: {expected_total_phonemes}   hyphens: {hyphen_count}   "
                  f"unrecognized: {unparseable}")
        else:
            words = sample.split()
            expected_total_phonemes = 0
            unparseable = 0
            for w in words:
                parts = split_into_phonemes(w, phonemes)
                if parts is None:
                    unparseable += 1
                else:
                    expected_total_phonemes += len(parts)
            print(f"  words: {len(words)}   expected phonemes: {expected_total_phonemes}   "
                  f"unparseable words: {unparseable}")

        # Qwen tokenization
        ids = tok.encode(" " + sample, add_special_tokens=False)
        pieces = tok.convert_ids_to_tokens(ids)
        print(f"  qwen tokens: {len(ids)}")
        print(f"  first 20 tokens: {pieces[:20]}")
        if len(pieces) > 20:
            print(f"  last  10 tokens: {pieces[-10:]}")

        # Check each piece maps to a known phoneme (stripping Ġ if present)
        unknown = 0
        mismatched = []
        for p in pieces:
            clean = p[1:] if p.startswith("Ġ") else p
            if clean not in phoneme_set:
                unknown += 1
                if len(mismatched) < 5:
                    mismatched.append(p)
        print(f"  unknown-phoneme tokens: {unknown}   examples: {mismatched}")

        # Deviation from expected count
        deviation = len(ids) - expected_total_phonemes
        status = "✓ clean" if deviation == 0 and unknown == 0 else "✗ mismatch"
        print(f"  token count - expected: {deviation:+d}   status: {status}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python
"""Empirically compare v1 and v2 phoneme alphabets on Qwen tokenization.

For each alphabet, generate 500 synthetic Neuroglot sentences (length
sampled from empirical IPA word-length stats), render with hyphen-within-
word + space-between-words, and feed through Qwen BPE.

Report per-alphabet:

  * Fraction of samples where **every** phoneme preserves its own Qwen
    token (strict per-sample pass).
  * Mean per-sample phoneme-preservation ratio.
  * Per-phoneme preservation rate (which ones still fail).

This tells us whether the v2 chain gate actually eliminates the bad
tokenization we saw in v1 samples.
"""

from __future__ import annotations

import json
import random
import statistics
import unicodedata
from pathlib import Path

from transformers import AutoTokenizer

MODEL = "Qwen/Qwen2.5-0.5B"
V1_ALPHABET = Path("data/phoneme_alphabet_multi.json")
V2_ALPHABET = Path("data/phoneme_alphabet_multi_v2.json")
SEP = "-"
N_SAMPLES = 500
WORDS_PER_SAMPLE = 6
SEED = 42


def load_phonemes(path: Path) -> list[str]:
    return json.loads(path.read_text())["phonemes"]


def render_sample(phonemes: list[str], word_lens: tuple[int, int], rng: random.Random) -> tuple[list[list[str]], str]:
    """Returns (words-as-phoneme-lists, rendered-surface-string)."""
    lo, hi = word_lens
    words = []
    for _ in range(WORDS_PER_SAMPLE):
        L = rng.randint(lo, hi)
        words.append([rng.choice(phonemes) for _ in range(L)])
    rendered = " ".join(SEP.join(w) for w in words)
    return words, rendered


def score_alphabet(tok, phonemes: list[str], word_lens: tuple[int, int], name: str) -> dict:
    rng = random.Random(SEED)
    per_sample_frac: list[float] = []
    perfect = 0
    per_phoneme_pass = {p: {"attempts": 0, "passes": 0} for p in phonemes}

    for _ in range(N_SAMPLES):
        words, rendered = render_sample(phonemes, word_lens, rng)
        ids = tok.encode(" " + rendered, add_special_tokens=False)
        pieces = tok.convert_ids_to_tokens(ids)
        # Walk each word and check preservation of each phoneme
        total_phonemes = sum(len(w) for w in words)
        total_preserved = 0
        # We check by set-membership: each phoneme should appear as its
        # own piece (Ġp for word-initial, bare p elsewhere).  This mirrors
        # the designer's chain-stability probe.
        for word in words:
            for pos, p in enumerate(word):
                per_phoneme_pass[p]["attempts"] += 1
                needle = ("\u0120" + p) if pos == 0 else p
                if needle in pieces:
                    total_preserved += 1
                    per_phoneme_pass[p]["passes"] += 1
        frac = total_preserved / max(total_phonemes, 1)
        per_sample_frac.append(frac)
        if total_preserved == total_phonemes:
            perfect += 1

    mean_frac = statistics.fmean(per_sample_frac)
    worst_phonemes = sorted(
        [(p, s["passes"] / max(s["attempts"], 1)) for p, s in per_phoneme_pass.items()],
        key=lambda x: x[1],
    )[:10]
    return {
        "name": name,
        "n_samples": N_SAMPLES,
        "n_phonemes": len(phonemes),
        "perfect_samples": perfect,
        "perfect_pct": 100 * perfect / N_SAMPLES,
        "mean_phoneme_preservation": mean_frac,
        "worst_phonemes": worst_phonemes,
    }


def main() -> None:
    tok = AutoTokenizer.from_pretrained(MODEL)

    # Empirical word-length stats baked into v2 artifact
    v2_meta = json.loads(V2_ALPHABET.read_text())
    lo = v2_meta["empirical_word_length"]["lo_2sigma"]
    hi = v2_meta["empirical_word_length"]["hi_2sigma"]
    print(f"Word-length range (from v2 artifact empirical stats): [{lo}, {hi}]")
    print(f"Samples: {N_SAMPLES} × {WORDS_PER_SAMPLE} words, separator={SEP!r}\n")

    v1 = load_phonemes(V1_ALPHABET)
    v2 = load_phonemes(V2_ALPHABET)

    r1 = score_alphabet(tok, v1, (lo, hi), "v1 (original)")
    r2 = score_alphabet(tok, v2, (lo, hi), "v2 (chain-gated)")

    print(f"{'metric':<40} {'v1':>18} {'v2':>18}")
    print(f"{'samples with ALL phonemes preserved':<40} {r1['perfect_samples']:>8} / {r1['n_samples']} ({r1['perfect_pct']:.1f}%){r2['perfect_samples']:>8} / {r2['n_samples']} ({r2['perfect_pct']:.1f}%)")
    print(f"{'mean per-sample preservation ratio':<40} {r1['mean_phoneme_preservation']:>18.3f} {r2['mean_phoneme_preservation']:>18.3f}")
    print()
    print("v1 worst 10 phonemes (pass rate):")
    for p, r in r1["worst_phonemes"]:
        print(f"  {p!r:>7} {r:.2f}")
    print()
    print("v2 worst 10 phonemes (pass rate):")
    for p, r in r2["worst_phonemes"]:
        print(f"  {p!r:>7} {r:.2f}")


if __name__ == "__main__":
    main()

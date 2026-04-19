"""Build p2g training data from the v15 IPA paired corpus + CMU dict.

Extracts word-level (IPA, English) pairs from the sentence-aligned
corpus, merges with CMU dict pairs, deduplicates, and writes HDF5
with frequency weights for inverse-frequency-weighted loss.

Improvements over v11:
  - 634K+ pairs (vs 122K from CMU alone)
  - Frequency weights for rare-word emphasis
  - Max length 32 (vs 24)
  - Filters non-IPA words (raw English that leaked through epitran)
  - Noise-augmented pairs for neologism robustness
"""

from __future__ import annotations

import argparse
import random
import re
from collections import Counter
from pathlib import Path

import h5py
import numpy as np


_IPA_CHARS = set("aɑæbdðeɛəɝfɡhiɪjklmnŋoɔpɹsʃtθuʊʌvwzʒ-")


def _is_ipa(word: str) -> bool:
    """Check if a word consists of IPA characters."""
    return all(c in _IPA_CHARS for c in word)


def _is_english(word: str) -> bool:
    """Check if a word is valid English spelling (a-z + hyphens)."""
    return bool(re.match(r'^[a-z][-a-z]*[a-z]$', word)) and len(word) >= 2


# Noise augmentation: perturb IPA to simulate decoder neologisms
_VOWELS = list("aɑæeɛəɝiɪoɔuʊʌ")
_CONSONANTS = list("bdfɡhjklmnŋpɹsʃtθvwzʒð")


def _augment_ipa(ipa: str, rng: random.Random) -> str | None:
    """Random phonetic perturbation of an IPA word."""
    chars = list(ipa)
    if len(chars) < 3:
        return None

    op = rng.choice(["swap_vowel", "swap_consonant", "drop", "insert"])

    if op == "swap_vowel":
        vowel_pos = [i for i, c in enumerate(chars) if c in _VOWELS]
        if not vowel_pos:
            return None
        pos = rng.choice(vowel_pos)
        chars[pos] = rng.choice(_VOWELS)
    elif op == "swap_consonant":
        cons_pos = [i for i, c in enumerate(chars) if c in _CONSONANTS]
        if not cons_pos:
            return None
        pos = rng.choice(cons_pos)
        chars[pos] = rng.choice(_CONSONANTS)
    elif op == "drop":
        pos = rng.randint(1, len(chars) - 2)
        chars.pop(pos)
    elif op == "insert":
        pos = rng.randint(1, len(chars) - 1)
        chars.insert(pos, rng.choice(_VOWELS + _CONSONANTS))

    return "".join(chars)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--paired-corpus",
                    default="data/datasets/english-sentences-v15/ipa_sentences.paired.txt")
    ap.add_argument("--output-dir", default="data/datasets/p2g_v15")
    ap.add_argument("--max-ipa-len", type=int, default=32)
    ap.add_argument("--max-spelling-len", type=int, default=32)
    ap.add_argument("--augment-ratio", type=float, default=0.2,
                    help="Fraction of augmented (noise-perturbed) pairs to add")
    ap.add_argument("--val-fraction", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    rng = random.Random(args.seed)

    # --- Load word pairs from sentence corpus ---
    print("Loading word pairs from paired corpus...")
    pair_counts: Counter[tuple[str, str]] = Counter()
    with open(args.paired_corpus) as f:
        for line in f:
            parts = line.strip().split("\t", 1)
            if len(parts) != 2:
                continue
            eng_words = parts[0].split()
            ipa_words = parts[1].split()
            if len(eng_words) != len(ipa_words):
                continue
            for e, i in zip(eng_words, ipa_words):
                if (
                    _is_ipa(i)
                    and _is_english(e)
                    and len(i) <= args.max_ipa_len
                    and len(e) <= args.max_spelling_len
                ):
                    pair_counts[(i, e)] += 1

    print(f"  Corpus pairs: {len(pair_counts):,}")

    # --- Also load CMU dict pairs ---
    print("Loading CMU dict pairs...")
    try:
        import nltk
        nltk.download("cmudict", quiet=True)
        from nltk.corpus import cmudict
        cmu = cmudict.dict()

        from lfm.data.loaders.ipa import IPAConverter
        conv = IPAConverter(drop_unconvertible=False)

        cmu_added = 0
        for word, pronunciations in cmu.items():
            if not _is_english(word):
                continue
            for pron in pronunciations:
                ipa = conv._arpabet_to_ipa(pron)
                if ipa and _is_ipa(ipa) and len(ipa) <= args.max_ipa_len:
                    if (ipa, word) not in pair_counts:
                        pair_counts[(ipa, word)] = 1
                        cmu_added += 1
        print(f"  CMU additions: {cmu_added:,}")
    except Exception as ex:
        print(f"  CMU dict failed ({ex}), skipping")

    print(f"  Total unique pairs: {len(pair_counts):,}")

    # --- Noise augmentation ---
    augmented: list[tuple[str, str]] = []
    n_augment = int(len(pair_counts) * args.augment_ratio)
    print(f"Generating {n_augment:,} noise-augmented pairs...")
    source_pairs = list(pair_counts.keys())
    for _ in range(n_augment):
        ipa, eng = rng.choice(source_pairs)
        noisy = _augment_ipa(ipa, rng)
        if noisy and len(noisy) <= args.max_ipa_len:
            augmented.append((noisy, eng))
    print(f"  Augmented: {len(augmented):,}")

    # --- Build final dataset ---
    all_pairs: list[tuple[str, str, int]] = []
    for (ipa, eng), freq in pair_counts.items():
        all_pairs.append((ipa, eng, freq))
    for ipa, eng in augmented:
        all_pairs.append((ipa, eng, 1))

    rng.shuffle(all_pairs)
    n_val = int(len(all_pairs) * args.val_fraction)
    val = all_pairs[:n_val]
    train = all_pairs[n_val:]

    print(f"\n  Train: {len(train):,}")
    print(f"  Val:   {len(val):,}")

    # --- Compute inverse frequency weights ---
    freqs = np.array([f for _, _, f in train], dtype=np.float32)
    inv_freq = 1.0 / np.sqrt(freqs.clip(min=1))
    inv_freq /= inv_freq.mean()  # normalize so mean weight = 1.0

    # --- Write HDF5 ---
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    dt = h5py.string_dtype()
    for split, data in [("train", train), ("val", val)]:
        path = out_dir / f"{split}.h5"
        with h5py.File(path, "w") as f:
            grp = f.create_group("pairs")
            grp.create_dataset("ipa", data=[d[0] for d in data], dtype=dt)
            grp.create_dataset("spelling", data=[d[1] for d in data], dtype=dt)
            grp.create_dataset("word_freq", data=np.array([d[2] for d in data], dtype=np.int32))
            if split == "train":
                grp.create_dataset("loss_weight", data=inv_freq)
        print(f"  Wrote {path} ({len(data):,} pairs)")

    # --- Save vocab ---
    ipa_chars = sorted(set(c for ipa, _, _ in all_pairs for c in ipa))
    eng_chars = sorted(set(c for _, eng, _ in all_pairs for c in eng))
    import json
    (out_dir / "ipa_vocab.json").write_text(json.dumps({"chars": ["<pad>", "<bos>", "<eos>"] + ipa_chars}))
    (out_dir / "spelling_vocab.json").write_text(json.dumps({"chars": ["<pad>", "<bos>", "<eos>"] + eng_chars}))
    print(f"  IPA vocab: {len(ipa_chars)+3} chars")
    print(f"  Spelling vocab: {len(eng_chars)+3} chars")

    # --- Stats ---
    ipa_lens = [len(d[0]) for d in train]
    eng_lens = [len(d[1]) for d in train]
    print(f"\n  IPA length: min={min(ipa_lens)} max={max(ipa_lens)} mean={np.mean(ipa_lens):.1f} p99={int(np.percentile(ipa_lens, 99))}")
    print(f"  Eng length: min={min(eng_lens)} max={max(eng_lens)} mean={np.mean(eng_lens):.1f} p99={int(np.percentile(eng_lens, 99))}")

    # Sample
    print(f"\nSample pairs:")
    for ipa, eng, freq in train[:15]:
        print(f"  {ipa:<25} → {eng:<20} (freq={freq})")


if __name__ == "__main__":
    main()

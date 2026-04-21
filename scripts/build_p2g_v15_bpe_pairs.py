"""Build p2g BPE training data from the v15 paired corpus.

Same IPA inputs as char-level, but output is BPE tokens (Qwen tokenizer)
instead of individual characters. Larger dataset — includes all word
pairs regardless of IPA purity since BPE can handle any spelling.

Usage:
    poetry run python scripts/build_p2g_v15_bpe_pairs.py
"""

from __future__ import annotations

import argparse
import json
import random
import re
from collections import Counter
from pathlib import Path

import h5py
import numpy as np


_IPA_CHARS = set("aɑæbdðeɛəɝfɡhiɪjklmnŋoɔpɹsʃtθuʊʌvwzʒ")


def _is_ipa(word: str) -> bool:
    return all(c in _IPA_CHARS for c in word)


_VOWELS = list("aɑæeɛəɝiɪoɔuʊʌ")
_CONSONANTS = list("bdfɡhjklmnŋpɹsʃtθvwzʒð")


def _augment_ipa(ipa: str, rng: random.Random) -> str | None:
    chars = list(ipa)
    if len(chars) < 3:
        return None
    op = rng.choice(["swap_vowel", "swap_consonant", "drop", "insert"])
    if op == "swap_vowel":
        pos = [i for i, c in enumerate(chars) if c in _VOWELS]
        if not pos: return None
        chars[rng.choice(pos)] = rng.choice(_VOWELS)
    elif op == "swap_consonant":
        pos = [i for i, c in enumerate(chars) if c in _CONSONANTS]
        if not pos: return None
        chars[rng.choice(pos)] = rng.choice(_CONSONANTS)
    elif op == "drop":
        chars.pop(rng.randint(1, len(chars) - 2))
    elif op == "insert":
        chars.insert(rng.randint(1, len(chars) - 1), rng.choice(_VOWELS + _CONSONANTS))
    return "".join(chars)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--paired-corpus",
                    default="data/datasets/english-sentences-v15/ipa_sentences.paired.txt")
    ap.add_argument("--output-dir", default="data/datasets/p2g_v15_bpe")
    ap.add_argument("--max-ipa-len", type=int, default=32)
    ap.add_argument("--max-bpe-len", type=int, default=8)
    ap.add_argument("--augment-ratio", type=float, default=0.2)
    ap.add_argument("--val-fraction", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    rng = random.Random(args.seed)

    # Load Qwen tokenizer for BPE
    from transformers import AutoTokenizer
    qwen_tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B")

    # Extract word pairs — keep ALL pairs (not just strict IPA)
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
                e = e.replace("-", "")
                if len(i) >= 1 and len(e) >= 2 and len(i) <= args.max_ipa_len:
                    pair_counts[(i, e)] += 1

    print(f"  Raw pairs: {len(pair_counts):,}")

    # Build BPE vocab from all spellings
    all_spellings = set(e for _, e in pair_counts.keys())
    bpe_token_ids: set[int] = set()
    for spelling in all_spellings:
        ids = qwen_tok.encode(spelling, add_special_tokens=False)
        bpe_token_ids.update(ids)

    # Build local vocab: qwen_id → local_id
    sorted_ids = sorted(bpe_token_ids)
    PAD, BOS, EOS = 0, 1, 2
    qwen_to_local = {qid: i + 3 for i, qid in enumerate(sorted_ids)}
    local_to_qwen = {v: k for k, v in qwen_to_local.items()}
    local_vocab_size = len(sorted_ids) + 3
    print(f"  BPE vocab: {local_vocab_size} local tokens ({len(sorted_ids)} Qwen tokens + 3 specials)")

    # Tokenize all spellings to BPE
    def spelling_to_bpe(s: str) -> list[int]:
        ids = qwen_tok.encode(s, add_special_tokens=False)
        return [qwen_to_local.get(i, PAD) for i in ids]

    # Filter pairs where BPE output fits
    pairs: list[tuple[str, str, list[int], int]] = []
    for (ipa, eng), freq in pair_counts.items():
        bpe = spelling_to_bpe(eng)
        if len(bpe) <= args.max_bpe_len and len(bpe) >= 1:
            pairs.append((ipa, eng, bpe, freq))

    print(f"  Pairs after BPE length filter: {len(pairs):,}")

    # IPA vocab from all inputs
    ipa_chars = sorted(set(c for ipa, _, _, _ in pairs for c in ipa))
    ipa_char_to_id = {c: i + 3 for i, c in enumerate(ipa_chars)}
    ipa_vocab_size = len(ipa_chars) + 3

    # Noise augmentation
    n_augment = int(len(pairs) * args.augment_ratio)
    print(f"Generating {n_augment:,} augmented pairs...")
    augmented = []
    for _ in range(n_augment):
        ipa, eng, bpe, freq = rng.choice(pairs)
        if not _is_ipa(ipa):
            continue
        noisy = _augment_ipa(ipa, rng)
        if noisy and len(noisy) <= args.max_ipa_len:
            augmented.append((noisy, eng, bpe, 1))
    pairs.extend(augmented)
    print(f"  Total pairs: {len(pairs):,}")

    # Shuffle + split
    rng.shuffle(pairs)
    n_val = int(len(pairs) * args.val_fraction)
    val = pairs[:n_val]
    train = pairs[n_val:]

    # Inverse-frequency weights for training
    freqs = np.array([f for _, _, _, f in train], dtype=np.float32)
    inv_freq = 1.0 / np.sqrt(freqs.clip(min=1))
    inv_freq /= inv_freq.mean()

    # Write HDF5
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    dt = h5py.string_dtype()
    for split, data in [("train", train), ("val", val)]:
        path = out_dir / f"{split}.h5"
        with h5py.File(path, "w") as f:
            grp = f.create_group("pairs")
            grp.create_dataset("ipa", data=[d[0] for d in data], dtype=dt)
            grp.create_dataset("spelling", data=[d[1] for d in data], dtype=dt)
            # Store BPE ids as variable-length int arrays
            bpe_dt = h5py.vlen_dtype(np.int32)
            bpe_ds = grp.create_dataset("token_ids", shape=(len(data),), dtype=bpe_dt)
            for i, (_, _, bpe, _) in enumerate(data):
                bpe_ds[i] = np.array(bpe, dtype=np.int32)
            grp.create_dataset("word_freq", data=np.array([d[3] for d in data], dtype=np.int32))
            if split == "train":
                grp.create_dataset("loss_weight", data=inv_freq)
        print(f"  Wrote {path} ({len(data):,} pairs)")

    # Save vocabs
    (out_dir / "ipa_vocab.json").write_text(json.dumps({
        "chars": ["<pad>", "<bos>", "<eos>"] + ipa_chars
    }))
    (out_dir / "bpe_vocab.json").write_text(json.dumps({
        "local_to_qwen": {str(k): v for k, v in local_to_qwen.items()},
        "qwen_to_local": {str(k): v for k, v in qwen_to_local.items()},
        "vocab_size": local_vocab_size,
        "tokens": {str(lid): qwen_tok.decode([qid]) for lid, qid in local_to_qwen.items()},
    }))

    print(f"\n  Train: {len(train):,}  Val: {len(val):,}")
    print(f"  IPA vocab: {ipa_vocab_size}  BPE vocab: {local_vocab_size}")

    # Sample
    print("\nSamples:")
    for ipa, eng, bpe, freq in train[:15]:
        bpe_text = " ".join(qwen_tok.decode([local_to_qwen[b]]) for b in bpe)
        print(f"  {ipa:<25} → {eng:<20} → [{bpe_text}] (freq={freq})")


if __name__ == "__main__":
    main()

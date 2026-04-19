"""Build context-windowed p2g BPE training data (multiprocessed).

Workers read their chunk of the paired corpus directly from disk —
no copying the full sentence list to each process.

Usage:
    poetry run python scripts/build_p2g_v15_bpe_context.py --context 1 --workers 16
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import random
from collections import Counter
from pathlib import Path

import h5py
import numpy as np

TARGET_START = "▸"
TARGET_END = "◂"
_IPA_CHARS = set("aɑæbdðeɛəɝfɡhiɪjklmnŋoɔpɹsʃtθuʊʌvwzʒ-")
_VOWELS = list("aɑæeɛəɝiɪoɔuʊʌ")
_CONSONANTS = list("bdfɡhjklmnŋpɹsʃtθvwzʒð")


def _is_ipa(word: str) -> bool:
    return all(c in _IPA_CHARS for c in word)


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


def _extract_pairs_from_file(args_tuple):
    """Worker: read a line range from the corpus file and extract pairs."""
    corpus_path, start_line, end_line, ctx, max_ipa_len = args_tuple
    use_context = ctx > 0
    local_counts: Counter[tuple[str, str]] = Counter()

    with open(corpus_path) as f:
        for line_no, line in enumerate(f):
            if line_no < start_line:
                continue
            if line_no >= end_line:
                break
            parts = line.strip().split("\t", 1)
            if len(parts) != 2:
                continue
            eng_words = parts[0].split()
            ipa_words = parts[1].split()
            if len(eng_words) != len(ipa_words):
                continue

            for idx in range(len(eng_words)):
                eng = eng_words[idx]
                ipa = ipa_words[idx]
                if len(eng) < 2 or len(ipa) < 1:
                    continue
                if use_context:
                    left = ipa_words[max(0, idx - ctx):idx]
                    right = ipa_words[idx + 1:idx + 1 + ctx]
                    ipa_input = " ".join(left + [TARGET_START, ipa, TARGET_END] + right)
                else:
                    ipa_input = ipa
                if len(ipa_input) > max_ipa_len:
                    continue
                local_counts[(ipa_input, eng)] += 1

    return local_counts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--paired-corpus",
                    default="data/datasets/english-sentences-v15/ipa_sentences.paired.txt")
    ap.add_argument("--output-dir", default="data/datasets/p2g_v15_bpe_ctx")
    ap.add_argument("--context", type=int, default=2)
    ap.add_argument("--max-ipa-len", type=int, default=96)
    ap.add_argument("--max-bpe-len", type=int, default=8)
    ap.add_argument("--augment-ratio", type=float, default=0.15)
    ap.add_argument("--val-fraction", type=float, default=0.05)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    ctx = args.context
    use_context = ctx > 0
    n_workers = min(args.workers, mp.cpu_count())
    corpus_path = args.paired_corpus

    # Count lines for chunking (fast — just counts newlines)
    print(f"Counting lines in {corpus_path}...")
    total_lines = sum(1 for _ in open(corpus_path))
    print(f"  {total_lines:,} lines")

    # Split line ranges for workers
    chunk_size = (total_lines + n_workers - 1) // n_workers
    worker_args = [
        (corpus_path, i * chunk_size, min((i + 1) * chunk_size, total_lines),
         ctx, args.max_ipa_len)
        for i in range(n_workers)
    ]

    print(f"  Extracting pairs with {n_workers} workers (context={ctx})...")
    with mp.Pool(n_workers) as pool:
        chunk_results = pool.map(_extract_pairs_from_file, worker_args)

    # Merge counters (sequential but lightweight — just Counter.update)
    print("  Merging results...")
    pair_counts: Counter[tuple[str, str]] = Counter()
    for c in chunk_results:
        pair_counts.update(c)
    del chunk_results
    print(f"  Unique context pairs: {len(pair_counts):,}")

    # BPE vocab
    from transformers import AutoTokenizer
    qwen_tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B")

    all_spellings = set(e for _, e in pair_counts.keys())
    print(f"  BPE-tokenizing {len(all_spellings):,} unique spellings...")
    bpe_token_ids: set[int] = set()
    spelling_to_bpe_cache: dict[str, list[int]] = {}
    for spelling in all_spellings:
        ids = qwen_tok.encode(spelling, add_special_tokens=False)
        bpe_token_ids.update(ids)
        spelling_to_bpe_cache[spelling] = ids

    sorted_ids = sorted(bpe_token_ids)
    PAD = 0
    qwen_to_local = {qid: i + 3 for i, qid in enumerate(sorted_ids)}
    local_to_qwen = {v: k for k, v in qwen_to_local.items()}
    local_vocab_size = len(sorted_ids) + 3
    print(f"  BPE vocab: {local_vocab_size}")

    # Filter + convert
    print("  Filtering and converting...")
    pairs: list[tuple[str, str, list[int], int]] = []
    for (ipa_input, eng), freq in pair_counts.items():
        raw_ids = spelling_to_bpe_cache.get(eng)
        if raw_ids is None:
            continue
        bpe = [qwen_to_local.get(i, PAD) for i in raw_ids]
        if len(bpe) <= args.max_bpe_len and len(bpe) >= 1:
            pairs.append((ipa_input, eng, bpe, freq))
    del pair_counts, spelling_to_bpe_cache
    print(f"  Pairs after BPE filter: {len(pairs):,}")

    # IPA vocab
    ipa_chars = sorted(set(c for ipa, _, _, _ in pairs for c in ipa if c not in (TARGET_START, TARGET_END)))
    if use_context:
        ipa_chars_full = ["<pad>", "<bos>", "<eos>", TARGET_START, TARGET_END] + ipa_chars
    else:
        ipa_chars_full = ["<pad>", "<bos>", "<eos>"] + ipa_chars

    # Noise augmentation
    n_augment = int(len(pairs) * args.augment_ratio)
    print(f"  Generating {n_augment:,} augmented pairs...")
    augmented = []
    for _ in range(n_augment):
        ipa_input, eng, bpe, freq = rng.choice(pairs)
        if use_context:
            parts = ipa_input.split(TARGET_START)
            if len(parts) != 2: continue
            rest = parts[1].split(TARGET_END)
            if len(rest) != 2: continue
            target = rest[0].strip()
            if not _is_ipa(target): continue
            noisy = _augment_ipa(target, rng)
            if noisy and len(noisy) <= 24:
                new_input = f"{parts[0].strip()} {TARGET_START} {noisy} {TARGET_END} {rest[1].strip()}".strip()
                if len(new_input) <= args.max_ipa_len:
                    augmented.append((new_input, eng, bpe, 1))
        else:
            if not _is_ipa(ipa_input): continue
            noisy = _augment_ipa(ipa_input, rng)
            if noisy and len(noisy) <= args.max_ipa_len:
                augmented.append((noisy, eng, bpe, 1))
    pairs.extend(augmented)
    print(f"  Total: {len(pairs):,}")

    # Shuffle + split
    rng.shuffle(pairs)
    n_val = int(len(pairs) * args.val_fraction)
    val = pairs[:n_val]
    train = pairs[n_val:]

    # Inverse-frequency weights
    freqs = np.array([f for _, _, _, f in train], dtype=np.float32)
    inv_freq = 1.0 / np.sqrt(freqs.clip(min=1))
    inv_freq /= inv_freq.mean()

    # Write HDF5
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    dt = h5py.string_dtype()
    for split, data in [("train", train), ("val", val)]:
        path = out_dir / f"{split}.h5"
        print(f"  Writing {path} ({len(data):,} pairs)...")
        with h5py.File(path, "w") as f:
            grp = f.create_group("pairs")
            grp.create_dataset("ipa", data=[d[0] for d in data], dtype=dt)
            grp.create_dataset("spelling", data=[d[1] for d in data], dtype=dt)
            bpe_dt = h5py.vlen_dtype(np.int32)
            bpe_ds = grp.create_dataset("token_ids", shape=(len(data),), dtype=bpe_dt)
            for i, (_, _, bpe, _) in enumerate(data):
                bpe_ds[i] = np.array(bpe, dtype=np.int32)
            grp.create_dataset("word_freq", data=np.array([d[3] for d in data], dtype=np.int32))
            if split == "train":
                grp.create_dataset("loss_weight", data=inv_freq)

    # Save vocabs
    (out_dir / "ipa_vocab.json").write_text(json.dumps({"chars": ipa_chars_full}))
    max_lid = max(local_to_qwen.keys())
    token_list = ["<pad>", "<bos>", "<eos>"] + [""] * (max_lid - 2)
    for lid, qid in local_to_qwen.items():
        token_list[lid] = qwen_tok.decode([qid])
    (out_dir / "bpe_vocab.json").write_text(json.dumps({
        "local_to_qwen": {str(k): v for k, v in local_to_qwen.items()},
        "qwen_to_local": {str(k): v for k, v in qwen_to_local.items()},
        "vocab_size": local_vocab_size,
        "tokens": token_list,
    }))
    (out_dir / "config.json").write_text(json.dumps({
        "context": ctx,
        "max_ipa_len": args.max_ipa_len,
        "max_bpe_len": args.max_bpe_len,
    }))

    print(f"\n  Train: {len(train):,}  Val: {len(val):,}")
    print(f"  IPA vocab: {len(ipa_chars_full)}  BPE vocab: {local_vocab_size}")
    print(f"  Context: ±{ctx}" if use_context else "  Context: none")

    print("\nSamples:")
    for ipa_input, eng, bpe, freq in train[:10]:
        print(f"  {ipa_input[:70]:<72} → {eng:<18} (f={freq})")


if __name__ == "__main__":
    main()

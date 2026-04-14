#!/usr/bin/env python
"""Transcode English corpus → v9 token-id sequences for VAE pretraining.

For each English sentence:
  1. Tokenize with Qwen BPE.
  2. Map each Qwen token id → its v9 alphabet index.
  3. Drop sentences with too many out-of-alphabet tokens (vocab is the
     top-1500 most common English BPE tokens; ~85-90% coverage of normal
     prose — sentences with rare/technical words skipped).
  4. Insert v9 word-boundary token (id = vocab_phonemes) at sentence
     start of each new "phrase" so the VAE learns explicit phrase
     structure.

Output: data/datasets/english-v9-pidgin/samples.h5 — same schema as
v8 transcoded corpus (phoneme_ids vlen-int32 + length + language).
"""

from __future__ import annotations

import json
import logging
import unicodedata
from collections import Counter
from pathlib import Path

import h5py
import numpy as np
from transformers import AutoTokenizer

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger(__name__)

MODEL = "Qwen/Qwen2.5-0.5B"
ENGLISH_CORPUS = Path("data/translator/english_corpus.txt")
ALPHABET_PATH = Path("data/phoneme_alphabet_v9.json")
OUTPUT_DIR = Path("data/datasets/english-v9-pidgin")
OUTPUT_H5 = OUTPUT_DIR / "samples.h5"
OUTPUT_MAPPING = OUTPUT_DIR / "mapping.json"
OUTPUT_MANIFEST = OUTPUT_DIR / "manifest.yaml"

MAX_LINES = 300_000
# OOV strategy: skip out-of-vocab positions (don't introduce an <unk>
# token at training time — the agent shouldn't learn to emit it).
# Only drop sentences that end up below MIN_LEN after OOV removal, OR
# where the in-vocab fraction is too low (mostly noise).
MIN_INVOCAB_FRACTION = 0.50  # need ≥50% of original tokens to be in v9
MIN_LEN = 5                  # drop sentences with <5 valid v9 tokens
MAX_LEN = 256                # cap sequence length at 256 tokens


def main() -> None:
    if not ENGLISH_CORPUS.exists():
        raise FileNotFoundError(f"english corpus not found: {ENGLISH_CORPUS}")
    if not ALPHABET_PATH.exists():
        raise FileNotFoundError(
            f"v9 alphabet not found: {ALPHABET_PATH}; run "
            f"scripts/design_alphabet_v9_english_pidgin.py first",
        )

    alphabet = json.loads(ALPHABET_PATH.read_text())
    qwen_token_ids: list[int] = alphabet["qwen_token_ids"]
    vocab_phonemes = len(qwen_token_ids)
    word_boundary_id = vocab_phonemes
    vocab_size = vocab_phonemes + 1
    qwen_to_v9: dict[int, int] = {tid: i for i, tid in enumerate(qwen_token_ids)}

    logger.info(
        f"v9 alphabet: {vocab_phonemes} tokens, "
        f"word_boundary_id={word_boundary_id}, vocab_size={vocab_size}",
    )

    logger.info(f"loading Qwen tokenizer: {MODEL}")
    tok = AutoTokenizer.from_pretrained(MODEL)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Save mapping artifact for reproducibility
    OUTPUT_MAPPING.write_text(json.dumps({
        "vocab_phonemes": vocab_phonemes,
        "word_boundary_id": word_boundary_id,
        "vocab_size": vocab_size,
        "alphabet_path": str(ALPHABET_PATH),
        "qwen_to_v9_id": {str(k): v for k, v in qwen_to_v9.items()},
    }, indent=2))
    logger.info(f"wrote mapping: {OUTPUT_MAPPING}")

    # Pass 1: tokenize and filter, accumulate sequences in memory.
    # Memory budget: 300K sentences × ~30 tokens × 4 bytes = ~36 MB — fine.
    logger.info(f"tokenizing English corpus (≤{MAX_LINES:,} lines)...")
    sequences: list[np.ndarray] = []
    n_scanned = 0
    n_oov_dropped = 0
    n_short_dropped = 0
    n_kept = 0
    length_hist: Counter[int] = Counter()

    with ENGLISH_CORPUS.open() as f:
        for i, line in enumerate(f):
            if i >= MAX_LINES:
                break
            n_scanned += 1
            text = line.strip()
            if not text:
                continue

            qwen_ids = tok.encode(text, add_special_tokens=False)
            if not qwen_ids:
                continue

            # Map to v9; track OOV rate
            v9_ids: list[int] = []
            oov = 0
            for qid in qwen_ids:
                v9 = qwen_to_v9.get(qid)
                if v9 is None:
                    oov += 1
                else:
                    v9_ids.append(v9)
            oov_frac = oov / len(qwen_ids)

            # Drop only sentences where most tokens are OOV (mostly noise);
            # otherwise keep the in-vocab portion.
            in_vocab_frac = 1.0 - oov_frac
            if in_vocab_frac < MIN_INVOCAB_FRACTION:
                n_oov_dropped += 1
                continue
            if len(v9_ids) < MIN_LEN:
                n_short_dropped += 1
                continue
            if len(v9_ids) > MAX_LEN:
                v9_ids = v9_ids[:MAX_LEN]

            # Append a word-boundary token at end (phrase terminator);
            # the VAE will then learn to emit <ws> as a structural marker.
            # We don't insert <ws> within because Qwen-BPE space-prefixed
            # tokens already carry implicit word-boundary info.
            v9_ids.append(word_boundary_id)
            sequences.append(np.asarray(v9_ids, dtype=np.int32))
            length_hist[len(v9_ids)] += 1
            n_kept += 1

            if n_scanned % 25_000 == 0:
                logger.info(
                    f"  scanned {n_scanned:,}  kept {n_kept:,}  "
                    f"oov-dropped {n_oov_dropped:,}  short {n_short_dropped:,}",
                )

    logger.info(
        f"final: scanned {n_scanned:,}, kept {n_kept:,}, "
        f"oov-dropped {n_oov_dropped:,}, short {n_short_dropped:,}",
    )

    if n_kept == 0:
        raise RuntimeError("no sequences kept — check OOV threshold / corpus")

    lens = np.asarray([len(s) for s in sequences], dtype=np.int32)
    logger.info(
        f"length stats: min={lens.min()} max={lens.max()} "
        f"mean={lens.mean():.1f} median={np.median(lens):.1f} "
        f"p99={np.percentile(lens, 99):.0f}",
    )

    # Write HDF5 in same schema as v8 corpus
    logger.info(f"writing {OUTPUT_H5}...")
    vlen_int32 = h5py.special_dtype(vlen=np.dtype("int32"))
    n = len(sequences)
    with h5py.File(OUTPUT_H5, "w") as out:
        g = out.create_group("samples")
        ph_ds = g.create_dataset("phoneme_ids", shape=(n,), dtype=vlen_int32)
        len_ds = g.create_dataset("phoneme_length", shape=(n,), dtype=np.int32)
        lang_ds = g.create_dataset(
            "language", shape=(n,), dtype=h5py.string_dtype(),
        )
        seq_ds = g.create_dataset("seq", shape=(n,), dtype=np.int64)
        for i, s in enumerate(sequences):
            ph_ds[i] = s
        len_ds[:] = lens
        lang_ds[:] = [b"eng"] * n
        seq_ds[:] = np.arange(n, dtype=np.int64)
        g.attrs["vocab_size"] = vocab_size
        g.attrs["vocab_phonemes"] = vocab_phonemes
        g.attrs["word_boundary_id"] = word_boundary_id
        g.attrs["alphabet_path"] = str(ALPHABET_PATH)
        g.attrs["alphabet_version"] = "v9-english-pidgin"

    import yaml
    OUTPUT_MANIFEST.write_text(yaml.safe_dump({
        "name": "english-v9-pidgin",
        "description": "English prose tokenized into top-1500 Qwen BPE "
                       "subwords (v9 pidgin alphabet) for VAE pretraining.",
        "source": str(ENGLISH_CORPUS),
        "alphabet_path": str(ALPHABET_PATH),
        "vocab_phonemes": vocab_phonemes,
        "word_boundary_id": word_boundary_id,
        "vocab_size": vocab_size,
        "total_samples": n,
        "min_invocab_fraction": MIN_INVOCAB_FRACTION,
        "min_seq_len": MIN_LEN,
        "max_seq_len": MAX_LEN,
    }))
    logger.info(f"wrote manifest: {OUTPUT_MANIFEST}")
    logger.info("done.")


if __name__ == "__main__":
    main()

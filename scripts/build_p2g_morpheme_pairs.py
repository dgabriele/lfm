#!/usr/bin/env python
"""Build (IPA, morpheme-seq) pairs for morpheme-output p2g.

Pipeline:
  1. Train Morfessor Baseline on unique English words (with frequency
     weights from english_corpus.txt).
  2. Segment every CMU-IPA-mapped word into morphemes.
  3. Save:
       - morphemes.json — full morpheme vocab (with ids + special tokens)
       - train.h5 / val.h5 — (ipa: str, morphemes: vlen-int32, freq) per row

Morfessor Baseline (Creutz & Lagus) uses MDL to find sub-word units
that act as morpheme proxies.  Output morphemes are real substrings of
training words, so the seq2seq's output vocab is by construction a set
of *real* morphemes — pseudoword IPA can only decode to real-morpheme
sequences.
"""

from __future__ import annotations

import argparse
import json
import logging
from collections import Counter
from pathlib import Path

import h5py
import morfessor
import numpy as np

from lfm.data.loaders.ipa import IPAConverter

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger(__name__)

CORPUS = Path("data/translator/english_corpus.txt")
OUT_DIR = Path("data/datasets/p2g_morpheme")
MIN_FREQ = 2
MIN_LEN = 2
MAX_LEN = 20
VAL_FRAC = 0.05
SEED = 42

# Reserved morpheme ids.
PAD_ID = 0
BOS_ID = 1
EOS_ID = 2
SPECIAL = ["<pad>", "<bos>", "<eos>"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", type=Path, default=CORPUS)
    ap.add_argument("--out", type=Path, default=OUT_DIR)
    ap.add_argument("--min-freq", type=int, default=MIN_FREQ)
    ap.add_argument("--max-lines", type=int, default=None)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)

    # ── 1. word frequencies ──
    logger.info(f"scanning {args.corpus}")
    counts: Counter[str] = Counter()
    n_lines = 0
    with args.corpus.open() as f:
        for i, line in enumerate(f):
            if args.max_lines and i >= args.max_lines:
                break
            n_lines += 1
            for raw in line.split():
                w = raw.strip(".,!?;:\"'()[]{}—–-").lower()
                if not w.isalpha():
                    continue
                if not (MIN_LEN <= len(w) <= MAX_LEN):
                    continue
                counts[w] += 1
    logger.info(f"  {n_lines:,} lines, {len(counts):,} unique words")

    train_words = [(w, c) for w, c in counts.items() if c >= args.min_freq]
    logger.info(f"  {len(train_words):,} words above min_freq={args.min_freq}")

    # ── 2. Morfessor Baseline training ──
    # corpusweight α controls splitting aggressiveness: α=1 (default) gives
    # near-no segmentation on small corpora; α=0.001 forces aggressive
    # sub-word splits (encoding cost dominates → MDL prefers more morphemes).
    logger.info("training Morfessor Baseline (this is fast — minutes)")
    model = morfessor.BaselineModel(corpusweight=100.0)
    # Morfessor wants (count, atom_seq) pairs; default compound atoms = chars.
    model.load_data([(c, tuple(w)) for w, c in train_words])
    model.train_batch()
    logger.info("  done.")

    # Save the trained model for inspection / reuse.
    morfessor_io = morfessor.MorfessorIO()
    morfessor_io.write_binary_model_file(
        str(args.out / "morfessor.bin"), model,
    )

    # ── 3. Build morpheme vocab from segmenting all training words ──
    logger.info("segmenting all words to extract morpheme vocabulary")
    morpheme_counts: Counter[str] = Counter()
    word_to_segs: dict[str, list[str]] = {}
    for w, c in train_words:
        segs = model.viterbi_segment(w)[0]   # returns (segments, log_prob)
        word_to_segs[w] = segs
        for s in segs:
            morpheme_counts[s] += c
    morphemes_sorted = SPECIAL + sorted(morpheme_counts.keys())
    m_to_id = {m: i for i, m in enumerate(morphemes_sorted)}
    logger.info(
        f"  morpheme vocab size: {len(morphemes_sorted):,} "
        f"(specials + {len(morpheme_counts):,} morphemes)",
    )

    # ── 4. Convert each word to IPA, pair with morpheme-id sequence ──
    logger.info("converting to IPA + building (IPA, morpheme-seq) pairs")
    conv = IPAConverter(drop_unconvertible=True)
    pairs: list[tuple[str, list[int], int]] = []
    n_oov = 0
    for w, c in train_words:
        ipa = conv._english_word_to_ipa(w)  # noqa: SLF001
        if ipa is None:
            n_oov += 1
            continue
        seg_ids = [m_to_id[s] for s in word_to_segs[w]]
        pairs.append((ipa, seg_ids, c))
    logger.info(f"  kept {len(pairs):,} pairs (CMU OOV dropped: {n_oov:,})")

    # ── 5. train/val split ──
    rng.shuffle(pairs)  # type: ignore[arg-type]
    n_val = max(1, int(len(pairs) * VAL_FRAC))
    val = pairs[:n_val]
    train = pairs[n_val:]
    logger.info(f"  train={len(train):,}  val={len(val):,}")

    # ── 6. write artifacts ──
    str_dt = h5py.string_dtype(encoding="utf-8")
    vlen_int = h5py.vlen_dtype(np.int32)
    for name, rows in [("train", train), ("val", val)]:
        out_path = args.out / f"{name}.h5"
        logger.info(f"writing {out_path}")
        with h5py.File(out_path, "w") as h:
            g = h.create_group("pairs")
            n = len(rows)
            g.create_dataset("ipa", shape=(n,), dtype=str_dt,
                             data=[r[0] for r in rows])
            seg_ds = g.create_dataset("morpheme_ids", shape=(n,), dtype=vlen_int)
            for i, r in enumerate(rows):
                seg_ds[i] = np.asarray(r[1], dtype=np.int32)
            g.create_dataset("seg_len", shape=(n,), dtype=np.int32,
                             data=[len(r[1]) for r in rows])
            g.create_dataset("word_freq", shape=(n,), dtype=np.int32,
                             data=[r[2] for r in rows])
            # Recover the spelling for eval (decode morpheme ids).
            spellings = ["".join([morphemes_sorted[m] for m in r[1]]) for r in rows]
            g.create_dataset("spelling", shape=(n,), dtype=str_dt, data=spellings)
            g.attrs["source_corpus"] = str(args.corpus)
            g.attrs["min_freq"] = args.min_freq
            g.attrs["morpheme_vocab_size"] = len(morphemes_sorted)

    # Save the morpheme vocab.
    (args.out / "morphemes.json").write_text(
        json.dumps({
            "morphemes": morphemes_sorted,
            "specials": SPECIAL,
            "pad_id": PAD_ID, "bos_id": BOS_ID, "eos_id": EOS_ID,
        }, ensure_ascii=False, indent=2),
    )

    # ── 7. diagnostic: average segments-per-word + samples ──
    seg_lens = [len(r[1]) for r in pairs]
    logger.info(
        f"segments/word: mean={np.mean(seg_lens):.2f} "
        f"max={max(seg_lens)} p99={np.percentile(seg_lens, 99):.0f}",
    )
    logger.info("frequent-word segmentations:")
    for w, c in sorted(train_words, key=lambda x: -x[1])[:15]:
        if w in word_to_segs:
            logger.info(f"  {w!r:>14} ({c:,}) → {word_to_segs[w]}")
    logger.info("morphology samples (random):")
    for w in ["orthography", "consciousness", "machine", "running",
              "unstoppable", "transformation", "philosophy", "unbelievable"]:
        if w in word_to_segs:
            logger.info(f"  {w!r:>16} → {word_to_segs[w]}")
    logger.info("done.")


if __name__ == "__main__":
    main()

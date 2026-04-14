#!/usr/bin/env python
"""Build word-level (IPA, spelling) pairs for p2g VAE training.

Extracts unique English words from data/translator/english_corpus.txt,
converts each to IPA via the CMU Pronouncing Dictionary (same path used
elsewhere in the repo), and writes paired HDF5 splits for v10 p2g VAE
training.

Output schema (HDF5 group ``pairs``):
    ipa        : vlen-str  — IPA form (e.g. "ˈɔɹθəɡɹæfi")
    spelling   : vlen-str  — canonical lowercase spelling (e.g. "orthography")
    word_freq  : int32     — corpus frequency (for optional weighting)
"""

from __future__ import annotations

import argparse
import logging
from collections import Counter
from pathlib import Path

import h5py
import numpy as np

from lfm.data.loaders.ipa import IPAConverter

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger(__name__)

CORPUS = Path("data/translator/english_corpus.txt")
OUT_DIR = Path("data/datasets/p2g_v11")
MIN_FREQ = 1           # keep all words seen at least once
MIN_LEN = 2
MAX_LEN = 20           # longest real English word ≈ 18
VAL_FRAC = 0.05
SEED = 42


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", type=Path, default=CORPUS)
    ap.add_argument("--out", type=Path, default=OUT_DIR)
    ap.add_argument("--min-freq", type=int, default=MIN_FREQ)
    ap.add_argument("--max-lines", type=int, default=None,
                    help="Cap source lines for quick dev runs")
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)

    # ── 1. count word frequencies ──
    logger.info(f"scanning {args.corpus}")
    counts: Counter[str] = Counter()
    n_lines = 0
    with args.corpus.open() as f:
        for i, line in enumerate(f):
            if args.max_lines and i >= args.max_lines:
                break
            n_lines += 1
            for raw in line.split():
                # Keep alphabetic only (strip punctuation; skip numerics,
                # hyphenated compounds, contractions).
                w = raw.strip(".,!?;:\"'()[]{}—–-").lower()
                if not w.isalpha():
                    continue
                if not (MIN_LEN <= len(w) <= MAX_LEN):
                    continue
                counts[w] += 1
    logger.info(f"  {n_lines:,} lines, {len(counts):,} unique words")

    # ── 2. IPA-convert each unique word above MIN_FREQ ──
    # Use ALL CMU pronunciations per word (not just the first).  Multi-pron
    # entries teach the model that near-homophone pronunciations map to the
    # same canonical spelling, broadening its IPA-input robustness.
    from lfm.data.loaders.ipa import _ARPA_TO_IPA  # noqa: SLF001 — module-private table
    logger.info(f"converting to IPA (CMU dict, all pronunciations, min_freq={args.min_freq})")
    conv = IPAConverter(drop_unconvertible=True)
    cmu = conv._get_cmu_dict()  # noqa: SLF001
    pairs: list[tuple[str, str, int]] = []
    n_oov = 0
    for word, freq in counts.items():
        if freq < args.min_freq:
            continue
        if word not in cmu:
            n_oov += 1
            continue
        for phones in cmu[word]:
            ipa = "".join(
                _ARPA_TO_IPA.get(p.rstrip("012"), p) for p in phones
            )
            if ipa:
                pairs.append((ipa, word, freq))
    logger.info(
        f"  kept {len(pairs):,} pairs across {len(set(p[1] for p in pairs)):,} unique words  "
        f"(OOV-CMU dropped: {n_oov:,})"
    )

    if not pairs:
        raise RuntimeError("no pairs built — check corpus / CMU dict install")

    # ── 3. train/val split (stratified by frequency bucket so val has
    #    realistic rare-word coverage, not just top-k) ──
    rng.shuffle(pairs)  # type: ignore[arg-type]
    n_val = max(1, int(len(pairs) * VAL_FRAC))
    val_pairs = pairs[:n_val]
    train_pairs = pairs[n_val:]
    logger.info(f"  train={len(train_pairs):,}  val={len(val_pairs):,}")

    # ── 4. write HDF5 ──
    str_dt = h5py.string_dtype(encoding="utf-8")
    for name, rows in [("train", train_pairs), ("val", val_pairs)]:
        out_path = args.out / f"{name}.h5"
        logger.info(f"writing {out_path}")
        with h5py.File(out_path, "w") as h:
            g = h.create_group("pairs")
            n = len(rows)
            g.create_dataset("ipa", shape=(n,), dtype=str_dt,
                             data=[r[0] for r in rows])
            g.create_dataset("spelling", shape=(n,), dtype=str_dt,
                             data=[r[1] for r in rows])
            g.create_dataset("word_freq", shape=(n,), dtype=np.int32,
                             data=[r[2] for r in rows])
            g.attrs["source_corpus"] = str(args.corpus)
            g.attrs["min_freq"] = args.min_freq

    # ── 5. quick stats ──
    lens_ipa = [len(p[0]) for p in pairs]
    lens_sp = [len(p[1]) for p in pairs]
    logger.info(
        f"IPA len:  min={min(lens_ipa)} max={max(lens_ipa)} "
        f"mean={np.mean(lens_ipa):.1f} p99={np.percentile(lens_ipa, 99):.0f}",
    )
    logger.info(
        f"spell len: min={min(lens_sp)} max={max(lens_sp)} "
        f"mean={np.mean(lens_sp):.1f} p99={np.percentile(lens_sp, 99):.0f}",
    )
    logger.info("sample pairs:")
    for ipa, sp, freq in sorted(pairs, key=lambda x: -x[2])[:10]:
        logger.info(f"  {ipa!r:>22} → {sp!r:<16} (freq={freq:,})")
    logger.info("rare samples:")
    for ipa, sp, freq in sorted(pairs, key=lambda x: x[2])[:10]:
        logger.info(f"  {ipa!r:>22} → {sp!r:<16} (freq={freq:,})")
    logger.info("done.")


if __name__ == "__main__":
    main()

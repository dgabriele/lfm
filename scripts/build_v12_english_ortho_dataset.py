#!/usr/bin/env python
"""Build the v12 ORTHOGRAPHIC-BPE English dataset (no IPA).

Same v7 HDF5 schema + SPM-8000 pipeline as build_v7_english_ipa_dataset.py,
but trains SPM on raw English text rather than CMU-IPA-converted text.
Skips the CMUDict step entirely — every constituent gets through (no
OOV drops), and the SPM learns subwords like ``the``, ``Ġgov``,
``ernment``, ``ing``, etc., directly from the orthography.

The result is a VAE that emits English subwords as output, ready for
Qwen to re-tokenize and read zero-shot — no p2g rendering layer needed.

Output:
  - data/datasets/english-constituents-v12/samples.h5
  - data/datasets/english-constituents-v12/spm.model / .vocab
"""

from __future__ import annotations

import argparse
import logging
import random
from pathlib import Path

import h5py
import numpy as np
import sentencepiece as spm

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger(__name__)

SOURCE = Path("data/datasets/english-diverse-constituents/constituents.txt")
OUT_DIR = Path("data/datasets/english-constituents-v12")
SPM_MODEL_PREFIX = OUT_DIR / "spm"
OUT_H5 = OUT_DIR / "samples.h5"

SPM_VOCAB_SIZE = 8000
SPM_TRAIN_LINES = 500_000
MIN_LEN_CHARS = 10             # drop very short constituents


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", type=Path, default=SOURCE)
    ap.add_argument("--out", type=Path, default=OUT_DIR)
    ap.add_argument("--max-lines", type=int, default=None)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    # ── 1. Collect constituents directly (no IPA conversion) ──
    logger.info(f"reading {args.source}")
    text_list: list[str] = []
    n_scanned = 0
    with args.source.open() as f:
        for i, line in enumerate(f):
            if args.max_lines and i >= args.max_lines:
                break
            n_scanned += 1
            raw = line.strip()
            if len(raw) < MIN_LEN_CHARS:
                continue
            text_list.append(raw)
            if n_scanned % 1_000_000 == 0:
                logger.info(f"  {n_scanned:,} scanned  {len(text_list):,} kept")
    logger.info(
        f"final: {n_scanned:,} scanned, {len(text_list):,} kept "
        f"({len(text_list) / max(n_scanned, 1):.1%})",
    )

    # ── 2. Train SPM on a random subset of raw text ──
    tmp_train = args.out / "text_for_spm.txt"
    rng = random.Random(42)
    spm_train = (
        rng.sample(text_list, SPM_TRAIN_LINES)
        if len(text_list) > SPM_TRAIN_LINES else text_list
    )
    logger.info(f"writing SPM training text ({len(spm_train):,} lines)")
    tmp_train.write_text("\n".join(spm_train))

    logger.info(f"training SentencePiece (vocab={SPM_VOCAB_SIZE})")
    spm.SentencePieceTrainer.Train(
        input=str(tmp_train),
        model_prefix=str(SPM_MODEL_PREFIX),
        vocab_size=SPM_VOCAB_SIZE,
        model_type="bpe",
        character_coverage=0.9995,
        pad_id=-1,
        unk_id=0,
        bos_id=-1,
        eos_id=-1,
        normalization_rule_name="identity",
        shuffle_input_sentence=True,
        input_sentence_size=len(spm_train),
    )
    tmp_train.unlink()
    logger.info(f"wrote {SPM_MODEL_PREFIX}.model + .vocab")

    # ── 3. Write HDF5 in v7 schema (reusing 'ipa' field as the
    # general 'training text' slot — downstream loader just reads it) ──
    logger.info(f"writing {OUT_H5}")
    str_dt = h5py.string_dtype(encoding="utf-8")
    n = len(text_list)
    text_lengths = np.asarray([len(s) for s in text_list], dtype=np.int32)
    logger.info(
        f"text length stats: min={text_lengths.min()} max={text_lengths.max()} "
        f"mean={text_lengths.mean():.1f} median={np.median(text_lengths):.0f} "
        f"p99={np.percentile(text_lengths, 99):.0f}",
    )
    with h5py.File(OUT_H5, "w") as h:
        g = h.create_group("samples")
        # 'ipa' slot holds the training text (orthographic English here).
        g.create_dataset("ipa", shape=(n,), dtype=str_dt, data=text_list)
        g.create_dataset("ipa_length", shape=(n,), dtype=np.int32, data=text_lengths)
        g.create_dataset("language", shape=(n,), dtype=str_dt, data=["eng"] * n)
        g.create_dataset("raw", shape=(n,), dtype=str_dt, data=text_list)
        g.create_dataset("seq", shape=(n,), dtype=np.int64, data=np.arange(n, dtype=np.int64))
        g.create_dataset("source", shape=(n,), dtype=str_dt, data=["constituent"] * n)
        g.create_dataset("source_file", shape=(n,), dtype=str_dt, data=[str(args.source)] * n)
    logger.info("done.")


if __name__ == "__main__":
    main()

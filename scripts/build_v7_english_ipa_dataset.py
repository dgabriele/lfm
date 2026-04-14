#!/usr/bin/env python
"""Build the v7-format IPA HDF5 dataset from English constituents.

Pipeline:
  1. Read constituents.txt (one constituent per line, 11.66M total).
  2. Convert each to IPA via CMUDict (English).  Drop OOV-only lines.
  3. Train a SentencePiece model on the IPA strings (v7 vocab=8000).
  4. Write the HDF5 dataset in the same schema as
     ``data/datasets/constituents-12lang-all/samples.h5`` so the
     existing pretrain loader can consume it.

Output:
  - data/datasets/english-constituents-v7/samples.h5
  - data/datasets/english-constituents-v7/spm.model
  - data/datasets/english-constituents-v7/spm.vocab
"""

from __future__ import annotations

import argparse
import logging
import random
from pathlib import Path

import h5py
import numpy as np
import sentencepiece as spm

from lfm.data.loaders.ipa import IPAConverter, _clean_ipa

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger(__name__)

SOURCE = Path("data/datasets/english-diverse-constituents/constituents.txt")
OUT_DIR = Path("data/datasets/english-constituents-v7")
SPM_MODEL_PREFIX = OUT_DIR / "spm"
OUT_H5 = OUT_DIR / "samples.h5"

SPM_VOCAB_SIZE = 8000
SPM_TRAIN_LINES = 500_000          # subset of IPA for SPM training (enough signal)
MIN_IPA_LEN = 5


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", type=Path, default=SOURCE)
    ap.add_argument("--out", type=Path, default=OUT_DIR)
    ap.add_argument("--max-lines", type=int, default=None)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    # ── 1. Convert all constituents to IPA ──
    logger.info(f"loading CMUDict + reading {args.source}")
    conv = IPAConverter(drop_unconvertible=True)

    ipa_list: list[str] = []
    raw_list: list[str] = []
    n_scanned = 0
    n_dropped = 0
    with args.source.open() as f:
        for i, line in enumerate(f):
            if args.max_lines and i >= args.max_lines:
                break
            n_scanned += 1
            raw = line.strip()
            if not raw:
                continue
            ipa = conv.convert_line("eng", raw)
            if ipa is None:
                n_dropped += 1
                continue
            ipa = _clean_ipa(ipa)
            if len(ipa) < MIN_IPA_LEN:
                n_dropped += 1
                continue
            ipa_list.append(ipa)
            raw_list.append(raw)
            if n_scanned % 500_000 == 0:
                logger.info(f"  {n_scanned:,} scanned  {len(ipa_list):,} kept  {n_dropped:,} dropped")
    logger.info(f"final: {n_scanned:,} scanned, {len(ipa_list):,} kept ({len(ipa_list)/max(n_scanned,1):.1%}), {n_dropped:,} dropped")

    # ── 2. Train SPM on a random subset of IPA ──
    tmp_ipa_txt = args.out / "ipa_for_spm.txt"
    logger.info(f"writing SPM training text ({min(SPM_TRAIN_LINES, len(ipa_list)):,} lines)")
    rng = random.Random(42)
    spm_train = (
        rng.sample(ipa_list, SPM_TRAIN_LINES)
        if len(ipa_list) > SPM_TRAIN_LINES else ipa_list
    )
    tmp_ipa_txt.write_text("\n".join(spm_train))

    logger.info(f"training SentencePiece (vocab={SPM_VOCAB_SIZE})")
    spm.SentencePieceTrainer.Train(
        input=str(tmp_ipa_txt),
        model_prefix=str(SPM_MODEL_PREFIX),
        vocab_size=SPM_VOCAB_SIZE,
        model_type="bpe",
        character_coverage=1.0,
        pad_id=-1,
        unk_id=0,
        bos_id=-1,
        eos_id=-1,
        normalization_rule_name="identity",
        shuffle_input_sentence=True,
        input_sentence_size=len(spm_train),
    )
    tmp_ipa_txt.unlink()
    logger.info(f"wrote {SPM_MODEL_PREFIX}.model + .vocab")

    # ── 3. Write HDF5 in v7 schema ──
    logger.info(f"writing {OUT_H5}")
    str_dt = h5py.string_dtype(encoding="utf-8")
    n = len(ipa_list)
    ipa_lengths = np.asarray([len(s) for s in ipa_list], dtype=np.int32)
    logger.info(
        f"IPA length stats: min={ipa_lengths.min()} max={ipa_lengths.max()} "
        f"mean={ipa_lengths.mean():.1f} median={np.median(ipa_lengths):.0f} "
        f"p99={np.percentile(ipa_lengths, 99):.0f}",
    )
    with h5py.File(OUT_H5, "w") as h:
        g = h.create_group("samples")
        g.create_dataset("ipa", shape=(n,), dtype=str_dt, data=ipa_list)
        g.create_dataset("ipa_length", shape=(n,), dtype=np.int32, data=ipa_lengths)
        g.create_dataset("language", shape=(n,), dtype=str_dt, data=["eng"] * n)
        g.create_dataset("raw", shape=(n,), dtype=str_dt, data=raw_list)
        g.create_dataset("seq", shape=(n,), dtype=np.int64, data=np.arange(n, dtype=np.int64))
        g.create_dataset("source", shape=(n,), dtype=str_dt, data=["constituent"] * n)
        g.create_dataset("source_file", shape=(n,), dtype=str_dt, data=[str(args.source)] * n)
    logger.info("done.")


if __name__ == "__main__":
    main()

#!/usr/bin/env python
"""Transcode the 11.6M IPA phrase corpus to phoneme-id sequences.

Reads IPA strings from ``data/datasets/constituents-12lang-all/samples.h5``
and produces phoneme-id sequences using the multilingual-Latin alphabet
from ``data/phoneme_alphabet_multi.json``.

Token space
-----------
The phoneme alphabet has 50 entries (ids 0-49).  We reserve id=50 as a
``<ws>`` (word-boundary) marker so the VAE learns to produce
space-separated Neuroglot words rather than one flat phoneme stream.
Full vocab size is 51.

Implementation
--------------
Single-process.  The per-sample work is a trivial character loop
(microseconds), so the real bottleneck is h5py variable-length write
cost.  Workers add IPC overhead without throughput gain.  If this ever
becomes the critical path we can switch to a flat-array + offsets
layout for O(n) I/O, but at the current scale (~10 min wall time) it's
not worth the complexity.
"""

from __future__ import annotations

import json
import logging
import unicodedata
from collections import Counter
from pathlib import Path

import h5py
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger(__name__)

SOURCE_H5 = Path("data/datasets/constituents-12lang-all/samples.h5")
ALPHABET_PATH = Path("data/phoneme_alphabet_multi.json")
OUTPUT_DIR = Path("data/datasets/constituents-12lang-phonemes")
OUTPUT_H5 = OUTPUT_DIR / "samples.h5"
OUTPUT_MAPPING = OUTPUT_DIR / "mapping.json"
OUTPUT_MANIFEST = OUTPUT_DIR / "manifest.yaml"

CHUNK = 50_000
BUF_SIZE = 10_000


def transcode_text(
    text: str, mapping: dict[str, int], vocab_phonemes: int, wb_id: int,
) -> np.ndarray:
    """Convert an IPA string to a phoneme-id sequence.

    * Letters → phoneme id via ``mapping`` (fallback: codepoint modulo).
    * Runs of whitespace / hyphens → single ``<ws>`` token.
    * Other non-letter characters (digits, punctuation, diacritics) elided.
    * Leading/trailing ``<ws>`` stripped.
    """
    ids: list[int] = []
    prev_was_wb = True  # avoid emitting a leading <ws>
    for ch in text:
        cat = unicodedata.category(ch)
        if cat.startswith("L"):
            pid = mapping.get(ch)
            if pid is None:
                pid = ord(ch) % vocab_phonemes
            ids.append(pid)
            prev_was_wb = False
        elif ch.isspace() or ch == "-":
            if not prev_was_wb and ids:
                ids.append(wb_id)
                prev_was_wb = True
    while ids and ids[-1] == wb_id:
        ids.pop()
    return np.asarray(ids, dtype=np.int32)


def iter_source_chunks(h5_path: Path):
    with h5py.File(h5_path, "r") as f:
        g = f["samples"]
        n = g["ipa"].shape[0]
        for start in range(0, n, CHUNK):
            end = min(start + CHUNK, n)
            yield start, end, g["ipa"][start:end], g["language"][start:end]


def build_ipa_frequency_table() -> Counter[str]:
    counts: Counter[str] = Counter()
    total = 0
    with h5py.File(SOURCE_H5, "r") as f:
        n = f["samples/ipa"].shape[0]
    logger.info(f"scanning {n:,} samples for IPA character distribution...")
    for start, end, ipa_chunk, _ in iter_source_chunks(SOURCE_H5):
        for b in ipa_chunk:
            try:
                text = b.decode("utf-8") if isinstance(b, bytes) else b
            except UnicodeDecodeError:
                continue
            for ch in text:
                if unicodedata.category(ch).startswith("L"):
                    counts[ch] += 1
        total = end
        if end % 2_000_000 == 0 or end == n:
            logger.info(f"  scanned {end:,} / {n:,}")
    return counts


def build_mapping(ipa_counts: Counter[str], vocab_phonemes: int) -> dict[str, int]:
    return {
        ch: i % vocab_phonemes
        for i, (ch, _) in enumerate(ipa_counts.most_common())
    }


def main() -> None:
    if not SOURCE_H5.exists():
        raise FileNotFoundError(f"Source HDF5 not found: {SOURCE_H5}")
    with open(ALPHABET_PATH) as f:
        alphabet = json.load(f)
    vocab_phonemes = len(alphabet["phonemes"])
    wb_id = vocab_phonemes
    vocab_size = vocab_phonemes + 1
    logger.info(
        f"phonemes={vocab_phonemes}  word_boundary_id={wb_id}  "
        f"vocab_size={vocab_size}",
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # [1] frequency table + mapping
    logger.info("[1] counting IPA characters...")
    ipa_counts = build_ipa_frequency_table()
    logger.info(f"  {len(ipa_counts)} unique letter characters")
    mapping = build_mapping(ipa_counts, vocab_phonemes)

    with OUTPUT_MAPPING.open("w") as f:
        json.dump({
            "vocab_phonemes": vocab_phonemes,
            "word_boundary_id": wb_id,
            "vocab_size": vocab_size,
            "alphabet_path": str(ALPHABET_PATH),
            "ipa_char_to_phoneme_id": dict(mapping),
            "ipa_char_counts": dict(ipa_counts.most_common()),
        }, f, indent=2, ensure_ascii=False)
    logger.info(f"  wrote mapping → {OUTPUT_MAPPING}")

    # [2] transcode
    with h5py.File(SOURCE_H5, "r") as f:
        n = f["samples/ipa"].shape[0]

    logger.info(f"[2] transcoding {n:,} samples (single-process)...")
    vlen_int32 = h5py.special_dtype(vlen=np.dtype("int32"))

    with h5py.File(OUTPUT_H5, "w") as out:
        g = out.create_group("samples")
        ph_ds = g.create_dataset("phoneme_ids", shape=(n,), dtype=vlen_int32)
        len_ds = g.create_dataset("phoneme_length", shape=(n,), dtype=np.int32)
        lang_ds = g.create_dataset("language", shape=(n,), dtype=h5py.string_dtype())
        seq_ds = g.create_dataset("seq", shape=(n,), dtype=np.int64)
        g.attrs["vocab_size"] = vocab_size
        g.attrs["vocab_phonemes"] = vocab_phonemes
        g.attrs["word_boundary_id"] = wb_id
        g.attrs["alphabet_path"] = str(ALPHABET_PATH)

        idx = 0
        buf_seqs: list[np.ndarray] = []
        buf_lens: list[int] = []
        buf_langs: list[bytes] = []

        def flush() -> None:
            nonlocal idx
            end = idx + len(buf_seqs)
            # Variable-length writes: unavoidable per-sample loop.
            for i, s in enumerate(buf_seqs):
                ph_ds[idx + i] = s
            # Fixed-length batch writes (fast): lengths, languages, seq.
            len_ds[idx:end] = np.asarray(buf_lens, dtype=np.int32)
            lang_ds[idx:end] = buf_langs
            seq_ds[idx:end] = np.arange(idx, end, dtype=np.int64)
            idx = end
            buf_seqs.clear()
            buf_lens.clear()
            buf_langs.clear()

        for start, end, ipa_chunk, lang_chunk in iter_source_chunks(SOURCE_H5):
            for i in range(end - start):
                try:
                    text = ipa_chunk[i].decode("utf-8")
                except (AttributeError, UnicodeDecodeError):
                    text = ""
                seq = transcode_text(text, mapping, vocab_phonemes, wb_id)
                buf_seqs.append(seq)
                buf_lens.append(len(seq))
                lb = lang_chunk[i]
                buf_langs.append(
                    lb if isinstance(lb, bytes) else str(lb).encode(),
                )
                if len(buf_seqs) >= BUF_SIZE:
                    flush()
            if end % 500_000 == 0 or end == n:
                logger.info(f"  transcoded {end:,} / {n:,}")

        if buf_seqs:
            flush()

    logger.info(f"  wrote {idx:,} samples → {OUTPUT_H5}")

    import yaml
    with OUTPUT_MANIFEST.open("w") as f:
        yaml.safe_dump({
            "name": "constituents-12lang-phonemes",
            "description": "12-lang phrase constituents transcoded to "
                           "multilingual-Latin phoneme alphabet with <ws> "
                           "word-boundary marker",
            "source": str(SOURCE_H5),
            "alphabet_path": str(ALPHABET_PATH),
            "vocab_phonemes": vocab_phonemes,
            "word_boundary_id": wb_id,
            "vocab_size": vocab_size,
            "total_samples": n,
        }, f)
    logger.info(f"wrote manifest → {OUTPUT_MANIFEST}")


if __name__ == "__main__":
    main()

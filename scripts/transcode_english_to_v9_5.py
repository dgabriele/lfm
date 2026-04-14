#!/usr/bin/env python
"""Transcode English corpus → v9.5 token-id sequences for VAE pretraining.

Same flow as v9 transcoder, but the v9.5 alphabet includes both
Ġ-prefixed AND bare BPE tokens.  Each Qwen token id is looked up in
v9.5's index by its (qwen_id) — bare and Ġ variants live at distinct
positions, so the mapping is unambiguous.

Output: data/datasets/english-v9-5/samples.h5
"""

from __future__ import annotations

import json
import logging
from collections import Counter
from pathlib import Path

import h5py
import numpy as np
from transformers import AutoTokenizer

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger(__name__)

MODEL = "Qwen/Qwen2.5-0.5B"
ENGLISH_CORPUS = Path(
    "data/datasets/english-diverse-constituents/constituents.txt"
)
ALPHABET_PATH = Path("data/phoneme_alphabet_v9_5.json")
OUTPUT_DIR = Path("data/datasets/english-v9-5")
OUTPUT_H5 = OUTPUT_DIR / "samples.h5"
OUTPUT_MAPPING = OUTPUT_DIR / "mapping.json"
OUTPUT_MANIFEST = OUTPUT_DIR / "manifest.yaml"

MAX_LINES = 12_000_000   # 11.66M constituents available; cap above that
MIN_INVOCAB_FRACTION = 0.70  # higher than v9 (0.50) because v9.5 has way better coverage
MIN_LEN = 5
MAX_LEN = 256


def main() -> None:
    if not ENGLISH_CORPUS.exists():
        raise FileNotFoundError(f"english corpus not found: {ENGLISH_CORPUS}")
    if not ALPHABET_PATH.exists():
        raise FileNotFoundError(
            f"v9.5 alphabet not found: {ALPHABET_PATH}; run "
            f"scripts/design_alphabet_v9_5.py first",
        )

    alphabet = json.loads(ALPHABET_PATH.read_text())
    phonemes: list[str] = alphabet["phonemes"]
    qwen_token_ids: list[int] = alphabet["qwen_token_ids"]
    is_word_start: list[bool] = alphabet["is_word_start"]
    vocab_phonemes = len(qwen_token_ids)
    word_boundary_id = vocab_phonemes
    vocab_size = vocab_phonemes + 1
    qwen_to_v95: dict[int, int] = {tid: i for i, tid in enumerate(qwen_token_ids)}

    # Fallback lookup for greedy decomposition: (bare_str, is_word_start) → v95_id.
    bare_lookup: dict[tuple[str, bool], int] = {
        (p, ws): i for i, (p, ws) in enumerate(zip(phonemes, is_word_start))
    }
    # Bucket by first char + is_word_start for fast longest-match search.
    by_first: dict[tuple[str, bool], list[tuple[str, int]]] = {}
    for (p, ws), idx in bare_lookup.items():
        if not p:
            continue
        by_first.setdefault((p[0], ws), []).append((p, idx))
    for k in by_first:
        by_first[k].sort(key=lambda x: -len(x[0]))  # longest first

    n_ws = sum(is_word_start)
    logger.info(
        f"v9.5 alphabet: {vocab_phonemes} ({n_ws} Ġ + {vocab_phonemes - n_ws} bare), "
        f"word_boundary_id={word_boundary_id}, vocab_size={vocab_size}",
    )

    def decompose_unknown(qwen_token_str: str) -> list[int] | None:
        """Greedy longest-match decompose a Qwen token surface form into v9.5 ids.

        For "Ġunstoppable": peel "Ġ" → first piece must be is_word_start=True;
        try to find longest matching prefix at each position.  Subsequent pieces
        (after first match) are bare (is_word_start=False).
        """
        if qwen_token_str.startswith("Ġ"):
            target = qwen_token_str[1:]
            first_ws = True
        else:
            target = qwen_token_str
            first_ws = False
        out: list[int] = []
        pos = 0
        is_first = True
        while pos < len(target):
            ws_needed = first_ws if is_first else False
            ch = target[pos]
            cands = by_first.get((ch, ws_needed), [])
            chosen: int | None = None
            for cand_str, cand_id in cands:
                if target.startswith(cand_str, pos):
                    chosen = cand_id
                    pos += len(cand_str)
                    break
            if chosen is None:
                return None  # cannot decompose
            out.append(chosen)
            is_first = False
        return out

    logger.info(f"loading Qwen tokenizer: {MODEL}")
    tok = AutoTokenizer.from_pretrained(MODEL)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_MAPPING.write_text(json.dumps({
        "vocab_phonemes": vocab_phonemes,
        "word_boundary_id": word_boundary_id,
        "vocab_size": vocab_size,
        "alphabet_path": str(ALPHABET_PATH),
        "qwen_to_v95_id": {str(k): v for k, v in qwen_to_v95.items()},
    }, indent=2))
    logger.info(f"wrote mapping: {OUTPUT_MAPPING}")

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
            v95_ids: list[int] = []
            oov = 0
            for qid in qwen_ids:
                v = qwen_to_v95.get(qid)
                if v is not None:
                    v95_ids.append(v)
                    continue
                # Fallback: decompose the unknown Qwen token into v9.5 atoms.
                decomp = decompose_unknown(tok.convert_ids_to_tokens([qid])[0])
                if decomp is not None:
                    v95_ids.extend(decomp)
                else:
                    oov += 1
            if (1.0 - oov / len(qwen_ids)) < MIN_INVOCAB_FRACTION:
                n_oov_dropped += 1
                continue
            if len(v95_ids) < MIN_LEN:
                n_short_dropped += 1
                continue
            if len(v95_ids) > MAX_LEN:
                v95_ids = v95_ids[:MAX_LEN]
            v95_ids.append(word_boundary_id)
            sequences.append(np.asarray(v95_ids, dtype=np.int32))
            length_hist[len(v95_ids)] += 1
            n_kept += 1
            if n_scanned % 25_000 == 0:
                logger.info(
                    f"  scanned {n_scanned:,}  kept {n_kept:,}  "
                    f"oov-dropped {n_oov_dropped:,}  short {n_short_dropped:,}",
                )
    logger.info(
        f"final: scanned {n_scanned:,}, kept {n_kept:,} "
        f"({n_kept/n_scanned:.1%}), oov-dropped {n_oov_dropped:,}, "
        f"short {n_short_dropped:,}",
    )
    if n_kept == 0:
        raise RuntimeError("no sequences kept — check OOV threshold")

    lens = np.asarray([len(s) for s in sequences], dtype=np.int32)
    logger.info(
        f"length stats: min={lens.min()} max={lens.max()} "
        f"mean={lens.mean():.1f} median={np.median(lens):.1f} "
        f"p99={np.percentile(lens, 99):.0f}",
    )

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
        g.attrs["alphabet_version"] = "v9.5-english-pidgin-morphology"

    import yaml
    OUTPUT_MANIFEST.write_text(yaml.safe_dump({
        "name": "english-v9-5",
        "description": "English prose tokenized into v9.5 morphology alphabet "
                       "(Ġ-prefixed + bare Qwen BPE) for VAE pretraining.",
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

#!/usr/bin/env python
"""Build (IPA, BPE-token-seq) pairs for sub-word-output p2g.

Uses Qwen's BPE tokenizer to segment each English spelling into sub-word
units (proxy for morphemes — Qwen BPE tokens are largely morphological:
'orth', 'ography', 'un', 'able', 'tion').  The seq2seq's output vocab is
the set of Qwen-BPE tokens that appear when tokenizing our spelling
corpus, so any model emission is by construction a sequence of real
sub-words that concatenates to a real-looking English spelling.

Why Qwen BPE over morfessor:
  - Morfessor Baseline refused to segment our corpus (everything came
    out as a single morpheme regardless of corpusweight).
  - Qwen BPE is deterministic, trained on a vast English corpus, and
    aligns with the downstream LLM we'll feed the spellings to.
"""

from __future__ import annotations

import argparse
import json
import logging
from collections import Counter
from pathlib import Path

import h5py
import numpy as np
from transformers import AutoTokenizer

from lfm.data.loaders.ipa import IPAConverter

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger(__name__)

CORPUS = Path("data/translator/english_corpus.txt")
OUT_DIR = Path("data/datasets/p2g_bpe")
QWEN_MODEL = "Qwen/Qwen2.5-0.5B"
MIN_FREQ = 2
MIN_LEN = 2
MAX_LEN = 20
VAL_FRAC = 0.05
SEED = 42

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

    # ── 2. Tokenize spellings with Qwen BPE; collect used token set ──
    logger.info(f"loading Qwen BPE: {QWEN_MODEL}")
    tok = AutoTokenizer.from_pretrained(QWEN_MODEL)
    logger.info("BPE-segmenting spellings (no leading space → bare-mode tokens)")
    used_tokens: set[int] = set()
    word_to_qwen: dict[str, list[int]] = {}
    for w, _ in train_words:
        # Encode WITHOUT add_special_tokens; no leading space → bare BPE.
        ids = tok.encode(w, add_special_tokens=False)
        word_to_qwen[w] = ids
        used_tokens.update(ids)

    # ── 3. Local id space: specials + only Qwen ids actually used ──
    qwen_id_list = sorted(used_tokens)
    qwen_to_local = {qid: i + len(SPECIAL) for i, qid in enumerate(qwen_id_list)}
    local_vocab_size = len(SPECIAL) + len(qwen_id_list)
    logger.info(
        f"  local output vocab: {local_vocab_size:,} "
        f"({len(qwen_id_list):,} Qwen BPE tokens used + {len(SPECIAL)} specials)",
    )

    # Build readable string for each local id (for vocab json + decode).
    id_to_token_str: list[str] = list(SPECIAL)
    for qid in qwen_id_list:
        # convert_ids_to_tokens returns the raw BPE form (e.g. 'orth', 'ography', 'Ġthe').
        id_to_token_str.append(tok.convert_ids_to_tokens([qid])[0])

    # ── 4. IPA + map Qwen ids → local ids ──
    logger.info("converting to IPA + building (IPA, local-id-seq) pairs")
    conv = IPAConverter(drop_unconvertible=True)
    pairs: list[tuple[str, list[int], int, str]] = []
    n_oov = 0
    for w, c in train_words:
        ipa = conv._english_word_to_ipa(w)  # noqa: SLF001
        if ipa is None:
            n_oov += 1
            continue
        local_ids = [qwen_to_local[q] for q in word_to_qwen[w]]
        pairs.append((ipa, local_ids, c, w))
    logger.info(f"  kept {len(pairs):,} pairs (CMU OOV dropped: {n_oov:,})")

    # ── 5. split + write ──
    rng.shuffle(pairs)  # type: ignore[arg-type]
    n_val = max(1, int(len(pairs) * VAL_FRAC))
    val = pairs[:n_val]
    train = pairs[n_val:]
    logger.info(f"  train={len(train):,}  val={len(val):,}")

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
            seg_ds = g.create_dataset("token_ids", shape=(n,), dtype=vlen_int)
            for i, r in enumerate(rows):
                seg_ds[i] = np.asarray(r[1], dtype=np.int32)
            g.create_dataset("seg_len", shape=(n,), dtype=np.int32,
                             data=[len(r[1]) for r in rows])
            g.create_dataset("word_freq", shape=(n,), dtype=np.int32,
                             data=[r[2] for r in rows])
            g.create_dataset("spelling", shape=(n,), dtype=str_dt,
                             data=[r[3] for r in rows])
            g.attrs["source_corpus"] = str(args.corpus)
            g.attrs["min_freq"] = args.min_freq
            g.attrs["bpe_vocab_size"] = local_vocab_size
            g.attrs["qwen_model"] = QWEN_MODEL

    (args.out / "bpe_vocab.json").write_text(
        json.dumps({
            "tokens": id_to_token_str,
            "qwen_id_list": qwen_id_list,
            "specials": SPECIAL,
            "pad_id": PAD_ID, "bos_id": BOS_ID, "eos_id": EOS_ID,
            "qwen_model": QWEN_MODEL,
        }, ensure_ascii=False, indent=2),
    )

    # ── 6. diagnostics ──
    seg_lens = [len(r[1]) for r in pairs]
    logger.info(
        f"BPE units/word: mean={np.mean(seg_lens):.2f} "
        f"max={max(seg_lens)} p99={np.percentile(seg_lens, 99):.0f}",
    )
    one_token = sum(1 for s in seg_lens if s == 1)
    logger.info(f"  {one_token/len(seg_lens):.1%} of words are 1 BPE token")

    logger.info("frequent-word segmentations:")
    for w, c in sorted(train_words, key=lambda x: -x[1])[:15]:
        if w in word_to_qwen:
            toks = [tok.convert_ids_to_tokens([q])[0] for q in word_to_qwen[w]]
            logger.info(f"  {w!r:>14} ({c:,}) → {toks}")
    logger.info("morphology examples:")
    for w in ["orthography", "consciousness", "machine", "running",
              "unstoppable", "transformation", "philosophy", "unbelievable",
              "remarkable", "antidisestablishmentarianism"]:
        if w in word_to_qwen:
            toks = [tok.convert_ids_to_tokens([q])[0] for q in word_to_qwen[w]]
            logger.info(f"  {w!r:>30} → {toks}")
    logger.info("done.")


if __name__ == "__main__":
    main()

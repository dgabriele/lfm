#!/usr/bin/env python3
"""Fast contraction fix — pass through unchanged, parallel reparse changed.

Usage:
    poetry run python3 scripts/fix_contractions_fast.py \
        --input data/datasets/english-dep-trees-v16/sentences.jsonl \
        --output data/datasets/english-dep-trees-v16/sentences_fixed.jsonl \
        --workers 2 --batch-size 128 --device gpu
"""

from __future__ import annotations

import argparse
import json
import logging
import multiprocessing as mp
import os
import time
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger(__name__)


def parse_worker(args: tuple) -> str:
    """Worker: IPA convert + dep parse a chunk of changed sentences."""
    worker_id, chunk, out_path, batch_size, device = args

    from lfm.data.contractions import expand_contractions
    import epitran
    import stanza

    try:
        import orjson
        def dumps(obj):
            return orjson.dumps(obj, option=orjson.OPT_APPEND_NEWLINE).decode()
    except ImportError:
        def dumps(obj):
            return json.dumps(obj, ensure_ascii=False) + "\n"

    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    # IPA
    epi = epitran.Epitran("eng-Latn")
    to_parse = []
    failed_ipa = 0
    for raw, expanded in chunk:
        try:
            ipa = epi.transliterate(expanded)
            if ipa and len(ipa.split()) == len(expanded.split()):
                to_parse.append((raw, expanded, ipa))
            else:
                failed_ipa += 1
        except Exception:
            failed_ipa += 1

    logger.info("Worker %d: IPA done — %d aligned, %d failed", worker_id, len(to_parse), failed_ipa)

    # Dep parse
    stanza.download("en", processors="tokenize,pos,lemma,depparse", verbose=False)
    nlp = stanza.Pipeline("en", processors="tokenize,pos,lemma,depparse",
                          tokenize_pretokenized=True, use_gpu=(device == "gpu"), verbose=False)
    logger.info("Worker %d: Stanza loaded, parsing %d sentences...", worker_id, len(to_parse))

    parsed = 0
    failed_parse = 0
    t0 = time.time()

    with open(out_path, "w") as f:
        for i in range(0, len(to_parse), batch_size):
            batch = to_parse[i : i + batch_size]
            word_lists = [exp.split() for _, exp, _ in batch]
            try:
                doc = nlp(word_lists)
            except Exception:
                failed_parse += len(batch)
                continue
            if len(doc.sentences) != len(batch):
                failed_parse += len(batch)
                continue

            lines = []
            for (raw, expanded, ipa), sent in zip(batch, doc.sentences):
                ipa_words = ipa.split()
                if len(sent.words) != len(ipa_words):
                    failed_parse += 1
                    continue
                dep_labels = [w.deprel for w in sent.words]
                dep_heads = [w.head for w in sent.words]
                tagged_ipa = " ".join(f"[{l}] {w}" for l, w in zip(dep_labels, ipa_words))
                lines.append(dumps({
                    "raw_english": raw,
                    "english": expanded,
                    "ipa": ipa,
                    "dep_labels": dep_labels,
                    "dep_heads": dep_heads,
                    "tagged_ipa": tagged_ipa,
                }))
                parsed += 1
            if lines:
                f.write("".join(lines))

            if parsed > 0 and parsed % 10_000 < batch_size:
                rate = parsed / max(time.time() - t0, 1)
                logger.info("  Worker %d: %d/%d (%.0f/s)", worker_id, parsed, len(to_parse), rate)

    logger.info("Worker %d: DONE — %d parsed, %d failed", worker_id, parsed, failed_parse)
    return out_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--device", default="gpu", choices=["gpu", "cpu"])
    args = parser.parse_args()

    from lfm.data.contractions import expand_contractions

    try:
        import orjson
        def dumps(obj):
            return orjson.dumps(obj, option=orjson.OPT_APPEND_NEWLINE).decode()
    except ImportError:
        def dumps(obj):
            return json.dumps(obj, ensure_ascii=False) + "\n"

    # Scan: pass through unchanged, collect changed
    logger.info("Scanning %s ...", args.input)
    unchanged_path = args.output.with_suffix(".unchanged.jsonl")
    changed: list[tuple[str, str]] = []
    unchanged_count = 0

    with open(args.input) as fin, open(unchanged_path, "w") as fout:
        for line in fin:
            rec = json.loads(line)
            raw = rec["english"]
            expanded = expand_contractions(raw)
            if expanded == raw:
                rec["raw_english"] = raw
                fout.write(dumps(rec))
                unchanged_count += 1
            else:
                changed.append((raw, expanded))
            if (unchanged_count + len(changed)) % 2_000_000 == 0:
                logger.info("  %dM scanned...", (unchanged_count + len(changed)) // 1_000_000)

    logger.info(
        "Scan done: %d unchanged, %d need reprocessing",
        unchanged_count, len(changed),
    )

    if not changed:
        os.rename(unchanged_path, args.output)
        logger.info("No contractions. Done.")
        return

    # Split changed across workers
    n = len(changed)
    w = args.workers
    worker_args = []
    for i in range(w):
        lo = i * n // w
        hi = (i + 1) * n // w
        chunk_path = args.output.with_suffix(f".chunk{i}.jsonl")
        worker_args.append((i, changed[lo:hi], str(chunk_path), args.batch_size, args.device))

    logger.info("Launching %d workers for %d changed sentences...", w, n)
    ctx = mp.get_context("spawn")
    with ctx.Pool(w) as pool:
        chunk_paths = pool.map(parse_worker, worker_args)

    # Merge: unchanged + all chunks
    logger.info("Merging...")
    total = 0
    with open(args.output, "w") as fout:
        with open(unchanged_path) as f:
            for line in f:
                fout.write(line)
                total += 1
        for cp in chunk_paths:
            with open(cp) as f:
                for line in f:
                    fout.write(line)
                    total += 1
            os.unlink(cp)
    os.unlink(unchanged_path)

    logger.info("Done: %d total sentences in %s", total, args.output)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Parallel dependency parsing with multiple GPU workers.

Each worker loads its own Stanza pipeline and processes a chunk of
sentences. Workers share the GPU via CUDA's time-slicing. Results
are written to separate chunk files, then merged.

Usage:
    # Vast (3 workers, sentences 0-1M):
    python3.11 scripts/parse_dep_parallel.py --workers 3 --start 0 --end 1000000

    # Local (2 workers, sentences 1M-2M):
    poetry run python3 scripts/parse_dep_parallel.py --workers 2 --start 1000000 --end 2000000

    # Merge:
    python3.11 scripts/parse_dep_parallel.py --merge
"""

from __future__ import annotations

import argparse
import json
import logging
import multiprocessing as mp
import os
import random
import time
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(message)s",
)
logger = logging.getLogger(__name__)

PAIRED_FILE = Path("data/datasets/english-sentences-v15/ipa_sentences.paired.txt")
OUT_DIR = Path("data/datasets/english-dep-trees-v16")
SEED = 42
MIN_WORDS = 5
MAX_WORDS = 30


def load_eligible_pairs() -> list[tuple[str, str]]:
    """Load all eligible sentence pairs (deterministic order)."""
    pairs = []
    with open(PAIRED_FILE) as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) != 2:
                continue
            english, ipa = parts
            ew = english.split()
            iw = ipa.split()
            if MIN_WORDS <= len(ew) <= MAX_WORDS and len(ew) == len(iw):
                pairs.append((english, ipa))
    return pairs


def sample_pairs(pairs: list[tuple[str, str]], target: int) -> list[tuple[str, str]]:
    """Deterministic reservoir sample."""
    rng = random.Random(SEED)
    if len(pairs) <= target:
        result = list(pairs)
    else:
        result = list(pairs[:target])
        for i in range(target, len(pairs)):
            j = rng.randint(0, i)
            if j < target:
                result[j] = pairs[i]
    rng.shuffle(result)
    return result


def worker_fn(args: tuple) -> None:
    """Single worker: load Stanza, parse chunk, write to chunk file."""
    worker_id, chunk, out_path, batch_size, device = args

    try:
        import orjson
        def dumps(obj):
            return orjson.dumps(obj, option=orjson.OPT_APPEND_NEWLINE).decode()
    except ImportError:
        def dumps(obj):
            return json.dumps(obj, ensure_ascii=False) + "\n"

    import stanza
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    logger.info(f"Worker {worker_id}: loading Stanza ({len(chunk)} sentences)...")
    stanza.download("en", processors="tokenize,pos,lemma,depparse", verbose=False)
    nlp = stanza.Pipeline(
        "en",
        processors="tokenize,pos,lemma,depparse",
        tokenize_pretokenized=True,
        use_gpu=(device == "gpu"),
        verbose=False,
    )
    logger.info(f"Worker {worker_id}: Stanza loaded, parsing...")

    parsed = 0
    failed = 0
    t0 = time.time()

    with open(out_path, "w") as f:
        for i in range(0, len(chunk), batch_size):
            batch = chunk[i : i + batch_size]
            word_lists = [eng.split() for eng, _ in batch]

            try:
                doc = nlp(word_lists)
            except Exception as e:
                logger.warning(f"Worker {worker_id}: batch failed at {i}: {e}")
                failed += len(batch)
                continue

            if len(doc.sentences) != len(batch):
                failed += len(batch)
                continue

            lines = []
            for (eng, ipa), sent in zip(batch, doc.sentences):
                ipa_words = ipa.split()
                words = sent.words
                if len(words) != len(ipa_words):
                    failed += 1
                    continue

                dep_labels = [w.deprel for w in words]
                dep_heads = [w.head for w in words]
                tagged_ipa = " ".join(
                    f"[{lab}] {iw}" for lab, iw in zip(dep_labels, ipa_words)
                )
                lines.append(dumps({
                    "english": eng,
                    "ipa": ipa,
                    "dep_labels": dep_labels,
                    "dep_heads": dep_heads,
                    "tagged_ipa": tagged_ipa,
                }))
                parsed += 1

            if lines:
                f.write("".join(lines))

            if parsed > 0 and parsed % 10_000 < batch_size:
                elapsed = time.time() - t0
                rate = parsed / max(elapsed, 1)
                remaining = (len(chunk) - i - batch_size) / max(rate, 1)
                logger.info(
                    f"  Worker {worker_id}: {parsed:,}/{len(chunk):,} "
                    f"({rate:.0f} sent/sec, ETA {remaining/60:.0f}m)"
                )

    elapsed = time.time() - t0
    rate = parsed / max(elapsed, 1)
    logger.info(
        f"Worker {worker_id}: DONE — {parsed:,} parsed, {failed:,} failed, "
        f"{rate:.0f} sent/sec, {elapsed/60:.1f}m"
    )


def merge_chunks(out_dir: Path) -> None:
    """Merge all chunk_*.jsonl files into sentences.jsonl + tagged_ipa.txt."""
    chunk_files = sorted(out_dir.glob("chunk_*.jsonl"))
    if not chunk_files:
        logger.error("No chunk files found to merge")
        return

    jsonl_out = out_dir / "sentences.jsonl"
    tagged_out = out_dir / "tagged_ipa.txt"
    total = 0

    with open(jsonl_out, "w") as jf, open(tagged_out, "w") as tf:
        for cf in chunk_files:
            logger.info(f"Merging {cf.name}...")
            with open(cf) as f:
                for line in f:
                    jf.write(line)
                    rec = json.loads(line)
                    tf.write(rec["tagged_ipa"] + "\n")
                    total += 1

    logger.info(f"Merged {total:,} sentences from {len(chunk_files)} chunks")


def main():
    parser = argparse.ArgumentParser(description="Parallel dependency parsing")
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--start", type=int, default=0,
                        help="Start index in the sampled sentence list")
    parser.add_argument("--end", type=int, default=2_000_000,
                        help="End index in the sampled sentence list")
    parser.add_argument("--target", type=int, default=4_000_000,
                        help="Total sample pool size (must match across runs)")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--device", default="gpu", choices=["gpu", "cpu"])
    parser.add_argument("--merge", action="store_true",
                        help="Merge chunk files instead of parsing")
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    if args.merge:
        merge_chunks(OUT_DIR)
        return

    # Load and sample (deterministic — same result on both machines)
    logger.info("Loading eligible pairs...")
    all_pairs = load_eligible_pairs()
    logger.info(f"  {len(all_pairs):,} eligible pairs")

    sampled = sample_pairs(all_pairs, args.target)
    logger.info(f"  sampled {len(sampled):,} (target={args.target:,})")

    # Slice to our range
    chunk = sampled[args.start : args.end]
    logger.info(f"  this run: [{args.start:,}, {args.end:,}) = {len(chunk):,} sentences")

    # Split across workers
    n = len(chunk)
    w = args.workers
    chunks = []
    for i in range(w):
        lo = i * n // w
        hi = (i + 1) * n // w
        out_path = OUT_DIR / f"chunk_{args.start + lo}_{args.start + hi}.jsonl"
        chunks.append((i, chunk[lo:hi], str(out_path), args.batch_size, args.device))

    logger.info(f"Launching {w} workers...")

    # Use spawn to avoid CUDA fork issues
    ctx = mp.get_context("spawn")
    with ctx.Pool(w) as pool:
        pool.map(worker_fn, chunks)

    logger.info("All workers done.")


if __name__ == "__main__":
    main()

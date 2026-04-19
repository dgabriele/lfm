#!/usr/bin/env python3
"""Parse 2M English sentences with Stanza dependency parser.

Reads English-IPA paired sentences, filters to 5-30 words, samples 2M,
parses each with Stanza dependency parser (GPU, pretokenized), and writes
JSONL with dependency labels/heads plus tagged IPA.

Output: data/datasets/english-dep-trees-v16/sentences.jsonl
        data/datasets/english-dep-trees-v16/tagged_ipa.txt

Usage:
    poetry run python3 scripts/parse_dependency_trees.py [--target 2000000] [--batch-size 128]
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
import sys
import time
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(message)s",
    handlers=[logging.StreamHandler(sys.stderr)],
)
logger = logging.getLogger(__name__)

PAIRED_FILE = Path("data/datasets/english-sentences-v15/ipa_sentences.paired.txt")
OUT_DIR = Path("data/datasets/english-dep-trees-v16")
JSONL_FILE = OUT_DIR / "sentences.jsonl"
TAGGED_IPA_FILE = OUT_DIR / "tagged_ipa.txt"
PROGRESS_FILE = OUT_DIR / "progress.jsonl"

MIN_WORDS = 5
MAX_WORDS = 30
SEED = 42


def load_and_sample(target: int) -> list[tuple[str, str]]:
    """Load paired file, filter by word count, sample target sentences.

    Uses reservoir sampling for memory-efficient random selection from
    the 16.5M-line paired file.

    Returns list of (english, ipa) tuples.
    """
    logger.info(f"Loading paired sentences from {PAIRED_FILE}")
    logger.info(f"Filter: {MIN_WORDS}-{MAX_WORDS} words, target: {target:,}")

    random.seed(SEED)
    reservoir: list[tuple[str, str]] = []
    total_eligible = 0
    line_no = 0

    with open(PAIRED_FILE) as f:
        for line_no, line in enumerate(f):
            line = line.rstrip("\n")
            parts = line.split("\t")
            if len(parts) != 2:
                continue

            english, ipa = parts
            eng_words = english.split()
            ipa_words = ipa.split()

            # Filter: word count range + alignment check
            if not (MIN_WORDS <= len(eng_words) <= MAX_WORDS):
                continue
            if len(eng_words) != len(ipa_words):
                continue

            total_eligible += 1

            # Reservoir sampling
            if len(reservoir) < target:
                reservoir.append((english, ipa))
            else:
                j = random.randint(0, total_eligible - 1)
                if j < target:
                    reservoir[j] = (english, ipa)

            if (line_no + 1) % 2_000_000 == 0:
                logger.info(
                    f"  scanned {line_no + 1:,} lines, "
                    f"{total_eligible:,} eligible, "
                    f"reservoir: {len(reservoir):,}"
                )

    logger.info(
        f"Scan complete: {line_no + 1:,} lines, "
        f"{total_eligible:,} eligible, "
        f"sampled: {len(reservoir):,}"
    )

    # Shuffle the reservoir so processing order is random
    random.shuffle(reservoir)
    return reservoir


def parse_batch_pretokenized(nlp, sentences: list[str]) -> list:
    """Parse a batch of pretokenized sentences with Stanza dependency parser.

    Uses tokenize_pretokenized=True, passing sentences as list of word lists.
    Returns list of parsed sentence objects.
    """
    # Build list of word lists for pretokenized input
    word_lists = [sent.split() for sent in sentences]

    doc = nlp(word_lists)
    return doc.sentences


def process_sentence(
    english: str, ipa: str, parsed_sent
) -> dict | None:
    """Convert a Stanza-parsed sentence into the output JSON format.

    Returns dict with english, ipa, dep_labels, dep_heads, tagged_ipa,
    or None if the parse doesn't align with the input.
    """
    eng_words = english.split()
    ipa_words = ipa.split()

    # Extract dependency labels and heads from parsed sentence
    dep_labels: list[str] = []
    dep_heads: list[int] = []
    parsed_words: list[str] = []

    for word in parsed_sent.words:
        parsed_words.append(word.text)
        dep_labels.append(word.deprel)
        dep_heads.append(word.head)  # 1-indexed, 0 = ROOT

    # Verify alignment: parsed tokens must match input tokens
    if len(parsed_words) != len(eng_words):
        return None

    # Check tokens actually match (Stanza shouldn't change them with
    # pretokenized input, but verify anyway)
    for pw, ew in zip(parsed_words, eng_words):
        if pw != ew:
            return None

    # Build tagged IPA: interleave [dep_label] before each IPA word
    tagged_parts = []
    for label, iw in zip(dep_labels, ipa_words):
        tagged_parts.append(f"[{label}]")
        tagged_parts.append(iw)
    tagged_ipa = " ".join(tagged_parts)

    return {
        "english": english,
        "ipa": ipa,
        "dep_labels": dep_labels,
        "dep_heads": dep_heads,
        "tagged_ipa": tagged_ipa,
    }


def count_existing_progress(path: Path) -> int:
    """Count how many lines have already been written (for resume)."""
    if not path.exists():
        return 0
    count = 0
    with open(path) as f:
        for _ in f:
            count += 1
    return count


def main():
    parser = argparse.ArgumentParser(
        description="Parse English dependency trees with Stanza"
    )
    parser.add_argument(
        "--target",
        type=int,
        default=2_000_000,
        help="Target number of sentences to parse",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=128,
        help="Stanza batch size (number of sentences per GPU call)",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from existing output file",
    )
    parser.add_argument(
        "--device",
        default="gpu",
        choices=["gpu", "cpu"],
        help="Device for Stanza (gpu or cpu)",
    )
    args = parser.parse_args()

    # Create output directory
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load and sample sentences
    pairs = load_and_sample(args.target)
    if not pairs:
        logger.error("No sentences found!")
        return

    logger.info(f"Sampled {len(pairs):,} sentences for parsing")

    # Handle resume
    skip = 0
    if args.resume:
        skip = count_existing_progress(JSONL_FILE)
        if skip > 0:
            logger.info(f"Resuming: skipping first {skip:,} already-parsed sentences")
            if skip >= len(pairs):
                logger.info("All sentences already parsed!")
                return

    # Initialize Stanza pipeline
    import stanza

    use_gpu = args.device == "gpu"
    logger.info(
        f"Loading Stanza English dependency parser (device={args.device})..."
    )
    stanza.download("en", processors="tokenize,pos,lemma,depparse", verbose=False)
    nlp = stanza.Pipeline(
        "en",
        processors="tokenize,pos,lemma,depparse",
        tokenize_pretokenized=True,
        use_gpu=use_gpu,
        verbose=False,
    )
    logger.info("Stanza pipeline loaded")

    # Parse in batches
    total = len(pairs)
    batch_size = args.batch_size
    parsed_count = skip
    failed_count = 0
    t0 = time.time()
    last_checkpoint_count = skip
    last_log_count = skip

    mode = "a" if args.resume and skip > 0 else "w"
    jsonl_f = open(JSONL_FILE, mode)
    tagged_f = open(TAGGED_IPA_FILE, mode)

    try:
        for batch_start in range(skip, total, batch_size):
            batch_end = min(batch_start + batch_size, total)
            batch_pairs = pairs[batch_start:batch_end]
            batch_english = [eng for eng, _ in batch_pairs]

            try:
                parsed_sents = parse_batch_pretokenized(nlp, batch_english)
            except Exception as e:
                logger.warning(
                    f"Batch parse failed at {batch_start}: {e}"
                )
                failed_count += len(batch_pairs)
                continue

            # Verify count alignment
            if len(parsed_sents) != len(batch_pairs):
                logger.debug(
                    f"Sentence count mismatch: {len(parsed_sents)} parsed for "
                    f"{len(batch_pairs)} input at batch {batch_start}. "
                    f"Falling back to one-by-one."
                )
                for eng, ipa in batch_pairs:
                    try:
                        single_parsed = parse_batch_pretokenized(nlp, [eng])
                        if single_parsed:
                            result = process_sentence(eng, ipa, single_parsed[0])
                            if result:
                                jsonl_f.write(
                                    json.dumps(result, ensure_ascii=False) + "\n"
                                )
                                tagged_f.write(result["tagged_ipa"] + "\n")
                                parsed_count += 1
                            else:
                                failed_count += 1
                        else:
                            failed_count += 1
                    except Exception:
                        failed_count += 1
            else:
                for (eng, ipa), parsed_sent in zip(batch_pairs, parsed_sents):
                    result = process_sentence(eng, ipa, parsed_sent)
                    if result:
                        jsonl_f.write(
                            json.dumps(result, ensure_ascii=False) + "\n"
                        )
                        tagged_f.write(result["tagged_ipa"] + "\n")
                        parsed_count += 1
                    else:
                        failed_count += 1

            # Log progress every 50K sentences
            if parsed_count - last_log_count >= 50_000:
                elapsed = time.time() - t0
                done_this_run = parsed_count - skip
                rate = done_this_run / max(elapsed, 1e-9)
                remaining = (total - parsed_count) / max(rate, 1e-9)
                logger.info(
                    f"  [{parsed_count:,}/{total:,}] "
                    f"{done_this_run:,} parsed this run, "
                    f"{failed_count:,} failed | "
                    f"{rate:.1f} sent/sec | "
                    f"ETA {remaining / 3600:.1f}h"
                )
                last_log_count = parsed_count

            # Incremental checkpoint every 500K sentences
            if parsed_count - last_checkpoint_count >= 500_000:
                jsonl_f.flush()
                os.fsync(jsonl_f.fileno())
                tagged_f.flush()
                os.fsync(tagged_f.fileno())
                last_checkpoint_count = parsed_count

                elapsed = time.time() - t0
                done_this_run = parsed_count - skip
                rate = done_this_run / max(elapsed, 1e-9)
                logger.info(
                    f"  ** CHECKPOINT at {parsed_count:,} ** "
                    f"({done_this_run:,} this run, "
                    f"{rate:.1f} sent/sec, "
                    f"{elapsed / 60:.1f} min elapsed)"
                )
                with open(PROGRESS_FILE, "a") as pf:
                    pf.write(
                        json.dumps(
                            {
                                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                                "parsed": parsed_count,
                                "failed": failed_count,
                                "total": total,
                                "elapsed_min": round(elapsed / 60, 1),
                                "rate_per_sec": round(rate, 1),
                            }
                        )
                        + "\n"
                    )

    except KeyboardInterrupt:
        logger.info(
            f"Interrupted! Saving progress at {parsed_count:,} sentences..."
        )
    finally:
        jsonl_f.flush()
        os.fsync(jsonl_f.fileno())
        jsonl_f.close()
        tagged_f.flush()
        os.fsync(tagged_f.fileno())
        tagged_f.close()

    elapsed = time.time() - t0
    done_this_run = parsed_count - skip
    rate = done_this_run / max(elapsed, 1e-9)
    logger.info(
        f"Done: {parsed_count:,} parsed, {failed_count:,} failed "
        f"in {elapsed / 60:.1f} min ({rate:.1f} sent/sec)"
    )
    logger.info(f"Output JSONL: {JSONL_FILE}")
    logger.info(f"Output tagged IPA: {TAGGED_IPA_FILE}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Parse 2M English sentences with Stanza constituency parser.

Reads English-IPA paired sentences, filters to 5-30 words, samples 2M,
parses each with Stanza constituency parser (GPU), and writes JSONL with
full linearized bracket trees, phrase spans, and paired IPA.

Output: data/datasets/english-constituency-trees-v16/trees.jsonl

Usage:
    poetry run python3 scripts/parse_constituency_trees.py [--target 2000000] [--batch-size 64]
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
OUT_DIR = Path("data/datasets/english-constituency-trees-v16")
OUT_FILE = OUT_DIR / "trees.jsonl"
PROGRESS_FILE = OUT_DIR / "progress.jsonl"

MIN_WORDS = 5
MAX_WORDS = 30
SEED = 42


def linearize_tree(node) -> str:
    """Convert a Stanza constituency tree to bracket notation string.

    E.g.: (S (NP (DT the) (NN cat)) (VP (VBD sat)))
    """
    label = node.label
    children = node.children

    if not children:
        # Leaf node (word) — label is the word itself
        return label

    parts = []
    for child in children:
        parts.append(linearize_tree(child))

    return f"({label} {' '.join(parts)})"


def extract_phrase_spans(node, offset: int = 0) -> tuple[list[dict], int]:
    """Extract phrase spans with word-level start/end indices.

    Returns (spans_list, num_leaves_under_this_node).
    """
    label = node.label
    children = node.children

    if not children:
        # Leaf node — occupies 1 word position
        return [], 1

    spans = []
    current_offset = offset
    total_leaves = 0

    for child in children:
        child_spans, child_leaves = extract_phrase_spans(child, current_offset)
        spans.extend(child_spans)
        current_offset += child_leaves
        total_leaves += child_leaves

    # Record this node's span (skip ROOT if present)
    if label != "ROOT":
        spans.append({
            "type": label,
            "start": offset,
            "end": offset + total_leaves,
        })

    return spans, total_leaves


def get_top_level_children(node) -> list[str]:
    """Get labels of top-level children of the root S node.

    Navigates through ROOT -> S (or similar) to find the main clause's
    direct children phrase labels (NP, VP, etc.).
    """
    # Stanza trees typically have ROOT -> S -> children
    children = node.children
    if not children:
        return []

    # If this is ROOT, descend to its child
    if node.label == "ROOT" and len(children) == 1:
        return get_top_level_children(children[0])

    # Return labels of children that are phrases (not POS tags over single words)
    result = []
    for child in children:
        if child.children and len(child.children) > 0:
            # Check if it's a phrase (has grandchildren) vs POS tag (single leaf child)
            if any(gc.children for gc in child.children):
                result.append(child.label)
            elif len(child.children) > 1:
                # Multiple leaf children still counts as a phrase
                result.append(child.label)
            else:
                # Single-word "phrase" — include POS tag labeled nodes too
                result.append(child.label)
        else:
            # Bare word at top level — rare but possible
            pass

    return result


def get_leaf_words(node) -> list[str]:
    """Extract leaf words from a parse tree in order."""
    if not node.children:
        return [node.label]
    words = []
    for child in node.children:
        words.extend(get_leaf_words(child))
    return words


def load_and_sample(target: int) -> list[tuple[str, str]]:
    """Load paired file, filter by word count, sample target sentences.

    Uses reservoir sampling for memory-efficient random selection.

    Returns list of (english, ipa) tuples.
    """
    logger.info(f"Loading paired sentences from {PAIRED_FILE}")
    logger.info(f"Filter: {MIN_WORDS}-{MAX_WORDS} words, target: {target:,}")

    random.seed(SEED)
    reservoir: list[tuple[str, str]] = []
    total_seen = 0
    total_eligible = 0

    with open(PAIRED_FILE) as f:
        for line_no, line in enumerate(f):
            line = line.rstrip("\n")
            parts = line.split("\t")
            if len(parts) != 2:
                continue

            english, ipa = parts
            words = english.split()
            if not (MIN_WORDS <= len(words) <= MAX_WORDS):
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


def parse_batch_stanza(nlp, sentences: list[str]) -> list:
    """Parse a batch of sentences with Stanza, return constituency trees.

    Stanza expects sentences separated by double newlines for batch processing.
    """
    # Join sentences with double newlines so Stanza treats each as separate
    doc = nlp("\n\n".join(sentences))

    trees = []
    for sent in doc.sentences:
        trees.append(sent.constituency)

    return trees


def process_tree(english: str, ipa: str, tree) -> dict | None:
    """Convert a Stanza tree into the output JSON format."""
    if tree is None:
        return None

    try:
        linearized = linearize_tree(tree)
        spans, _ = extract_phrase_spans(tree)
        top_children = get_top_level_children(tree)
        leaf_words = get_leaf_words(tree)

        # Verify the tree leaves match the input words (Stanza may retokenize)
        # We'll store whatever the parser produced as the canonical tokenization
        return {
            "english": english,
            "ipa": ipa,
            "tree": linearized,
            "tokens": leaf_words,
            "phrase_spans": spans,
            "phrase_sequence": top_children,
        }
    except Exception as e:
        logger.debug(f"Failed to process tree for: {english[:60]}... — {e}")
        return None


def count_existing_progress() -> int:
    """Count how many sentences have already been parsed (for resume)."""
    if not OUT_FILE.exists():
        return 0
    count = 0
    with open(OUT_FILE) as f:
        for _ in f:
            count += 1
    return count


def main():
    parser = argparse.ArgumentParser(description="Parse English constituency trees")
    parser.add_argument("--target", type=int, default=2_000_000,
                        help="Target number of sentences to parse")
    parser.add_argument("--batch-size", type=int, default=128,
                        help="Stanza batch size (128 optimal for RTX 3060 Ti)")
    parser.add_argument("--resume", action="store_true",
                        help="Resume from existing output file")
    parser.add_argument("--device", default="gpu",
                        choices=["gpu", "cpu"],
                        help="Device for Stanza (gpu or cpu)")
    args = parser.parse_args()

    # Create output directory
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load and sample sentences
    pairs = load_and_sample(args.target)
    if not pairs:
        logger.error("No sentences found!")
        return

    # Handle resume
    skip = 0
    if args.resume:
        skip = count_existing_progress()
        if skip > 0:
            logger.info(f"Resuming: skipping first {skip:,} already-parsed sentences")
            if skip >= len(pairs):
                logger.info("All sentences already parsed!")
                return

    # Initialize Stanza pipeline
    import stanza

    use_gpu = args.device == "gpu"
    logger.info(f"Loading Stanza English constituency parser (device={args.device})...")
    stanza.download("en", processors="tokenize,pos,constituency", verbose=False)
    nlp = stanza.Pipeline(
        "en",
        processors="tokenize,pos,constituency",
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
    last_save_count = skip

    mode = "a" if args.resume and skip > 0 else "w"
    outf = open(OUT_FILE, mode)

    try:
        for batch_start in range(skip, total, batch_size):
            batch_end = min(batch_start + batch_size, total)
            batch_pairs = pairs[batch_start:batch_end]
            batch_english = [eng for eng, _ in batch_pairs]

            try:
                trees = parse_batch_stanza(nlp, batch_english)
            except Exception as e:
                logger.warning(f"Batch parse failed at {batch_start}: {e}")
                failed_count += len(batch_pairs)
                continue

            # Sometimes Stanza splits sentences differently than expected.
            # If tree count doesn't match, fall back to one-by-one parsing.
            if len(trees) != len(batch_pairs):
                logger.debug(
                    f"Tree count mismatch: {len(trees)} trees for "
                    f"{len(batch_pairs)} sentences at batch {batch_start}. "
                    f"Falling back to one-by-one."
                )
                # Re-parse one by one
                for eng, ipa in batch_pairs:
                    try:
                        single_trees = parse_batch_stanza(nlp, [eng])
                        if single_trees:
                            result = process_tree(eng, ipa, single_trees[0])
                            if result:
                                outf.write(json.dumps(result, ensure_ascii=False) + "\n")
                                parsed_count += 1
                            else:
                                failed_count += 1
                        else:
                            failed_count += 1
                    except Exception:
                        failed_count += 1
            else:
                for (eng, ipa), tree in zip(batch_pairs, trees):
                    result = process_tree(eng, ipa, tree)
                    if result:
                        outf.write(json.dumps(result, ensure_ascii=False) + "\n")
                        parsed_count += 1
                    else:
                        failed_count += 1

            # Flush periodically
            if parsed_count - last_save_count >= 10_000:
                outf.flush()

            # Log progress every ~10K sentences
            elapsed = time.time() - t0
            done_this_run = parsed_count - skip
            if done_this_run > 0 and done_this_run % 10_000 < batch_size:
                rate = done_this_run / elapsed
                remaining = (total - parsed_count) / max(rate, 1e-9)
                logger.info(
                    f"  [{parsed_count:,}/{total:,}] "
                    f"{done_this_run:,} parsed this run, "
                    f"{failed_count:,} failed | "
                    f"{rate:.1f} sent/sec | "
                    f"ETA {remaining/3600:.1f}h"
                )

            # Incremental save checkpoint every 100K
            if parsed_count - last_save_count >= 100_000:
                outf.flush()
                os.fsync(outf.fileno())
                last_save_count = parsed_count
                elapsed = time.time() - t0
                done_this_run = parsed_count - skip
                rate = done_this_run / max(elapsed, 1e-9)
                logger.info(
                    f"  ** CHECKPOINT at {parsed_count:,} ** "
                    f"({done_this_run:,} this run, "
                    f"{rate:.1f} sent/sec, "
                    f"{elapsed/60:.1f} min elapsed)"
                )
                # Write progress metadata
                with open(PROGRESS_FILE, "a") as pf:
                    pf.write(json.dumps({
                        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                        "parsed": parsed_count,
                        "failed": failed_count,
                        "total": total,
                        "elapsed_min": round(elapsed / 60, 1),
                        "rate_per_sec": round(rate, 1),
                    }) + "\n")

    except KeyboardInterrupt:
        logger.info(f"Interrupted! Saving progress at {parsed_count:,} sentences...")
    finally:
        outf.flush()
        os.fsync(outf.fileno())
        outf.close()

    elapsed = time.time() - t0
    done_this_run = parsed_count - skip
    rate = done_this_run / max(elapsed, 1e-9)
    logger.info(
        f"Done: {parsed_count:,} parsed, {failed_count:,} failed "
        f"in {elapsed/60:.1f} min ({rate:.1f} sent/sec)"
    )
    logger.info(f"Output: {OUT_FILE}")


if __name__ == "__main__":
    main()

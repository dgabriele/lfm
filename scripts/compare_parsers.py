#!/usr/bin/env python
"""Side-by-side quality comparison: Stanza vs benepar constituency parsing.

Runs both parsers (CPU-only so it never contends with any running GPU
workload) on the same small set of sentences and prints their output
side by side plus per-parser throughput.  No HDF5/SPM/corpus side
effects — this is strictly a parser bake-off.

Usage::

    # Pick 30 random sentences from the current corpus and compare:
    poetry run python scripts/compare_parsers.py \\
        --sentences-file data/datasets/english-constituents-v13/sentences.txt \\
        --num-samples 30
"""

from __future__ import annotations

import argparse
import logging
import random
import sys
import time
from pathlib import Path

# Use CPU only — avoid GPU contention with any running Stanza parse.
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────
# Stanza
# ──────────────────────────────────────────────────────────────────────
def run_stanza(sentences: list[str]) -> tuple[list[str | None], float]:
    """Parse each sentence with Stanza.  Returns (tree_strings, seconds)."""
    import stanza

    logger.info("loading stanza (CPU)...")
    pipe = stanza.Pipeline(
        lang="en",
        processors="tokenize,pos,constituency",
        use_gpu=False,
        tokenize_pretokenized=False,
        verbose=False,
    )
    trees: list[str | None] = []
    t0 = time.time()
    for s in sentences:
        try:
            doc = pipe(s)
            if doc.sentences:
                trees.append(str(doc.sentences[0].constituency))
            else:
                trees.append(None)
        except Exception as e:
            logger.debug("stanza failed: %s", e)
            trees.append(None)
    dt = time.time() - t0
    return trees, dt


# ──────────────────────────────────────────────────────────────────────
# Benepar
# ──────────────────────────────────────────────────────────────────────
def _patch_t5_tokenizer() -> None:
    """Compatibility shim: recent transformers removed
    T5Tokenizer.build_inputs_with_special_tokens, but benepar 0.2 still
    calls it.  The real method just wraps the ids with no specials — we
    replicate that behavior.
    """
    try:
        from transformers import T5Tokenizer, T5TokenizerFast
        for cls in (T5Tokenizer, T5TokenizerFast):
            if not hasattr(cls, "build_inputs_with_special_tokens"):
                def _bi(self, token_ids_0, token_ids_1=None):
                    if token_ids_1 is None:
                        return list(token_ids_0) + [self.eos_token_id]
                    return (
                        list(token_ids_0) + [self.eos_token_id]
                        + list(token_ids_1) + [self.eos_token_id]
                    )
                cls.build_inputs_with_special_tokens = _bi
    except ImportError:
        pass


def run_benepar(sentences: list[str]) -> tuple[list[str | None], float]:
    _patch_t5_tokenizer()
    import benepar
    import spacy

    logger.info("loading benepar (CPU)...")
    try:
        nlp = spacy.load("en_core_web_sm")
    except OSError:
        from spacy.cli import download
        download("en_core_web_sm")
        nlp = spacy.load("en_core_web_sm")

    if "benepar" not in nlp.pipe_names:
        # Will download benepar_en3 on first use (~90MB).
        try:
            nlp.add_pipe("benepar", config={"model": "benepar_en3"})
        except LookupError:
            benepar.download("benepar_en3")
            nlp.add_pipe("benepar", config={"model": "benepar_en3"})

    trees: list[str | None] = []
    t0 = time.time()
    for s in sentences:
        try:
            doc = nlp(s)
            tree = None
            for sent in doc.sents:
                tree = sent._.parse_string
                break
            trees.append(tree)
        except Exception as e:
            logger.debug("benepar failed: %s", e)
            trees.append(None)
    dt = time.time() - t0
    return trees, dt


# ──────────────────────────────────────────────────────────────────────
# Comparison helpers
# ──────────────────────────────────────────────────────────────────────
def count_constituents(tree_str: str | None) -> tuple[int, dict[str, int]]:
    """Count how many labeled constituents of which types a tree has."""
    if not tree_str:
        return 0, {}
    labels: dict[str, int] = {}
    total = 0
    # Crude: find each "(LABEL " opener
    import re
    for m in re.finditer(r"\((\w+)\s", tree_str):
        lab = m.group(1)
        # Skip POS tags (short labels that aren't phrase types)
        if lab in {"NP", "VP", "PP", "S", "SBAR", "ADJP", "ADVP", "SINV",
                   "SQ", "SBARQ", "WHNP", "WHVP", "WHPP", "PRN", "FRAG",
                   "NML", "QP", "UCP"}:
            labels[lab] = labels.get(lab, 0) + 1
            total += 1
    return total, labels


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sentences-file", type=Path, required=True)
    ap.add_argument("--num-samples", type=int, default=30)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    all_sents = args.sentences_file.read_text().splitlines()
    rng = random.Random(args.seed)
    sents = rng.sample(all_sents, args.num_samples)
    # Sanity-clip to reasonable lengths
    sents = [s for s in sents if 30 <= len(s) <= 250]
    logger.info("comparing parsers on %d sentences", len(sents))

    stanza_trees, t_stanza = run_stanza(sents)
    benepar_trees, t_benepar = run_benepar(sents)

    print()
    print("=" * 78)
    print("PARSER COMPARISON")
    print("=" * 78)
    print(f"N sentences      : {len(sents)}")
    print(f"Stanza   total   : {t_stanza:6.2f}s  ({len(sents)/t_stanza:5.1f} sents/sec CPU)")
    print(f"Benepar  total   : {t_benepar:6.2f}s  ({len(sents)/t_benepar:5.1f} sents/sec CPU)")
    print(f"Speedup          : {t_stanza / t_benepar:5.2f}× (benepar vs stanza)")
    print()

    # Per-parser constituent counts
    stanza_counts = [count_constituents(t) for t in stanza_trees]
    benepar_counts = [count_constituents(t) for t in benepar_trees]
    stanza_mean = sum(c[0] for c in stanza_counts) / max(len(sents), 1)
    benepar_mean = sum(c[0] for c in benepar_counts) / max(len(sents), 1)
    print(f"Stanza  mean constituents / sentence : {stanza_mean:.1f}")
    print(f"Benepar mean constituents / sentence : {benepar_mean:.1f}")

    print()
    print("─" * 78)
    print("SIDE-BY-SIDE (first 8 samples)")
    print("─" * 78)
    for i, (src, stz, ben) in enumerate(
        zip(sents[:8], stanza_trees[:8], benepar_trees[:8]),
    ):
        print()
        print(f"[{i}] src: {src[:120]}")
        print()
        print(f"    STANZA : {(stz or '<none>')[:400]}")
        print()
        print(f"    BENEPAR: {(ben or '<none>')[:400]}")
        print()
        _, stz_labels = count_constituents(stz)
        _, ben_labels = count_constituents(ben)
        print(f"    stanza  labels : {stz_labels}")
        print(f"    benepar labels : {ben_labels}")
        print("─" * 78)


if __name__ == "__main__":
    main()

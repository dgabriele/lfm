#!/usr/bin/env python
"""Parse a chunk of sentences with Stanza constituency → wrapped lines.

Invoked on each vast.ai instance by the fan-out orchestrator.  Reads
one sentence per line, runs Stanza GPU constituency (2 GPU workers),
wraps each constituent with phrase-type tags, and writes one wrapped
constituent per line.

The Stanza backend is selected explicitly here (bypassing the
registry's benepar preference) because benepar's model download is
fragile on fresh vast instances and Stanza GPU is proven good.
"""

import os
# Must be set BEFORE importing anything from lfm.data.parsers so the
# registry's get_backend dispatches to stanza instead of benepar.
os.environ.setdefault("LFM_FORCE_STANZA", "1")

from __future__ import annotations

import argparse
import importlib.util
import logging
import time
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger(__name__)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()

    from lfm.data.constituents import extract_constituents_parallel

    # Reuse wrap_constituent from build_v13_corpus (they're identical rules).
    spec = importlib.util.spec_from_file_location(
        "bv13", Path(__file__).parent / "build_v13_corpus.py",
    )
    bv13 = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(bv13)

    sents = args.input.read_text().splitlines()
    logger.info("chunk %s: %d sentences", args.input.name, len(sents))
    samples = [("eng", s) for s in sents]

    t0 = time.time()
    results = extract_constituents_parallel(samples, min_length=10)
    elapsed = time.time() - t0
    logger.info(
        "parsed in %.1f min → %d raw constituents (%.1f sents/sec)",
        elapsed / 60, len(results), len(sents) / max(elapsed, 1e-9),
    )

    wrapped: list[str] = []
    skipped = 0
    for _lang, text, label, _parent in results:
        w = bv13.wrap_constituent(text, label)
        if w is None:
            skipped += 1
            continue
        wrapped.append(w)
    logger.info("wrapped %d, skipped %d", len(wrapped), skipped)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(wrapped))
    logger.info("wrote %s (%d lines)", args.output, len(wrapped))


if __name__ == "__main__":
    main()

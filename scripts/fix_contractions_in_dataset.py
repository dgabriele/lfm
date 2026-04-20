#!/usr/bin/env python3
"""Fix contractions in the English dep-tree dataset.

Reads the existing sentences.jsonl, expands contractions in the
English text, re-converts to IPA via epitran, re-parses deps via
Stanza, and writes a new fixed JSONL.

Usage:
    poetry run python3 scripts/fix_contractions_in_dataset.py \
        --input data/datasets/english-dep-trees-v16/sentences.jsonl \
        --output data/datasets/english-dep-trees-v16/sentences_fixed.jsonl \
        --batch-size 128 --device gpu
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--device", default="gpu", choices=["gpu", "cpu"])
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()

    from lfm.data.contractions import expand_contractions

    # Pass 1: read English texts, expand contractions, check which changed
    logger.info("Pass 1: expanding contractions...")
    originals: list[str] = []
    expanded: list[str] = []
    changed = 0
    total = 0

    with open(args.input) as f:
        for line in f:
            rec = json.loads(line)
            eng = rec["english"]
            exp = expand_contractions(eng)
            originals.append(eng)
            expanded.append(exp)
            if exp != eng:
                changed += 1
            total += 1
            if total % 2_000_000 == 0:
                logger.info("  scanned %dM...", total // 1_000_000)

    logger.info(
        "  %d/%d sentences have contractions (%.1f%%)",
        changed, total, 100 * changed / max(total, 1),
    )

    # Pass 2: IPA convert all expanded sentences
    logger.info("Pass 2: IPA conversion via epitran...")
    import epitran

    epi = epitran.Epitran("eng-Latn")
    ipa_list: list[str | None] = []
    failed_ipa = 0

    for i, eng in enumerate(expanded):
        try:
            ipa = epi.transliterate(eng)
            if ipa and len(ipa.split()) == len(eng.split()):
                ipa_list.append(ipa)
            else:
                ipa_list.append(None)
                failed_ipa += 1
        except Exception:
            ipa_list.append(None)
            failed_ipa += 1
        if (i + 1) % 1_000_000 == 0:
            logger.info("  IPA %dM / %dM (%d failed)...", (i + 1) // 1_000_000, total // 1_000_000, failed_ipa)

    logger.info("  IPA done: %d failed alignment", failed_ipa)

    # Pass 3: dep parse in batches
    logger.info("Pass 3: dependency parsing via Stanza...")
    import stanza

    try:
        import orjson
        def dumps(obj):
            return orjson.dumps(obj, option=orjson.OPT_APPEND_NEWLINE).decode()
    except ImportError:
        def dumps(obj):
            return json.dumps(obj, ensure_ascii=False) + "\n"

    stanza.download("en", processors="tokenize,pos,lemma,depparse", verbose=False)
    nlp = stanza.Pipeline(
        "en",
        processors="tokenize,pos,lemma,depparse",
        tokenize_pretokenized=True,
        use_gpu=(args.device == "gpu"),
        verbose=False,
    )
    logger.info("  Stanza loaded")

    parsed = 0
    failed_parse = 0
    t0 = time.time()
    batch_size = args.batch_size

    with open(args.output, "w") as fout:
        batch_eng: list[str] = []
        batch_ipa: list[str] = []
        batch_idx: list[int] = []

        def flush_batch():
            nonlocal parsed, failed_parse
            if not batch_eng:
                return
            word_lists = [e.split() for e in batch_eng]
            try:
                doc = nlp(word_lists)
            except Exception:
                failed_parse += len(batch_eng)
                batch_eng.clear()
                batch_ipa.clear()
                batch_idx.clear()
                return

            if len(doc.sentences) != len(batch_eng):
                failed_parse += len(batch_eng)
                batch_eng.clear()
                batch_ipa.clear()
                batch_idx.clear()
                return

            lines = []
            for (eng, ipa, idx), sent in zip(
                zip(batch_eng, batch_ipa, batch_idx), doc.sentences
            ):
                ipa_words = ipa.split()
                words = sent.words
                if len(words) != len(ipa_words):
                    failed_parse += 1
                    continue
                dep_labels = [w.deprel for w in words]
                dep_heads = [w.head for w in words]
                tagged_ipa = " ".join(
                    f"[{lab}] {iw}" for lab, iw in zip(dep_labels, ipa_words)
                )
                lines.append(dumps({
                    "raw_english": originals[idx],
                    "english": eng,
                    "ipa": ipa,
                    "dep_labels": dep_labels,
                    "dep_heads": dep_heads,
                    "tagged_ipa": tagged_ipa,
                }))
                parsed += 1

            if lines:
                fout.write("".join(lines))

            batch_eng.clear()
            batch_ipa.clear()
            batch_idx.clear()

        for i in range(total):
            if ipa_list[i] is None:
                continue
            batch_eng.append(expanded[i])
            batch_ipa.append(ipa_list[i])
            batch_idx.append(i)

            if len(batch_eng) >= batch_size:
                flush_batch()

            if parsed > 0 and parsed % 100_000 < batch_size:
                elapsed = time.time() - t0
                rate = parsed / max(elapsed, 1)
                remaining = (total - i) / max(rate, 1)
                logger.info(
                    "  [%d/%d] %d parsed, %d failed, %.0f/s, ETA %.0fm",
                    i, total, parsed, failed_parse, rate, remaining / 60,
                )

        flush_batch()

    elapsed = time.time() - t0
    logger.info(
        "Done: %d parsed, %d failed (IPA: %d, parse: %d), %.1fm",
        parsed, failed_ipa + failed_parse, failed_ipa, failed_parse, elapsed / 60,
    )


if __name__ == "__main__":
    main()

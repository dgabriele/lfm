#!/usr/bin/env python3
"""Extract phrase constituents from an existing HDF5 dataset.

Post-processes a full-sentence dataset by parsing raw text with the
unified dep→con backend and writing a constituency HDF5 alongside it.
Each constituent references its parent sentence by index, ensuring
perfect alignment with the source dataset.

Usage::

    poetry run python scripts/extract_constituents.py
    poetry run python scripts/extract_constituents.py \
        --dataset data/datasets/leipzig-16lang \
        --output data/datasets/leipzig-16lang-constituents
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import h5py
import numpy as np

sys.stderr = open(sys.stderr.fileno(), "w", buffering=1, closefd=False)
sys.stdout = open(sys.stdout.fileno(), "w", buffering=1, closefd=False)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
logger = logging.getLogger(__name__)


def main(
    dataset_path: str = "data/datasets/leipzig-16lang",
    output_path: str = "data/datasets/leipzig-16lang-constituents",
    min_length: int = 10,
    batch_size: int = 64,
    max_samples: int | None = None,
) -> None:
    """Extract constituents from an existing dataset's raw text.

    Args:
        dataset_path: Path to the source HDF5 dataset directory.
        output_path: Output directory for the constituency dataset.
        min_length: Minimum character length for constituents.
        batch_size: Sentences per parse batch.
        max_samples: Optional cap on total samples to process.
    """
    src = Path(dataset_path)
    dst = Path(output_path)
    dst.mkdir(parents=True, exist_ok=True)

    # Load source dataset
    h5_path = src / "samples.h5"
    logger.info("Loading source dataset: %s", h5_path)

    with h5py.File(h5_path, "r") as f:
        samples = f["samples"]
        raw_texts = [t.decode() for t in samples["raw"][:]]
        languages = [l.decode() for l in samples["language"][:]]
        ipa_texts = [t.decode() for t in samples["ipa"][:]]

    n_total = len(raw_texts)
    if max_samples and max_samples < n_total:
        raw_texts = raw_texts[:max_samples]
        languages = languages[:max_samples]
        ipa_texts = ipa_texts[:max_samples]
        n_total = max_samples

    logger.info("Source: %d samples, %d languages", n_total, len(set(languages)))

    # Group by language for efficient batch parsing
    by_lang: dict[str, list[tuple[int, str]]] = defaultdict(list)
    for idx, (lang, text) in enumerate(zip(languages, raw_texts)):
        by_lang[lang].append((idx, text))

    lang_counts = {lang: len(items) for lang, items in by_lang.items()}
    logger.info("Per-language: %s", dict(sorted(lang_counts.items())))

    # Parse and extract constituents
    from lfm.data.constituents import EXTRACT_LABELS, _extract_from_parse_tree
    from lfm.data.parsers import get_backend, supported_languages

    supported = supported_languages()
    results: list[tuple[int, str, str, str]] = []  # (parent_idx, text, label, language)

    for lang in sorted(by_lang.keys()):
        items = by_lang[lang]
        if lang not in supported:
            logger.info("[%s] %d sentences — no parser, skipping", lang, len(items))
            continue

        logger.info("[%s] Parsing %d sentences...", lang, len(items))
        backend = get_backend(lang, use_gpu=True)
        t0 = time.time()
        lang_constituents = 0

        for batch_start in range(0, len(items), batch_size):
            batch = items[batch_start : batch_start + batch_size]
            indices = [idx for idx, _ in batch]
            texts = [text for _, text in batch]

            trees = backend.parse(texts)

            for si, tree in enumerate(trees):
                if tree is None:
                    continue
                parent_idx = indices[si]
                raw_results: list[tuple[str, str, str, int]] = []
                _extract_from_parse_tree(
                    tree, lang, EXTRACT_LABELS, min_length,
                    raw_results, depth=0, parent_seq=parent_idx,
                )
                for _, con_text, label, pidx in raw_results:
                    results.append((pidx, con_text, label, lang))
                    lang_constituents += 1

            if (batch_start + batch_size) % (batch_size * 10) == 0:
                elapsed = time.time() - t0
                logger.info(
                    "  [%s] %d/%d → %d constituents (%.0fs)",
                    lang, batch_start + batch_size, len(items),
                    lang_constituents, elapsed,
                )

        elapsed = time.time() - t0
        logger.info(
            "[%s] Done: %d constituents from %d sentences (%.0fs)",
            lang, lang_constituents, len(items), elapsed,
        )

    logger.info("Total constituents: %d", len(results))

    # Label distribution
    label_counts = Counter(label for _, _, label, _ in results)
    logger.info("Label distribution: %s", dict(label_counts.most_common()))

    # Write output HDF5
    out_h5 = dst / "constituents.h5"
    logger.info("Writing: %s", out_h5)

    with h5py.File(out_h5, "w") as f:
        grp = f.create_group("constituents")
        n = len(results)

        parent_indices = np.array([r[0] for r in results], dtype=np.int64)
        texts = [r[1] for r in results]
        labels = [r[2] for r in results]
        langs = [r[3] for r in results]

        grp.create_dataset("parent_idx", data=parent_indices)
        grp.create_dataset(
            "text", data=np.array(texts, dtype=h5py.special_dtype(vlen=str)),
        )
        grp.create_dataset(
            "label", data=np.array(labels, dtype=h5py.special_dtype(vlen=str)),
        )
        grp.create_dataset(
            "language", data=np.array(langs, dtype=h5py.special_dtype(vlen=str)),
        )

        # Store reference to source dataset
        f.attrs["source_dataset"] = str(src)
        f.attrs["source_samples"] = n_total
        f.attrs["num_constituents"] = n

    # Write manifest
    import yaml
    manifest = {
        "name": "constituency",
        "description": "Phrase constituents extracted from Leipzig 16-lang dataset",
        "source_dataset": str(src),
        "source_samples": n_total,
        "num_constituents": len(results),
        "label_distribution": dict(label_counts.most_common()),
        "languages": dict(sorted(
            Counter(r[3] for r in results).most_common(),
        )),
        "parser_backend": "depcon (UD dep→con, unified for all languages)",
        "min_constituent_length": min_length,
    }
    with open(dst / "manifest.yaml", "w") as f:
        yaml.dump(manifest, f, default_flow_style=False)

    logger.info("Done. %d constituents → %s", len(results), dst)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Extract phrase constituents from existing HDF5 dataset",
    )
    parser.add_argument(
        "--dataset", default="data/datasets/leipzig-16lang",
        help="Source dataset directory",
    )
    parser.add_argument(
        "--output", default="data/datasets/leipzig-16lang-constituents",
        help="Output directory",
    )
    parser.add_argument("--min-length", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-samples", type=int, default=None)
    args = parser.parse_args()
    main(
        dataset_path=args.dataset,
        output_path=args.output,
        min_length=args.min_length,
        batch_size=args.batch_size,
        max_samples=args.max_samples,
    )

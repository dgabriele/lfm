"""Extract phrase constituents from an existing HDF5 dataset.

Post-processes a full-sentence dataset by parsing raw text with the
unified dep→con backend.  Each constituent references its parent
sentence by index, ensuring perfect alignment with the source dataset.
"""

from __future__ import annotations

import logging
import time
from collections import Counter, defaultdict
from pathlib import Path

import h5py
import numpy as np

logger = logging.getLogger(__name__)


def extract_from_dataset(
    dataset_path: str = "data/datasets/leipzig-16lang",
    output_path: str = "data/datasets/leipzig-16lang-constituents",
    min_length: int = 10,
    batch_size: int = 64,
    max_samples: int | None = None,
) -> Path:
    """Extract constituents from an existing dataset's raw text.

    Args:
        dataset_path: Path to the source HDF5 dataset directory.
        output_path: Output directory for the constituency dataset.
        min_length: Minimum character length for constituents.
        batch_size: Sentences per parse batch.
        max_samples: Optional cap on total samples to process.

    Returns:
        Path to the output directory.
    """
    src = Path(dataset_path)
    dst = Path(output_path)
    dst.mkdir(parents=True, exist_ok=True)

    h5_path = src / "samples.h5"
    logger.info("Loading source dataset: %s", h5_path)

    with h5py.File(h5_path, "r") as f:
        samples = f["samples"]
        raw_texts = [t.decode() for t in samples["raw"][:]]
        languages = [l.decode() for l in samples["language"][:]]

    n_total = len(raw_texts)
    if max_samples and max_samples < n_total:
        raw_texts = raw_texts[:max_samples]
        languages = languages[:max_samples]
        n_total = max_samples

    logger.info("Source: %d samples, %d languages", n_total, len(set(languages)))

    by_lang: dict[str, list[tuple[int, str]]] = defaultdict(list)
    for idx, (lang, text) in enumerate(zip(languages, raw_texts)):
        by_lang[lang].append((idx, text))

    logger.info(
        "Per-language: %s",
        dict(sorted((k, len(v)) for k, v in by_lang.items())),
    )

    from lfm.data.constituents import EXTRACT_LABELS, _extract_from_parse_tree
    from lfm.data.parsers import get_backend, supported_languages

    supported = supported_languages()
    results: list[tuple[int, str, str, str]] = []

    for lang in sorted(by_lang.keys()):
        items = by_lang[lang]
        if lang not in supported:
            logger.info("[%s] %d sentences — no parser, skipping", lang, len(items))
            continue

        logger.info("[%s] Parsing %d sentences...", lang, len(items))
        try:
            backend = get_backend(lang, use_gpu=True)
        except Exception as e:
            logger.warning("[%s] Failed to load parser: %s — skipping", lang, e)
            continue
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
                parent_idx = indices[min(si, len(indices) - 1)]
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

    label_counts = Counter(label for _, _, label, _ in results)
    logger.info("Label distribution: %s", dict(label_counts.most_common()))

    out_h5 = dst / "constituents.h5"
    logger.info("Writing: %s", out_h5)

    with h5py.File(out_h5, "w") as f:
        grp = f.create_group("constituents")

        grp.create_dataset(
            "parent_idx",
            data=np.array([r[0] for r in results], dtype=np.int64),
        )
        grp.create_dataset(
            "text",
            data=np.array([r[1] for r in results], dtype=h5py.special_dtype(vlen=str)),
        )
        grp.create_dataset(
            "label",
            data=np.array([r[2] for r in results], dtype=h5py.special_dtype(vlen=str)),
        )
        grp.create_dataset(
            "language",
            data=np.array([r[3] for r in results], dtype=h5py.special_dtype(vlen=str)),
        )

        f.attrs["source_dataset"] = str(src)
        f.attrs["source_samples"] = n_total
        f.attrs["num_constituents"] = len(results)

    import yaml
    manifest = {
        "name": "constituency",
        "description": "Phrase constituents extracted via unified dep→con",
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
    return dst

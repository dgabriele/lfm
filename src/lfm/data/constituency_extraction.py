"""Extract phrase constituents from an existing HDF5 dataset.

Post-processes a full-sentence dataset by parsing raw text with the
unified dep→con backend.  Each constituent is converted to IPA
inline (with word-alignment fallback for languages where direct
conversion fails).  Results are checkpointed per-language for
resumability.
"""

from __future__ import annotations

import logging
import time
from collections import Counter, defaultdict
from pathlib import Path

import h5py
import numpy as np

logger = logging.getLogger(__name__)


def _write_lang_results(lang_dir: Path, lang: str, results: list) -> None:
    import pickle
    lang_dir.mkdir(parents=True, exist_ok=True)
    with open(lang_dir / f"{lang}.pkl", "wb") as f:
        pickle.dump(results, f, protocol=pickle.HIGHEST_PROTOCOL)


def _load_lang_results(lang_dir: Path, lang: str) -> list | None:
    import pickle
    path = lang_dir / f"{lang}.pkl"
    if not path.exists():
        return None
    with open(path, "rb") as f:
        return pickle.load(f)


def _align_ipa(con_text: str, parent_raw: str, parent_ipa: str) -> str | None:
    """Word-align a constituent to the parent sentence's IPA."""
    if not parent_raw or not parent_ipa:
        return None
    raw_words = parent_raw.split()
    ipa_words = parent_ipa.split()
    if len(raw_words) != len(ipa_words):
        return None
    con_words = con_text.split()
    for start in range(len(raw_words)):
        if raw_words[start : start + len(con_words)] == con_words:
            return " ".join(ipa_words[start : start + len(con_words)])
    return None


def extract_from_dataset(
    dataset_path: str = "data/datasets/leipzig-16lang",
    output_path: str = "data/datasets/leipzig-16lang-constituents",
    min_length: int = 10,
    batch_size: int = 64,
    max_samples: int | None = None,
) -> Path:
    """Extract constituents with IPA conversion from an existing dataset.

    Each constituent is stored as ``(parent_idx, ipa, label, language)``.
    Direct IPA conversion is tried first; on failure, word-alignment
    against the parent sentence's IPA is used as fallback.

    Resumable: per-language results are checkpointed to disk.
    """
    src = Path(dataset_path)
    dst = Path(output_path)
    dst.mkdir(parents=True, exist_ok=True)
    lang_dir = dst / "_lang_checkpoints"

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

    by_lang: dict[str, list[tuple[int, str, str]]] = defaultdict(list)
    for idx, (lang, text, ipa) in enumerate(zip(languages, raw_texts, ipa_texts)):
        by_lang[lang].append((idx, text, ipa))

    logger.info(
        "Per-language: %s",
        dict(sorted((k, len(v)) for k, v in by_lang.items())),
    )

    from lfm.data.constituents import EXTRACT_LABELS, _extract_from_parse_tree
    from lfm.data.loaders.ipa import IPAConverter
    from lfm.data.parsers import get_backend, supported_languages

    supported = supported_languages()
    converter = IPAConverter()
    # Results: (parent_idx, ipa, label, language)
    all_results: list[tuple[int, str, str, str]] = []

    for lang in sorted(by_lang.keys()):
        items = by_lang[lang]
        if lang not in supported:
            logger.info("[%s] %d sentences — no parser, skipping", lang, len(items))
            continue

        cached = _load_lang_results(lang_dir, lang)
        if cached is not None:
            logger.info("[%s] Resuming: %d constituents from checkpoint", lang, len(cached))
            all_results.extend(cached)
            continue

        logger.info("[%s] Parsing %d sentences...", lang, len(items))
        try:
            backend = get_backend(lang, use_gpu=True)
        except Exception as e:
            logger.warning("[%s] Failed to load parser: %s — skipping", lang, e)
            continue

        t0 = time.time()
        lang_results: list[tuple[int, str, str, str]] = []
        direct_ok = 0
        aligned_ok = 0
        failed = 0

        for batch_start in range(0, len(items), batch_size):
            batch = items[batch_start : batch_start + batch_size]
            indices = [idx for idx, _, _ in batch]
            texts = [text for _, text, _ in batch]
            parent_ipas = [ipa for _, _, ipa in batch]

            trees = backend.parse(texts)

            for si, tree in enumerate(trees):
                if tree is None:
                    continue
                parent_idx = indices[min(si, len(indices) - 1)]
                raw_cons: list[tuple[str, str, str, int]] = []
                _extract_from_parse_tree(
                    tree, lang, EXTRACT_LABELS, min_length,
                    raw_cons, depth=0, parent_seq=parent_idx,
                )

                parent_raw = texts[min(si, len(texts) - 1)]
                parent_ipa = parent_ipas[min(si, len(parent_ipas) - 1)]

                for _, con_text, label, pidx in raw_cons:
                    # Skip digits
                    if any(c.isdigit() for c in con_text):
                        failed += 1
                        continue

                    # Try direct IPA conversion
                    ipa = converter.convert_line(lang, con_text)
                    if ipa and len(ipa) >= 5 and not any(c.isdigit() for c in ipa):
                        lang_results.append((pidx, ipa, label, lang))
                        direct_ok += 1
                    else:
                        # Fallback: word-align against parent IPA
                        ipa = _align_ipa(con_text, parent_raw, parent_ipa)
                        if ipa and len(ipa) >= 5 and not any(c.isdigit() for c in ipa):
                            lang_results.append((pidx, ipa, label, lang))
                            aligned_ok += 1
                        else:
                            failed += 1

            if (batch_start + batch_size) % (batch_size * 10) == 0:
                elapsed = time.time() - t0
                logger.info(
                    "  [%s] %d/%d → %d constituents (%.0fs)",
                    lang, batch_start + batch_size, len(items),
                    len(lang_results), elapsed,
                )

        elapsed = time.time() - t0
        logger.info(
            "[%s] Done: %d constituents (%d direct, %d aligned, %d failed) (%.0fs)",
            lang, len(lang_results), direct_ok, aligned_ok, failed, elapsed,
        )

        _write_lang_results(lang_dir, lang, lang_results)
        all_results.extend(lang_results)

    logger.info("Total constituents: %d", len(all_results))

    label_counts = Counter(label for _, _, label, _ in all_results)
    logger.info("Label distribution: %s", dict(label_counts.most_common()))

    out_h5 = dst / "constituents.h5"
    logger.info("Writing: %s", out_h5)

    with h5py.File(out_h5, "w") as f:
        grp = f.create_group("constituents")
        grp.create_dataset(
            "parent_idx",
            data=np.array([r[0] for r in all_results], dtype=np.int64),
        )
        grp.create_dataset(
            "ipa",
            data=np.array([r[1] for r in all_results], dtype=h5py.special_dtype(vlen=str)),
        )
        grp.create_dataset(
            "label",
            data=np.array([r[2] for r in all_results], dtype=h5py.special_dtype(vlen=str)),
        )
        grp.create_dataset(
            "language",
            data=np.array([r[3] for r in all_results], dtype=h5py.special_dtype(vlen=str)),
        )
        f.attrs["source_dataset"] = str(src)
        f.attrs["source_samples"] = n_total
        f.attrs["num_constituents"] = len(all_results)
        f.attrs["format"] = "ipa"

    import yaml
    manifest = {
        "name": "constituency",
        "description": "Phrase constituents as IPA (direct + word-aligned fallback)",
        "source_dataset": str(src),
        "source_samples": n_total,
        "num_constituents": len(all_results),
        "label_distribution": dict(label_counts.most_common()),
        "languages": dict(sorted(
            Counter(r[3] for r in all_results).most_common(),
        )),
        "parser_backend": "depcon (UD dep→con, unified)",
        "min_constituent_length": min_length,
        "format": "ipa",
    }
    with open(dst / "manifest.yaml", "w") as f:
        yaml.dump(manifest, f, default_flow_style=False)

    logger.info("Done. %d constituents → %s", len(all_results), dst)
    return dst

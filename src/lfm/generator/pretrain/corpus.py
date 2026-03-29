"""Corpus loading and sentencepiece training for VAE pretraining."""

from __future__ import annotations

import logging
from pathlib import Path

from lfm._registry import create

from .config import VAEPretrainConfig

logger = logging.getLogger(__name__)


def _load_corpus_labeled(cfg: VAEPretrainConfig) -> list[tuple[str, str]]:
    """Load corpus with language labels preserved through IPA conversion.

    When ``dataset_path`` is set, loads pre-generated IPA from a HDF5
    dataset (no inline sanitization or IPA conversion).  Otherwise falls
    back to the legacy pipeline: corpus loader → sanitize → IPA.

    Returns:
        List of ``(language_code, ipa_text)`` tuples.
    """
    # ── Fast path: pre-generated dataset ──────────────────────────────
    if cfg.dataset_path is not None:
        from lfm.data.dataset.reader import DatasetReader

        reader = DatasetReader(cfg.dataset_path)
        labeled = reader.load_ipa_tuples()
        logger.info(
            "Loaded %d IPA tuples from dataset: %s", len(labeled), cfg.dataset_path
        )
        return labeled

    # ── Legacy path: inline corpus loading + sanitization + IPA ───────
    if cfg.corpus_loader is not None:
        import importlib

        from lfm.data.loaders.base import CorpusLoaderConfig

        loader_registry: dict[str, tuple[str, type[CorpusLoaderConfig]]] = {}

        def _register_loaders() -> None:
            from lfm.data.loaders.leipzig import LeipzigCorpusConfig

            loader_registry["leipzig"] = (
                "lfm.data.loaders.leipzig",
                LeipzigCorpusConfig,
            )

        _register_loaders()

        if cfg.corpus_loader not in loader_registry:
            raise KeyError(
                f"Unknown corpus_loader {cfg.corpus_loader!r}. "
                f"Available: {sorted(loader_registry.keys())}"
            )

        mod_path, config_cls = loader_registry[cfg.corpus_loader]
        importlib.import_module(mod_path)  # triggers @register

        loader_kwargs = {"name": cfg.corpus_loader, **cfg.corpus_loader_config}
        loader_cfg = config_cls(**loader_kwargs)
        loader = create("corpus_loader", cfg.corpus_loader, loader_cfg)
        samples = loader.load()
    else:
        samples = []
        for path_str in cfg.corpus_paths:
            path = Path(path_str)
            files: list[Path] = []
            if path.is_file():
                files.append(path)
            elif path.is_dir():
                files.extend(sorted(path.glob("*.txt")))
            else:
                logger.warning("Corpus path not found: %s", path)
                continue
            for f in files:
                with open(f, encoding="utf-8") as fh:
                    for raw_line in fh:
                        raw_line = raw_line.strip()
                        if raw_line:
                            samples.append(("eng", raw_line))
        logger.info("Loaded %d lines from legacy corpus_paths", len(samples))

    # Sanitize
    samples = _sanitize_samples(samples)
    logger.info("Sanitized to %d lines (dropped noisy/short)", len(samples))

    # Convert to IPA preserving language labels
    from lfm.data.loaders.ipa import convert_corpus_to_ipa_labeled

    labeled = convert_corpus_to_ipa_labeled(samples)
    logger.info("IPA conversion complete: %d labeled lines", len(labeled))
    return labeled


def _load_corpus_lines(cfg: VAEPretrainConfig) -> list[str]:
    """Load text lines (IPA), discarding language labels.

    Backward-compatible wrapper around ``_load_corpus_labeled``.
    """
    return [ipa for _lang, ipa in _load_corpus_labeled(cfg)]


def _sanitize_samples(
    samples: list[tuple[str, str]],
) -> list[tuple[str, str]]:
    """Sanitize corpus samples using multiprocessing.

    Delegates to ``lfm.data.sanitize.sanitize_samples`` with legacy-compatible
    defaults (reject digits, no number spell-out).
    """
    from lfm.data.sanitize import SanitizeConfig, sanitize_samples

    # Legacy behaviour: reject any digits, no terminal punctuation requirement
    legacy_cfg = SanitizeConfig(
        number_policy="reject",
        symbol_policy="strip",
        require_terminal_punctuation=False,
    )
    return sanitize_samples(samples, cfg=legacy_cfg)


def _train_sentencepiece(
    lines: list[str],
    vocab_size: int,
    output_dir: str,
) -> str:
    """Train a sentencepiece model on the given lines.

    Args:
        lines: Text lines for training.
        vocab_size: Target vocabulary size.
        output_dir: Directory to save the model.

    Returns:
        Path to the trained ``.model`` file.
    """
    try:
        import sentencepiece as spm
    except ImportError as e:
        raise ImportError(
            "sentencepiece is required for VAE pretraining. "
            "Install it with: pip install sentencepiece>=0.2"
        ) from e

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    corpus_file = out_dir / "spm_train_corpus.txt"
    with open(corpus_file, "w", encoding="utf-8") as f:
        for line in lines:
            f.write(line + "\n")

    model_prefix = str(out_dir / "spm")
    spm.SentencePieceTrainer.train(
        input=str(corpus_file),
        model_prefix=model_prefix,
        vocab_size=vocab_size,
        model_type="bpe",
        character_coverage=0.9995,
        pad_id=3,
    )

    model_path = model_prefix + ".model"
    logger.info(
        "Trained sentencepiece model: %s (vocab_size=%d)",
        model_path,
        vocab_size,
    )
    return model_path

"""Data preprocessing, caching, and dataloader setup for VAE pretraining."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import torch
from torch import Tensor
from torch.utils.data import DataLoader, random_split

from lfm.data.corpus import MultilingualCorpusDataset

from .checkpoint import _file_hash
from .config import VAEPretrainConfig
from .corpus import _load_corpus_labeled, _train_sentencepiece, syllabify_for_bpe

logger = logging.getLogger(__name__)


def _encode_ipa(sp: Any, text: str) -> list[int]:
    """Encode IPA text. Syllable alignment is baked into the SPM vocabulary
    at training time — no pre-processing needed at encode time."""
    return sp.encode(text, out_type=int)


class PreprocessedData:
    """Container for preprocessed corpus data ready for training."""

    def __init__(
        self,
        *,
        token_ids_list: list[list[int]],
        languages_list: list[str],
        vocab_size: int,
        full_vocab: int,
        bos_id: int,
        eos_id: int,
        spm_path: str,
        sp: Any,  # SentencePieceProcessor
        dataset: MultilingualCorpusDataset,
        train_dataset: Any,
        val_dataset: Any,
        train_loader: DataLoader,
        val_loader: DataLoader,
        interleaved_loader: Any | None,
        corpus_embeddings: Tensor | None,
        use_contrastive: bool,
        use_constituent_context: bool,
        surviving_indices: list[int],
    ) -> None:
        self.token_ids_list = token_ids_list
        self.languages_list = languages_list
        self.vocab_size = vocab_size
        self.full_vocab = full_vocab
        self.bos_id = bos_id
        self.eos_id = eos_id
        self.spm_path = spm_path
        self.sp = sp
        self.dataset = dataset
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.interleaved_loader = interleaved_loader
        self.corpus_embeddings = corpus_embeddings
        self.use_contrastive = use_contrastive
        self.use_constituent_context = use_constituent_context
        self.surviving_indices = surviving_indices


def _load_constituents(
    cfg: VAEPretrainConfig,
) -> tuple[list[tuple[str, str]], list[tuple[str, str, int, str]]]:
    """Load constituency data, converting raw text to IPA if needed.

    Supports two formats:
    - **New format** (``constituents.h5``): raw text constituents with
      parent_idx references to the source sentence dataset.  Raw text
      is converted to IPA using epitran/CMU dictionary.
    - **Legacy format** (``samples.h5`` with constituency columns):
      pre-converted IPA via ``DatasetReader.load_constituent_tuples``.

    Returns:
        (sentences, constituents) where:
        - sentences: ``[(lang, ipa), ...]`` for all parent sentences
        - constituents: ``[(lang, ipa, parent_idx, label), ...]``
    """
    from pathlib import Path

    const_dir = Path(cfg.constituent_dataset_path)
    new_format = (const_dir / "constituents.h5").is_file()

    if new_format:
        return _load_constituents_new_format(cfg, const_dir)

    # Legacy format: DatasetReader with samples.h5
    from lfm.data.dataset.reader import DatasetReader

    const_reader = DatasetReader(cfg.constituent_dataset_path)
    if not const_reader.has_constituents():
        raise ValueError(
            f"Dataset at {cfg.constituent_dataset_path} lacks "
            f"constituency data. Run: lfm dataset extract-constituents"
        )
    return const_reader.load_constituent_tuples(
        max_per_language=cfg.constituent_max_per_language,
        balance_by_length=cfg.constituent_balance_by_length,
        seed=cfg.seed,
    )


def _load_constituents_new_format(
    cfg: VAEPretrainConfig,
    const_dir,
) -> tuple[list[tuple[str, str]], list[tuple[str, str, int, str]]]:
    """Load from the new constituents.h5 format with raw text → IPA."""
    import random
    from collections import defaultdict
    from pathlib import Path

    import h5py

    # Load constituent raw text
    h5_path = const_dir / "constituents.h5"
    logger.info("Loading constituents: %s", h5_path)

    with h5py.File(h5_path, "r") as f:
        grp = f["constituents"]
        parent_indices = grp["parent_idx"][:].tolist()
        texts = [t.decode() if isinstance(t, bytes) else t for t in grp["text"][:]]
        labels = [t.decode() if isinstance(t, bytes) else t for t in grp["label"][:]]
        languages = [t.decode() if isinstance(t, bytes) else t for t in grp["language"][:]]
        source_dataset = f.attrs.get("source_dataset", "")

    logger.info("Loaded %d constituents from %d source sentences",
                len(texts), len(set(parent_indices)))

    # Load parent sentences from the source dataset (for sentence-context z)
    src_dir = Path(source_dataset) if source_dataset else Path(cfg.dataset_path)
    src_h5 = src_dir / "samples.h5"
    logger.info("Loading parent sentences from: %s", src_h5)

    with h5py.File(src_h5, "r") as f:
        src = f["samples"]
        src_ipas = [t.decode() if isinstance(t, bytes) else t for t in src["ipa"][:]]
        src_langs = [t.decode() if isinstance(t, bytes) else t for t in src["language"][:]]

    # Build sentence list: unique parent sentences referenced by constituents
    parent_set = sorted(set(parent_indices))
    parent_to_sent_idx: dict[int, int] = {}
    sentences: list[tuple[str, str]] = []
    for global_idx in parent_set:
        if global_idx < len(src_ipas):
            parent_to_sent_idx[global_idx] = len(sentences)
            sentences.append((src_langs[global_idx], src_ipas[global_idx]))

    # Convert constituent raw text to IPA (cached to avoid repeated 15min conversion)
    import pickle

    ipa_cache_path = const_dir / "_ipa_cache.pkl"
    if ipa_cache_path.exists():
        logger.info("Loading cached IPA conversions from %s", ipa_cache_path)
        with open(ipa_cache_path, "rb") as pf:
            cached = pickle.load(pf)
        constituents = cached["constituents"]
        skipped = cached["skipped"]
    else:
        logger.info("Converting %d constituents to IPA (will cache)...", len(texts))
        from lfm.data.loaders.ipa import IPAConverter

        converter = IPAConverter()
        constituents: list[tuple[str, str, int, str]] = []
        skipped = 0

        for i, (raw_text, label, lang, parent_idx) in enumerate(
            zip(texts, labels, languages, parent_indices),
        ):
            sent_idx = parent_to_sent_idx.get(parent_idx)
            if sent_idx is None:
                skipped += 1
                continue

            try:
                ipa = converter.convert_line(lang, raw_text)
            except Exception:
                ipa = None

            if ipa and len(ipa) >= 5:
                constituents.append((lang, ipa, sent_idx, label))
            else:
                skipped += 1

            if i > 0 and i % 1000000 == 0:
                logger.info("  IPA conversion: %d / %d", i, len(texts))

        # Cache for future runs
        with open(ipa_cache_path, "wb") as pf:
            pickle.dump({"constituents": constituents, "skipped": skipped}, pf,
                        protocol=pickle.HIGHEST_PROTOCOL)
        logger.info("Cached IPA conversions to %s", ipa_cache_path)

    logger.info(
        "Converted: %d constituents, %d skipped, %d sentences",
        len(constituents), skipped, len(sentences),
    )

    # Optional per-language cap and length balancing
    if cfg.constituent_max_per_language is not None or cfg.constituent_balance_by_length:
        rng = random.Random(cfg.seed)
        by_lang: dict[str, list[tuple[str, str, int, str]]] = defaultdict(list)
        for item in constituents:
            by_lang[item[0]].append(item)

        balanced: list[tuple[str, str, int, str]] = []
        cap = cfg.constituent_max_per_language
        for lang in sorted(by_lang.keys()):
            items = by_lang[lang]
            if cap is not None and len(items) > cap:
                items = rng.sample(items, cap)
            balanced.extend(items)

        rng.shuffle(balanced)
        constituents = balanced

    return sentences, constituents


def load_and_preprocess(cfg: VAEPretrainConfig) -> PreprocessedData:
    """Load corpus, tokenize, build datasets and dataloaders.

    Handles preprocessing cache (v3 format), sentencepiece training,
    contrastive embedding alignment, and constituent context setup.

    Returns:
        A ``PreprocessedData`` container with everything needed to start
        the training loop.
    """
    output_dir = str(Path(cfg.output_path).parent)
    cache_path = Path(output_dir) / "preprocessed_cache.pt"
    spm_path_cached = Path(output_dir) / "spm.model"

    # Cache v3: includes 'languages' and 'spm_hash' for consistency.
    _cache_valid = False
    _surviving_indices: list[int] = []
    if cache_path.exists() and spm_path_cached.exists():
        logger.info("Loading preprocessed cache from %s", cache_path)
        cache = torch.load(cache_path, weights_only=False)
        spm_hash = _file_hash(spm_path_cached)
        if "languages" not in cache:
            logger.info(
                "Cache missing 'languages' (v1 format) — regenerating..."
            )
        elif cache.get("spm_hash") and cache["spm_hash"] != spm_hash:
            logger.info(
                "Cache SPM hash mismatch (cache=%s, file=%s) — "
                "regenerating...",
                cache["spm_hash"],
                spm_hash,
            )
        else:
            token_ids_list = cache["token_ids_list"]
            languages_list: list[str] = cache["languages"]
            vocab_size = cache["vocab_size"]
            spm_path = str(spm_path_cached)
            _cache_valid = True

    if not _cache_valid:
        # 1. Load corpus with language labels preserved
        labeled = _load_corpus_labeled(cfg)
        if not labeled:
            raise RuntimeError("No text lines loaded from corpus")

        # Extract IPA lines for sentencepiece training
        lines = [ipa for _lang, ipa in labeled]

        # 2. Sentencepiece
        if cfg.spm_model_path is not None:
            spm_path = cfg.spm_model_path
        else:
            spm_path = _train_sentencepiece(
                lines, cfg.spm_vocab_size, output_dir,
                syllable_aligned=getattr(cfg, "syllable_aligned_bpe", False),
            )

        try:
            import sentencepiece as spm_lib
        except ImportError as e:
            raise ImportError(
                "sentencepiece is required for VAE pretraining."
            ) from e

        sp = spm_lib.SentencePieceProcessor(model_file=spm_path)
        vocab_size = sp.vocab_size()

        # 3. Tokenize, keeping language labels aligned.
        # Track which input indices survived for embedding alignment.
        _spm_specials = {0, 1, 2, 3}
        token_ids_list = []
        languages_list = []
        _surviving_indices: list[int] = []
        for idx, (lang, ipa) in enumerate(labeled):
            ids = _encode_ipa(sp, ipa)
            ids = [x for x in ids if x not in _spm_specials]
            if len(ids) >= 5:
                token_ids_list.append(ids)
                languages_list.append(lang)
                _surviving_indices.append(idx)

        logger.info(
            "Tokenized %d sequences (vocab_size=%d)",
            len(token_ids_list),
            vocab_size,
        )

        # Save cache (v3 format with languages + SPM hash)
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "token_ids_list": token_ids_list,
                "languages": languages_list,
                "vocab_size": vocab_size,
                "spm_hash": _file_hash(spm_path),
            },
            cache_path,
        )
        logger.info("Saved preprocessed cache (v3) to %s", cache_path)

    full_vocab = vocab_size + 2
    bos_id = vocab_size
    eos_id = vocab_size + 1

    try:
        import sentencepiece as spm_lib
    except ImportError as e:
        raise ImportError(
            "sentencepiece is required for VAE pretraining."
        ) from e

    sp = spm_lib.SentencePieceProcessor(model_file=spm_path)

    # 4. Build dataset and split.
    # Identify word-boundary token IDs (sentencepiece ▁ prefix) so
    # truncated sequences are cut at word boundaries, not mid-word.
    word_boundary_ids = {
        i for i in range(vocab_size)
        if sp.id_to_piece(i).startswith("▁")
    }
    dataset = MultilingualCorpusDataset(
        token_ids_list, cfg.max_seq_len, eos_id,
        word_boundary_ids=word_boundary_ids,
    )
    val_size = max(1, int(len(dataset) * cfg.val_fraction))
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = random_split(
        dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(cfg.seed),
    )

    # Load contrastive embeddings if configured
    import numpy as np

    _use_contrastive = cfg.contrastive_weight > 0 and cfg.embeddings_path
    corpus_embeddings: Tensor | None = None
    if _use_contrastive:
        emb_path = Path(cfg.embeddings_path)
        if not emb_path.exists():
            raise FileNotFoundError(
                f"Contrastive embeddings not found: {emb_path}. "
                f"Run: python scripts/precompute_corpus_embeddings.py"
            )
        emb_np = np.load(emb_path)
        # Align embeddings with the tokenized dataset.  Tokenization
        # may drop samples (sequence < 5 tokens), creating a mismatch.
        if len(emb_np) != len(dataset):
            if _surviving_indices and len(_surviving_indices) == len(dataset):
                # Fresh tokenization: we know exactly which indices survived
                logger.info(
                    "Filtering embeddings: %d → %d (by surviving indices)",
                    len(emb_np), len(dataset),
                )
                emb_np = emb_np[_surviving_indices]
            elif len(emb_np) > len(dataset):
                # Cache hit: indices unknown, truncate to dataset length.
                # This works when the dropped samples are at the end or
                # are a small fraction (<0.1%) of the total.
                logger.info(
                    "Truncating embeddings: %d → %d (cache-hit alignment)",
                    len(emb_np), len(dataset),
                )
                emb_np = emb_np[:len(dataset)]
        if len(emb_np) != len(dataset):
            raise ValueError(
                f"Embeddings length {len(emb_np)} != dataset length {len(dataset)}. "
                f"Regenerate embeddings."
            )
        corpus_embeddings = torch.from_numpy(emb_np).float().pin_memory()
        embed_dim = corpus_embeddings.shape[1]
        logger.info(
            "Loaded contrastive embeddings: %s (%.0f MB, pinned)",
            corpus_embeddings.shape, corpus_embeddings.nbytes / 1e6,
        )

    # Build train/val DataLoaders
    _use_constituent_context = cfg.constituent_context

    if _use_contrastive:
        from lfm.data.corpus import IndexedDatasetWrapper

        indexed_train = IndexedDatasetWrapper(train_dataset)
        train_loader = DataLoader(
            indexed_train,
            batch_size=cfg.batch_size,
            shuffle=True,
            drop_last=True,  # InfoNCE needs consistent batch sizes
        )
    else:
        train_loader = DataLoader(
            train_dataset,
            batch_size=cfg.batch_size,
            shuffle=True,
            drop_last=True,
        )

    interleaved_loader = None
    if _use_constituent_context:
        from lfm.data.corpus import ConstituentDataset, InterleavedLoader

        sentences, constituents = _load_constituents(cfg)

        # Tokenize sentences and constituents with the same SPM
        sent_token_ids = [_encode_ipa(sp, ipa) for _, ipa in sentences]
        const_token_ids = [_encode_ipa(sp, ipa) for _, ipa, _, _ in constituents]
        const_parent_indices = [parent_idx for _, _, parent_idx, _ in constituents]

        constituent_dataset = ConstituentDataset(
            sentence_token_ids=sent_token_ids,
            constituent_token_ids=const_token_ids,
            parent_indices=const_parent_indices,
            max_seq_len=cfg.max_seq_len,
            eos_id=eos_id,
        )
        constituent_dl = DataLoader(
            constituent_dataset,
            batch_size=cfg.batch_size,
            shuffle=True,
            drop_last=True,
        )
        interleaved_loader = InterleavedLoader(
            sentence_loader=train_loader,
            constituent_loader=constituent_dl,
            mix_ratio=cfg.constituent_mix_ratio,
            seed=cfg.seed,
        )
        logger.info(
            "Constituent context: %d sentences, %d constituents, "
            "mix_ratio=%.1f, %d batches/epoch",
            len(sentences), len(constituents),
            cfg.constituent_mix_ratio, len(interleaved_loader),
        )

    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.batch_size,
        shuffle=False,
        drop_last=False,
    )

    return PreprocessedData(
        token_ids_list=token_ids_list,
        languages_list=languages_list,
        vocab_size=vocab_size,
        full_vocab=full_vocab,
        bos_id=bos_id,
        eos_id=eos_id,
        spm_path=spm_path,
        sp=sp,
        dataset=dataset,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        train_loader=train_loader,
        val_loader=val_loader,
        interleaved_loader=interleaved_loader,
        corpus_embeddings=corpus_embeddings,
        use_contrastive=bool(_use_contrastive),
        use_constituent_context=_use_constituent_context,
        surviving_indices=_surviving_indices,
    )

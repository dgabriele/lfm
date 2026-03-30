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


def _encode_ipa(sp: Any, text: str, syllable_aligned: bool = False) -> list[int]:
    """Encode IPA text with optional syllable pre-processing."""
    if syllable_aligned:
        text = syllabify_for_bpe(text)
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
            _syl_aligned = getattr(cfg, "syllable_aligned_bpe", False)
            ids = _encode_ipa(sp, ipa, syllable_aligned=_syl_aligned)
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
        from lfm.data.dataset.reader import DatasetReader

        const_reader = DatasetReader(cfg.constituent_dataset_path)
        if not const_reader.has_constituents():
            raise ValueError(
                f"Dataset at {cfg.constituent_dataset_path} lacks "
                f"parent_seq fields. Regenerate with --extract-constituents."
            )
        sentences, constituents = const_reader.load_constituent_tuples(
            max_per_language=cfg.constituent_max_per_language,
            balance_by_length=cfg.constituent_balance_by_length,
            seed=cfg.seed,
        )

        # Tokenize sentences and constituents with the same SPM
        _syl = getattr(cfg, "syllable_aligned_bpe", False)
        sent_token_ids = [_encode_ipa(sp, ipa, syllable_aligned=_syl) for _, ipa in sentences]
        const_token_ids = [_encode_ipa(sp, ipa, syllable_aligned=_syl) for _, ipa, _, _ in constituents]
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

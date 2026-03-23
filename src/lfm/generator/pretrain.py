"""VAE decoder pretraining on multilingual text data.

Pre-trains a VAE encoder-decoder on typologically diverse multilingual text
so the decoder learns the joint distribution of phonotactic, morphological,
and compositional structure across human languages.  After pretraining, only
the decoder weights are saved — the encoder is discarded because during agent
training a separate input projection maps agent embeddings to the same latent
space.

Corpus loading is modular via the registry system — the default ``"leipzig"``
loader handles the Leipzig Corpora Collection format, and additional loaders
(OPUS, UD, etc.) can be registered for future use.

Usage::

    from lfm.generator.pretrain import pretrain_vae_decoder, VAEPretrainConfig

    config = VAEPretrainConfig(
        corpus_loader="leipzig",
        corpus_loader_config={"data_dir": "data/leipzig"},
    )
    metrics = pretrain_vae_decoder(config)
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F  # noqa: N812
from torch import Tensor, nn
from torch.utils.data import DataLoader, random_split

from lfm._registry import create
from lfm.config.base import LFMBaseConfig
from lfm.data.corpus import MultilingualCorpusDataset

logger = logging.getLogger(__name__)

# IPA vowels — used to strip trailing orphan consonants from decoded text
_IPA_VOWELS = set(
    "iyɨʉɯuɪʏʊeøɘɵɤoəɛœɜɞʌɔæɐaɶɑɒ"
    "ãẽĩõũɛ̃ɔ̃"  # nasalized vowels
)


class VAEPretrainConfig(LFMBaseConfig):
    """Configuration for VAE decoder pretraining.

    These defaults are conservative for 6GB single-GPU.  For larger GPUs,
    consider ``decoder_hidden_dim=512``, ``decoder_num_layers=4``,
    ``decoder_num_heads=8``, ``latent_dim=256`` for richer linguistic priors.

    Attributes:
        corpus_loader: Registry name of the corpus loader to use.
        corpus_loader_config: Keyword arguments passed to the corpus loader
            config constructor (e.g. ``{"data_dir": "data/leipzig"}``).
        corpus_paths: Legacy fallback — paths to plain text files/directories.
            Used only when ``corpus_loader`` is ``None``.
        spm_model_path: Path to a trained sentencepiece ``.model`` file.
            If ``None``, trains a new model from the corpus data.
        spm_vocab_size: Vocabulary size for sentencepiece training.
        latent_dim: Must match target ``GeneratorConfig.latent_dim``.
        encoder_num_layers: Number of transformer encoder layers.
        decoder_hidden_dim: Must match target
            ``GeneratorConfig.decoder_hidden_dim``.
        decoder_num_layers: Must match target
            ``GeneratorConfig.decoder_num_layers``.
        decoder_num_heads: Must match target
            ``GeneratorConfig.decoder_num_heads``.
        decoder_dropout: Dropout rate during pretraining.
        max_seq_len: Maximum token sequence length (truncated).
        kl_weight: Final KL weight after warmup.
        kl_free_bits: Per-dimension free bits floor.  Set to ``2.0`` to
            prevent posterior collapse with the conservative decoder.
        kl_warmup_steps: Steps to linearly anneal KL weight from 0.
        batch_size: Pretraining batch size per device.
        gradient_accumulation_steps: Accumulate gradients over this many
            batches before stepping.  Effective batch size is
            ``batch_size * gradient_accumulation_steps``.
        use_amp: Enable mixed precision training (``torch.amp``).
            Halves activation memory and speeds up training.
        lr: Learning rate.
        num_epochs: Number of pretraining epochs.
        val_fraction: Validation split fraction.
        seed: Random seed for reproducibility.
        device: Torch device string.
        output_path: Where to save the pretrained decoder checkpoint.
        repetition_penalty: Multiplicative penalty on logits for tokens
            that appeared in the recent ``repetition_window`` positions.
            ``1.0`` disables.  Applied during both teacher-forced training
            and free-run decoding to suppress degenerate loops.
        repetition_window: Number of recent positions to check for
            repeated tokens.
        topo_weight: Weight for topological regularization loss.
            Penalizes distance mismatches between latent pairs and their
            decoded output pairs, enforcing Lipschitz continuity.
        topo_sample_pairs: Number of random pairs per batch for topo loss.
        use_adversarial: Enable structural adversarial discriminator.
        adv_weight: Weight of adversarial loss in total generator loss.
        adv_lr: Learning rate for discriminator (separate from VAE).
        adv_disc_hidden: Discriminator CNN channel width.
        adv_disc_embed_dim: Discriminator embedding dimensionality.
        adv_warmup_steps: Train discriminator alone for this many steps
            before adding adversarial signal to generator.
        adv_free_run_len: Max length for free-run decoding during
            adversarial training (shorter than ``max_seq_len`` for speed).
        adv_spectral_norm: Apply spectral normalization to discriminator.
    """

    corpus_loader: str | None = "leipzig"
    corpus_loader_config: dict[str, Any] = {}
    corpus_paths: list[str] = []
    spm_model_path: str | None = None
    spm_vocab_size: int = 8000
    latent_dim: int = 256
    encoder_num_layers: int = 2
    decoder_hidden_dim: int = 512
    decoder_num_layers: int = 4
    decoder_num_heads: int = 8
    decoder_dropout: float = 0.1

    # Linguistic attention structure
    attention_head_windows: tuple[int, ...] = (3, 3, 7, 7, 15, 15, 0, 0)
    attention_global_every: int = 7
    use_rope: bool = True
    share_decoder_layers: bool = True
    max_seq_len: int = 64
    repetition_penalty: float = 1.2
    repetition_window: int = 8
    kl_weight: float = 0.5
    kl_free_bits: float = 0.5
    kl_warmup_steps: int = 10000
    batch_size: int = 32
    gradient_accumulation_steps: int = 2
    use_amp: bool = True
    lr: float = 1e-3
    num_epochs: int = 20
    val_fraction: float = 0.1
    seed: int = 42
    device: str = "cuda"
    output_path: str = "data/vae_decoder.pt"

    # Topological regularization (smooth latent → smooth output)
    topo_weight: float = 0.0
    topo_sample_pairs: int = 32

    # Adversarial structural discriminator
    use_adversarial: bool = True
    adv_weight: float = 0.1
    adv_lr: float = 1e-4
    adv_disc_hidden: int = 256
    adv_disc_embed_dim: int = 128
    adv_warmup_steps: int = 1000
    adv_free_run_len: int = 32
    adv_spectral_norm: bool = True


# ---------------------------------------------------------------------------
# Corpus loading
# ---------------------------------------------------------------------------


def _load_corpus_lines(cfg: VAEPretrainConfig) -> list[str]:
    """Load text lines via the configured corpus loader or legacy paths.

    Uses the registry-based corpus loader system when ``corpus_loader`` is
    set, falling back to plain file reading from ``corpus_paths`` otherwise.

    Returns:
        List of text line strings.
    """
    if cfg.corpus_loader is not None:
        # Ensure concrete loaders are registered and build the typed config.
        # Each loader has its own config class; this mapping parallels the
        # registry.  Extend when adding new loaders (OPUS, UD, etc.).
        import importlib

        from lfm.data.loaders.base import CorpusLoaderConfig

        loader_registry: dict[str, tuple[str, type[CorpusLoaderConfig]]] = {}

        # Lazy imports — only load what's needed
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
        # Legacy fallback: plain text files as "unknown" language
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

    # Sanitize: drop lines with digits, URLs, emails, formatting artifacts
    samples = _sanitize_samples(samples)
    logger.info("Sanitized to %d lines (dropped noisy/short)", len(samples))

    # Convert to IPA (uniform phonetic representation across languages)
    from lfm.data.loaders.ipa import convert_corpus_to_ipa

    lines = convert_corpus_to_ipa(samples, drop_unconvertible=True)
    logger.info("IPA conversion complete: %d lines", len(lines))
    return lines


def _sanitize_one(sample: tuple[str, str]) -> tuple[str, str] | None:
    """Sanitize a single ``(lang, text)`` sample.  Returns ``None`` to drop."""
    import re

    from cleantext import clean

    lang, line = sample

    if re.search(r"\d", line):
        return None

    line = clean(
        line,
        fix_unicode=True,
        to_ascii=False,
        lower=False,
        no_urls=True,
        no_emails=True,
        no_phone_numbers=True,
        no_numbers=False,
        no_digits=False,
        no_currency_symbols=True,
        no_punct=False,
    )

    line = line.replace("<NUMBER>", "").replace("<EMAIL>", "")
    line = line.replace("<URL>", "").replace("<PHONE>", "")
    line = " ".join(line.split())

    if not line or len(line) < 20:
        return None
    alpha_ratio = sum(c.isalpha() for c in line) / max(len(line), 1)
    if alpha_ratio < 0.7:
        return None
    return (lang, line)


def _sanitize_samples(
    samples: list[tuple[str, str]],
) -> list[tuple[str, str]]:
    """Sanitize corpus samples using multiprocessing.

    Drops lines with digits, strips URLs/emails/formatting via
    ``clean-text``, filters by alphabetic ratio.  Uses all available
    CPU cores for parallel processing.
    """
    import multiprocessing as mp
    import os

    num_workers = max(1, int(os.cpu_count() * 0.9))
    logger.info(
        "Sanitizing %d samples with %d workers...", len(samples), num_workers
    )

    with mp.Pool(num_workers) as pool:
        results = pool.map(_sanitize_one, samples, chunksize=1000)

    return [r for r in results if r is not None]


# ---------------------------------------------------------------------------
# Sentencepiece training
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# VAE forward pass (shared between train and val)
# ---------------------------------------------------------------------------


def _vae_forward(
    batch_tokens: Tensor,
    batch_lengths: Tensor,
    *,
    enc_token_embedding: nn.Embedding,
    enc_pos_embedding: nn.Embedding,
    encoder: nn.Module,
    enc_to_latent: nn.Linear,
    latent_norm: nn.LayerNorm,
    latent_to_decoder: nn.Linear,
    dec_token_embedding: nn.Embedding,
    dec_pos_embedding: nn.Module,
    decoder: nn.Module,
    output_head: nn.Linear,
    bos_id: int,
    full_vocab: int,
    kl_free_bits: float,
    compute_kl: bool = True,
    repetition_penalty: float = 1.0,
    repetition_window: int = 8,
    _rope_freqs: Tensor | None = None,
    _cached_mask: Tensor | None = None,
    _cfg: object | None = None,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
    """Run one VAE forward pass (teacher-forced).

    Supports both standard ``nn.TransformerDecoder`` and the custom
    ``LinguisticDecoder`` with RoPE and multi-scale attention.

    Returns:
        Tuple of ``(ce_loss, kl_loss, kl_per_dim, z, dec_hidden)``.
    """
    from lfm.generator.layers import LinguisticDecoder, multiscale_causal_mask

    device = batch_tokens.device
    b, seq_len = batch_tokens.shape

    # Encode
    src_mask = (
        torch.arange(seq_len, device=device).unsqueeze(0)
        < batch_lengths.unsqueeze(1)
    )
    pos_ids = torch.arange(seq_len, device=device).unsqueeze(0)
    enc_input = enc_token_embedding(batch_tokens) + enc_pos_embedding(pos_ids)
    enc_out = encoder(enc_input, src_key_padding_mask=~src_mask)

    # Pool (masked mean)
    enc_masked = enc_out * src_mask.unsqueeze(-1).float()
    denom = batch_lengths.unsqueeze(-1).float().clamp(min=1)
    pooled = enc_masked.sum(dim=1) / denom

    # Latent
    h = enc_to_latent(pooled)
    mu, logvar = h.chunk(2, dim=-1)
    std = (0.5 * logvar).exp()
    z = mu + std * torch.randn_like(std)

    # Normalize z before decoding — makes decoder invariant to encoder
    # distribution, so the same decoder works with agent projections.
    z = latent_norm(z)

    # Decode (teacher-forced)
    memory = latent_to_decoder(z).unsqueeze(1)
    bos_col = torch.full((b, 1), bos_id, dtype=torch.long, device=device)
    dec_input_ids = torch.cat([bos_col, batch_tokens[:, :-1]], dim=1)

    if isinstance(decoder, LinguisticDecoder):
        # Linguistic decoder: token embedding only (RoPE handles positions)
        dec_input = dec_token_embedding(dec_input_ids)
        if not isinstance(dec_pos_embedding, nn.Identity):
            dec_input = dec_input + dec_pos_embedding(pos_ids)
        # Multi-scale mask (precomputed, sliced to seq_len)
        if _cached_mask is not None:
            tgt_mask = _cached_mask[:, :seq_len, :seq_len]
        else:
            tgt_mask = multiscale_causal_mask(
                seq_len,
                num_heads=_cfg.decoder_num_heads,
                head_windows=_cfg.attention_head_windows,
                global_every=_cfg.attention_global_every,
                device=device,
            )
        dec_out = decoder(
            dec_input, memory, tgt_mask=tgt_mask, rope_freqs=_rope_freqs
        )
    else:
        # Standard decoder (fallback)
        dec_input = dec_token_embedding(dec_input_ids) + dec_pos_embedding(pos_ids)
        tgt_mask = nn.Transformer.generate_square_subsequent_mask(
            seq_len, device=device
        )
        dec_out = decoder(tgt=dec_input, memory=memory, tgt_mask=tgt_mask)
    logits = output_head(dec_out)

    # Reconstruction loss (masked CE)
    ce = F.cross_entropy(
        logits.reshape(-1, full_vocab),
        batch_tokens.reshape(-1),
        reduction="none",
    ).reshape(b, seq_len)
    ce_loss = (ce * src_mask.float()).sum() / src_mask.float().sum().clamp(min=1)

    # KL loss (skipped when kl_weight=0)
    if compute_kl:
        kl_per_dim = 0.5 * (mu.pow(2) + logvar.exp() - 1 - logvar)
        if kl_free_bits > 0:
            kl_per_dim = torch.clamp(kl_per_dim, min=kl_free_bits)
        kl_loss = kl_per_dim.sum(dim=-1).mean()
    else:
        kl_per_dim = torch.zeros(b, mu.size(-1), device=device)
        kl_loss = torch.tensor(0.0, device=device)

    return ce_loss, kl_loss, kl_per_dim, z, dec_out


# ---------------------------------------------------------------------------
# Free-run decode (for adversarial training)
# ---------------------------------------------------------------------------


def _free_run_decode(
    z: Tensor,
    max_len: int,
    *,
    latent_norm: nn.LayerNorm,
    latent_to_decoder: nn.Linear,
    dec_token_embedding: nn.Embedding,
    dec_pos_embedding: nn.Embedding,
    decoder: nn.TransformerDecoder,
    output_head: nn.Linear,
    bos_id: int,
    temperature: float = 1.0,
) -> tuple[Tensor, Tensor, Tensor]:
    """Autoregressive decode from latent z with Gumbel-Softmax.

    Uses Gumbel-Softmax (hard=True) at each step for differentiable
    discrete generation.  Returns both hard token IDs and soft probability
    distributions so the discriminator can receive gradients.

    Args:
        z: Latent codes ``(batch, latent_dim)``.
        max_len: Maximum decode length.
        temperature: Gumbel-Softmax temperature.

    Returns:
        Tuple of ``(token_ids, soft_probs, mask)`` where:

        - ``token_ids``: ``(batch, max_len)`` hard token indices.
        - ``soft_probs``: ``(batch, max_len, vocab)`` differentiable
          Gumbel-Softmax distributions.
        - ``mask``: ``(batch, max_len)`` boolean (all True).
    """
    from lfm.utils.sampling import gumbel_softmax

    batch = z.size(0)
    device = z.device

    z = latent_norm(z)
    memory = latent_to_decoder(z).unsqueeze(1)  # (B, 1, H)
    bos_embed = dec_token_embedding(
        torch.full((batch, 1), bos_id, dtype=torch.long, device=device)
    )
    generated_embeds = bos_embed

    all_probs: list[Tensor] = []

    for _t in range(max_len):
        seq_len = generated_embeds.size(1)
        tgt = generated_embeds
        if not isinstance(dec_pos_embedding, nn.Identity):
            pos = torch.arange(seq_len, device=device).unsqueeze(0)
            tgt = tgt + dec_pos_embedding(pos)
        causal = nn.Transformer.generate_square_subsequent_mask(
            seq_len, device=device
        )
        out = decoder(tgt=tgt, memory=memory, tgt_mask=causal)
        logits = output_head(out[:, -1])  # (B, V)

        # Gumbel-Softmax: hard tokens forward, soft gradients backward
        probs = gumbel_softmax(logits, tau=temperature, hard=True)
        all_probs.append(probs.unsqueeze(1))

        # Next input: differentiable embed via soft probs
        next_embed = probs @ dec_token_embedding.weight  # (B, H)
        generated_embeds = torch.cat(
            [generated_embeds, next_embed.unsqueeze(1)], dim=1
        )

    soft_probs = torch.cat(all_probs, dim=1)  # (B, max_len, V)
    token_ids = soft_probs.argmax(dim=-1)  # (B, max_len)
    mask = torch.ones(batch, max_len, dtype=torch.bool, device=device)
    return token_ids, soft_probs, mask


# ---------------------------------------------------------------------------
# Pretrainer
# ---------------------------------------------------------------------------


class VAEPretrainer:
    """Pretrain a VAE encoder-decoder on multilingual text data.

    Uses modular corpus loaders via the registry system, mixed precision
    training, and gradient accumulation for memory-constrained GPUs.

    Args:
        config: Pretraining configuration.
    """

    def __init__(self, config: VAEPretrainConfig) -> None:
        self.config = config

    def pretrain(self) -> dict[str, float]:
        """Run the full pretraining pipeline.

        Returns:
            Metrics dict with ``train_loss``, ``val_loss``,
            ``num_samples``, and ``active_latent_dims``.
        """
        cfg = self.config
        torch.manual_seed(cfg.seed)
        device = torch.device(cfg.device)

        # -- Preprocessing cache: skip load/sanitize/IPA/tokenize if cached --
        output_dir = str(Path(cfg.output_path).parent)
        cache_path = Path(output_dir) / "preprocessed_cache.pt"
        spm_path_cached = Path(output_dir) / "spm.model"

        if cache_path.exists() and spm_path_cached.exists():
            logger.info("Loading preprocessed cache from %s", cache_path)
            cache = torch.load(cache_path, weights_only=False)
            token_ids_list = cache["token_ids_list"]
            vocab_size = cache["vocab_size"]
            spm_path = str(spm_path_cached)
        else:
            # 1. Load corpus via loader system
            lines = _load_corpus_lines(cfg)
            if not lines:
                raise RuntimeError("No text lines loaded from corpus")

            # 2. Sentencepiece
            if cfg.spm_model_path is not None:
                spm_path = cfg.spm_model_path
            else:
                spm_path = _train_sentencepiece(
                    lines, cfg.spm_vocab_size, output_dir
                )

            try:
                import sentencepiece as spm_lib
            except ImportError as e:
                raise ImportError(
                    "sentencepiece is required for VAE pretraining."
                ) from e

            sp = spm_lib.SentencePieceProcessor(model_file=spm_path)
            vocab_size = sp.vocab_size()

            # 3. Tokenize and strip SPM special tokens (unk, bos, eos, pad)
            _spm_specials = {0, 1, 2, 3}
            token_ids_list = []
            for line in lines:
                ids = sp.encode(line, out_type=int)
                ids = [x for x in ids if x not in _spm_specials]
                if len(ids) >= 5:
                    token_ids_list.append(ids)

            logger.info(
                "Tokenized %d sequences (vocab_size=%d)",
                len(token_ids_list),
                vocab_size,
            )

            # Save cache
            Path(output_dir).mkdir(parents=True, exist_ok=True)
            torch.save(
                {"token_ids_list": token_ids_list, "vocab_size": vocab_size},
                cache_path,
            )
            logger.info("Saved preprocessed cache to %s", cache_path)

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

        # 4. Build dataset and split
        dataset = MultilingualCorpusDataset(
            token_ids_list, cfg.max_seq_len, eos_id
        )
        val_size = max(1, int(len(dataset) * cfg.val_fraction))
        train_size = len(dataset) - val_size
        train_dataset, val_dataset = random_split(
            dataset,
            [train_size, val_size],
            generator=torch.Generator().manual_seed(cfg.seed),
        )

        train_loader = DataLoader(
            train_dataset,
            batch_size=cfg.batch_size,
            shuffle=True,
            drop_last=False,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=cfg.batch_size,
            shuffle=False,
            drop_last=False,
        )

        # 5. Build VAE model components
        hidden = cfg.decoder_hidden_dim
        modules = self._build_model(cfg, hidden, full_vocab, device)
        all_params: list[nn.Parameter] = []
        for m in modules.values():
            if isinstance(m, nn.Module):
                all_params.extend(m.parameters())

        optimizer = torch.optim.AdamW(all_params, lr=cfg.lr)
        scaler = torch.amp.GradScaler(enabled=cfg.use_amp)

        # 5b. Build adversarial discriminator (optional)
        disc: StructuralDiscriminator | None = None
        disc_optimizer: torch.optim.AdamW | None = None
        if cfg.use_adversarial:
            from lfm.generator.discriminator import StructuralDiscriminator

            disc = StructuralDiscriminator(
                vocab_size=full_vocab,
                embed_dim=cfg.adv_disc_embed_dim,
                hidden_dim=cfg.adv_disc_hidden,
                use_spectral_norm=cfg.adv_spectral_norm,
            ).to(device)
            disc_optimizer = torch.optim.AdamW(
                disc.parameters(), lr=cfg.adv_lr
            )

        # 5c. Phonetic distance cache for topo loss
        _topo_dist_cache = None
        if cfg.topo_weight > 0:
            from lfm.data.loaders.phonetic_distance import PhoneticDistanceCache

            _topo_dist_cache = PhoneticDistanceCache()

        # 6. Resume from checkpoint if available
        import math
        import time

        best_val_loss = float("inf")
        best_metrics: dict[str, float] = {}
        global_step = 0
        start_epoch = 0

        resume_path = Path(output_dir) / "vae_resume.pt"
        if resume_path.exists():
            logger.info("Resuming from %s", resume_path)
            ckpt = torch.load(resume_path, map_location=device, weights_only=False)
            for k, m in modules.items():
                if isinstance(m, nn.Module) and k in ckpt["modules"]:
                    m.load_state_dict(ckpt["modules"][k])
            optimizer.load_state_dict(ckpt["optimizer"])
            scaler.load_state_dict(ckpt["scaler"])
            start_epoch = ckpt["epoch"]
            global_step = ckpt["global_step"]
            best_val_loss = ckpt["best_val_loss"]
            logger.info(
                "Resumed at epoch %d, step %d, best_val=%.4f",
                start_epoch, global_step, best_val_loss,
            )

        # 7. Training loop
        accum = cfg.gradient_accumulation_steps
        log_every = 50  # log every N batches
        num_batches = len(train_loader)

        # Parameter count for logging
        total_params = sum(
            p.numel()
            for m in modules.values()
            if isinstance(m, nn.Module)
            for p in m.parameters()
        )
        logger.info(
            "Model: %d params (%.1fM), %d train batches/epoch",
            total_params,
            total_params / 1e6,
            num_batches,
        )

        for epoch in range(start_epoch, cfg.num_epochs):
            # -- Train --
            for m in modules.values():
                if isinstance(m, nn.Module):
                    m.train()

            train_ce_sum = 0.0
            train_kl_sum = 0.0
            train_topo_sum = 0.0
            train_count = 0
            last_grad_norm = 0.0
            optimizer.zero_grad()
            epoch_start = time.time()
            batch_start = time.time()

            for i, (batch_tokens, batch_lengths) in enumerate(train_loader):
                batch_tokens = batch_tokens.to(device)
                batch_lengths = torch.as_tensor(batch_lengths, device=device)
                b = batch_tokens.size(0)

                with torch.amp.autocast(
                    device_type=device.type, enabled=cfg.use_amp
                ):
                    _do_kl = cfg.kl_weight > 0
                    ce_loss, kl_loss, kl_per_dim_train, z_batch, dec_hidden = (
                        _vae_forward(
                            batch_tokens,
                            batch_lengths,
                            bos_id=bos_id,
                            full_vocab=full_vocab,
                            kl_free_bits=cfg.kl_free_bits,
                            compute_kl=_do_kl,
                            repetition_penalty=cfg.repetition_penalty,
                            repetition_window=cfg.repetition_window,
                            **modules,
                        )
                    )

                    # KL: cyclical annealing (ramp 0→1 over warmup, then hold,
                    # repeat each epoch).  Each cycle lets the encoder
                    # rediscover useful structure before KL compresses.
                    if _do_kl:
                        cycle_pos = global_step % max(cfg.kl_warmup_steps, 1)
                        kl_scale = (
                            min(cycle_pos / max(cfg.kl_warmup_steps, 1), 1.0)
                            * cfg.kl_weight
                        )
                    else:
                        kl_scale = 0.0

                    # Topological regularization: nearby z → similar output.
                    # Both distances are differentiable (cosine on decoder
                    # hidden states).  Phonetic Hamming is logged as a
                    # diagnostic but not used in the loss — its discrete,
                    # noisy gradients destabilize training.
                    topo_loss = torch.tensor(0.0, device=device)
                    if cfg.topo_weight > 0 and b >= 4:
                        n_pairs = min(cfg.topo_sample_pairs, b // 2)
                        idx_a = torch.randperm(b, device=device)[:n_pairs]
                        idx_b = torch.randperm(b, device=device)[:n_pairs]

                        # Both sides differentiable
                        d_z = (
                            (z_batch[idx_a] - z_batch[idx_b])
                            .pow(2)
                            .sum(-1)
                            .sqrt()
                        )
                        h_a = dec_hidden[idx_a].mean(dim=1)
                        h_b = dec_hidden[idx_b].mean(dim=1)
                        d_out = 1.0 - F.cosine_similarity(h_a, h_b, dim=-1)

                        # Normalize to [0,1]
                        d_z_norm = d_z / d_z.max().clamp(min=1e-6)
                        d_out_norm = d_out / d_out.max().clamp(min=1e-6)
                        topo_loss = (d_z_norm - d_out_norm).pow(2).mean()

                    loss = (
                        ce_loss + kl_scale * kl_loss + cfg.topo_weight * topo_loss
                    ) / accum

                scaler.scale(loss).backward()

                # -- Adversarial step --
                adv_loss_val = 0.0
                d_real_val = 0.0
                d_fake_val = 0.0

                if (
                    disc is not None
                    and disc_optimizer is not None
                    and global_step >= cfg.adv_warmup_steps
                    and (i + 1) % 5 == 0  # run every 5th batch (speed)
                ):
                    dec_mods = {
                        k: modules[k]
                        for k in [
                            "latent_norm",
                            "latent_to_decoder",
                            "dec_token_embedding",
                            "dec_pos_embedding",
                            "decoder",
                            "output_head",
                        ]
                    }

                    # Use a small sub-batch for adversarial step to avoid OOM
                    # (Gumbel-Softmax AR loop keeps full computation graph)
                    adv_batch = min(b, 32)

                    # Free-run decode (no grad — diagnostic only)
                    with torch.no_grad():
                        gen_tokens, _, gen_mask = _free_run_decode(
                            torch.randn(adv_batch, cfg.latent_dim, device=device),
                            cfg.adv_free_run_len,
                            bos_id=bos_id,
                            **dec_mods,
                        )

                    # Real sub-batch to match adversarial batch size
                    real_tokens = batch_tokens[:adv_batch]
                    real_lengths = batch_lengths[:adv_batch]
                    real_mask = (
                        torch.arange(
                            real_tokens.size(1), device=device
                        ).unsqueeze(0)
                        < real_lengths.unsqueeze(1)
                    )

                    with torch.amp.autocast(
                        device_type=device.type, enabled=cfg.use_amp
                    ):
                        # Discriminator update only
                        real_logits = disc(real_tokens, real_mask)
                        fake_logits = disc(gen_tokens, gen_mask)

                        # Label smoothing: real=0.9, fake=0.1
                        disc_loss = F.binary_cross_entropy_with_logits(
                            real_logits,
                            torch.full_like(real_logits, 0.9),
                        ) + F.binary_cross_entropy_with_logits(
                            fake_logits,
                            torch.full_like(fake_logits, 0.1),
                        )

                    disc_optimizer.zero_grad()
                    scaler.scale(disc_loss).backward()
                    scaler.step(disc_optimizer)
                    scaler.update()

                    # Log discriminator scores as diagnostics only.
                    # Generator adversarial gradient is disabled during
                    # pretraining — the Gumbel-Softmax AR loop accumulates
                    # unstable gradients over 32 steps causing NaN.  The
                    # discriminator scores (D_r, D_f) serve as a monitoring
                    # signal for output quality.  Adversarial generator
                    # training is deferred to agent integration (frozen
                    # decoder, single projection layer, stable gradient).
                    adv_loss_val = disc_loss.item()
                    d_real_val = torch.sigmoid(real_logits).mean().item()
                    d_fake_val = torch.sigmoid(fake_logits).mean().item()

                if (i + 1) % accum == 0 or (i + 1) == len(train_loader):
                    scaler.unscale_(optimizer)
                    last_grad_norm = nn.utils.clip_grad_norm_(
                        all_params, max_norm=10.0
                    ).item()
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()
                    global_step += 1

                train_ce_sum += ce_loss.item() * b
                train_kl_sum += kl_loss.item() * b
                train_topo_sum += topo_loss.item() * b
                train_count += b

                # -- Per-batch logging --
                if (i + 1) % log_every == 0 or (i + 1) == num_batches:
                    elapsed = time.time() - batch_start
                    tokens_per_sec = (
                        log_every * b * cfg.max_seq_len / max(elapsed, 0.01)
                    )
                    # Active dims in this batch (raw KL > 0.1)
                    # GPU memory
                    if device.type == "cuda":
                        mem_mb = torch.cuda.memory_allocated(device) / 1e6
                    else:
                        mem_mb = 0.0

                    # Perplexity (exp(CE) — more interpretable)
                    ppl = min(math.exp(ce_loss.item()), 99999.0)

                    # Build optional metric strings
                    extra_parts: list[str] = [
                        f"PPL={ppl:.0f}",
                        f"gnorm={last_grad_norm:.2f}",
                    ]
                    if _do_kl:
                        raw_kl = kl_per_dim_train.detach()
                        active = int(
                            (raw_kl.mean(dim=0) > 0.1).sum().item()
                        )
                        extra_parts.append(
                            f"KL={kl_loss.item():.3f} "
                            f"kl_scale={kl_scale:.4f} "
                            f"active={active}/{cfg.latent_dim}"
                        )
                    if cfg.topo_weight > 0:
                        extra_parts.append(f"topo={topo_loss.item():.4f}")
                    if disc is not None and global_step >= cfg.adv_warmup_steps:
                        extra_parts.append(
                            f"D_r={d_real_val:.2f} D_f={d_fake_val:.2f}"
                            f" adv={adv_loss_val:.3f}"
                        )
                    extra_str = (" " + " ".join(extra_parts)) if extra_parts else ""

                    logger.info(
                        "  [%d/%d] step=%d CE=%.3f "
                        "%.0f tok/s %.0fMB%s",
                        i + 1,
                        num_batches,
                        global_step,
                        ce_loss.item(),
                        tokens_per_sec,
                        mem_mb,
                        extra_str,
                    )
                    batch_start = time.time()

            train_ce = train_ce_sum / max(train_count, 1)
            train_kl = train_kl_sum / max(train_count, 1)
            train_topo = train_topo_sum / max(train_count, 1)
            train_loss = train_ce + cfg.kl_weight * train_kl + cfg.topo_weight * train_topo

            # -- Validate --
            for m in modules.values():
                if isinstance(m, nn.Module):
                    m.eval()

            val_ce_sum = 0.0
            val_kl_sum = 0.0
            val_count = 0
            all_kl_per_dim: list[Tensor] = []

            with torch.no_grad():
                for batch_tokens, batch_lengths in val_loader:
                    batch_tokens = batch_tokens.to(device)
                    batch_lengths = torch.as_tensor(
                        batch_lengths, device=device
                    )
                    b = batch_tokens.size(0)

                    with torch.amp.autocast(
                        device_type=device.type, enabled=cfg.use_amp
                    ):
                        ce_loss, kl_loss, kl_per_dim, _, _ = _vae_forward(
                            batch_tokens,
                            batch_lengths,
                            bos_id=bos_id,
                            full_vocab=full_vocab,
                            kl_free_bits=0.0,
                            compute_kl=_do_kl,
                            **modules,
                        )

                    val_ce_sum += ce_loss.item() * b
                    val_kl_sum += kl_loss.item() * b
                    val_count += b
                    all_kl_per_dim.append(kl_per_dim.detach().cpu())

            val_ce = val_ce_sum / max(val_count, 1)
            val_kl = val_kl_sum / max(val_count, 1)
            val_loss = val_ce + cfg.kl_weight * val_kl

            epoch_time = time.time() - epoch_start

            # Build epoch summary with only active components
            epoch_parts = [f"Epoch {epoch + 1}/{cfg.num_epochs} ({epoch_time:.0f}s)"]
            epoch_parts.append(f"train: CE={train_ce:.4f}")
            if _do_kl:
                epoch_parts.append(f"KL={train_kl:.4f}")
            if cfg.topo_weight > 0:
                epoch_parts.append(f"topo={train_topo:.4f}")
            epoch_parts.append(f"total={train_loss:.4f}")
            epoch_parts.append(f"| val: CE={val_ce:.4f}")
            if _do_kl:
                epoch_parts.append(f"KL={val_kl:.4f}")
            epoch_parts.append(f"total={val_loss:.4f}")
            if _do_kl and all_kl_per_dim:
                kl_cat = torch.cat(all_kl_per_dim, dim=0)
                mean_kl_per_dim = kl_cat.mean(dim=0)
                active_dims = int((mean_kl_per_dim > 0.1).sum().item())
                epoch_parts.append(f"| active={active_dims}/{cfg.latent_dim}")

            logger.info("  ".join(epoch_parts))

            # -- Epoch-end evaluation: reconstruction, interpolation, perturbation, random --
            with torch.no_grad():

                def _sample_decode(
                    z: Tensor,
                    top_p: float = 0.9,
                    temperature: float = 0.8,
                ) -> list[str]:
                    """Decode z via nucleus (top-p) sampling.

                    Replaces greedy argmax + repetition penalties with
                    sampling from the top-p probability mass.  Naturally
                    produces diverse, non-repetitive output.
                    """
                    from lfm.generator.layers import (
                        LinguisticDecoder,
                        multiscale_causal_mask,
                    )

                    _dec = modules["decoder"]
                    _is_ling = isinstance(_dec, LinguisticDecoder)
                    n = z.size(0)
                    z_normed = modules["latent_norm"](z)
                    mem = modules["latent_to_decoder"](z_normed).unsqueeze(1)
                    ids = torch.full(
                        (n, 1), bos_id, dtype=torch.long, device=device
                    )
                    for _t in range(cfg.max_seq_len - 1):
                        if _is_ling:
                            tgt = modules["dec_token_embedding"](ids)
                            if not isinstance(
                                modules["dec_pos_embedding"], nn.Identity
                            ):
                                p = torch.arange(
                                    ids.size(1), device=device
                                ).unsqueeze(0)
                                tgt = tgt + modules["dec_pos_embedding"](p)
                            cm = multiscale_causal_mask(
                                ids.size(1),
                                num_heads=cfg.decoder_num_heads,
                                head_windows=cfg.attention_head_windows,
                                global_every=cfg.attention_global_every,
                                device=device,
                            )
                            out = _dec(
                                tgt, mem, tgt_mask=cm,
                                rope_freqs=modules.get("_rope_freqs"),
                            )
                        else:
                            p = torch.arange(
                                ids.size(1), device=device
                            ).unsqueeze(0)
                            tok = modules["dec_token_embedding"](ids)
                            tgt = tok + modules["dec_pos_embedding"](p)
                            cm = nn.Transformer.generate_square_subsequent_mask(
                                ids.size(1), device=device
                            )
                            out = _dec(tgt=tgt, memory=mem, tgt_mask=cm)

                        logits = modules["output_head"](out[:, -1])
                        # Suppress special tokens
                        logits[:, 0:4] = float("-inf")
                        logits[:, bos_id] = float("-inf")
                        # Temperature
                        logits = logits / max(temperature, 1e-8)
                        # Nucleus (top-p) filtering
                        sorted_logits, sorted_idx = torch.sort(
                            logits, descending=True
                        )
                        cumprobs = torch.cumsum(
                            F.softmax(sorted_logits, dim=-1), dim=-1
                        )
                        remove = cumprobs - F.softmax(
                            sorted_logits, dim=-1
                        ) >= top_p
                        sorted_logits[remove] = float("-inf")
                        logits = torch.zeros_like(logits).scatter_(
                            1, sorted_idx, sorted_logits
                        )
                        # Sample
                        probs = F.softmax(logits, dim=-1)
                        nxt = torch.multinomial(probs, num_samples=1)
                        ids = torch.cat([ids, nxt], dim=1)
                    # SPM special token IDs to exclude from decode
                    _spm_specials = {0, 1, 2, 3, bos_id, eos_id}
                    texts = []
                    for j in range(n):
                        toks = ids[j, 1:].cpu().tolist()
                        if eos_id in toks:
                            toks = toks[: toks.index(eos_id)]
                        toks = [
                            x for x in toks
                            if x < vocab_size and x not in _spm_specials
                        ]
                        text = sp.decode(toks)
                        # Strip trailing orphan consonants (truncation artifacts)
                        words = text.split()
                        while words and len(words[-1]) == 1 and not _is_vowel(words[-1]):
                            words.pop()
                        texts.append(" ".join(words))
                    return texts

                def _is_vowel(char: str) -> bool:
                    """Check if a single IPA character is a vowel."""
                    return char in _IPA_VOWELS

                def _encode_text(tokens: Tensor, lengths: Tensor) -> Tensor:
                    """Encode token batch -> z via the VAE encoder."""
                    sl = tokens.size(1)
                    src_mask = (
                        torch.arange(sl, device=device).unsqueeze(0)
                        < lengths.unsqueeze(1)
                    )
                    pos = torch.arange(sl, device=device).unsqueeze(0)
                    enc_tok = modules["enc_token_embedding"](tokens)
                    enc_in = enc_tok + modules["enc_pos_embedding"](pos)
                    enc_out = modules["encoder"](enc_in, src_key_padding_mask=~src_mask)
                    masked = enc_out * src_mask.unsqueeze(-1).float()
                    denom = lengths.unsqueeze(-1).float().clamp(min=1)
                    pooled = masked.sum(dim=1) / denom
                    h = modules["enc_to_latent"](pooled)
                    mu, _ = h.chunk(2, dim=-1)
                    return mu  # deterministic at eval

                # Fixed seed per epoch for reproducible sampling
                torch.manual_seed(cfg.seed + epoch)

                # Grab 2 real sentences from validation set
                val_batch_tokens, val_batch_lengths = next(iter(val_loader))
                val_batch_tokens = val_batch_tokens[:2].to(device)
                val_batch_lengths = torch.as_tensor(
                    val_batch_lengths[:2], device=device
                )

                # --- 1. Reconstruction ---
                z_real = _encode_text(val_batch_tokens, val_batch_lengths)
                recon_texts = _sample_decode(z_real)
                for j in range(2):
                    orig_ids = val_batch_tokens[j].cpu().tolist()
                    orig_ids = [x for x in orig_ids[:val_batch_lengths[j]] if x < vocab_size]
                    orig = sp.decode(orig_ids)
                    logger.info("  recon[%d] orig: %s", j, orig[:100])
                    logger.info("  recon[%d]  dec: %s", j, recon_texts[j][:100])

                # --- 2. Interpolation (between the two sentences) ---
                alphas = [0.0, 0.25, 0.5, 0.75, 1.0]
                z_interp = torch.stack(
                    [z_real[0] * (1 - a) + z_real[1] * a for a in alphas]
                )
                interp_texts = _sample_decode(z_interp)
                for k, (a, txt) in enumerate(zip(alphas, interp_texts)):
                    logger.info("  interp[%.2f]: %s", a, txt[:100])

                # --- 3. Perturbation (add small noise to z_real[0]) ---
                noise_scales = [0.0, 0.1, 0.5, 1.0]
                z_perturbed = torch.stack(
                    [z_real[0] + s * torch.randn_like(z_real[0]) for s in noise_scales]
                )
                perturb_texts = _sample_decode(z_perturbed)
                for k, (s, txt) in enumerate(zip(noise_scales, perturb_texts)):
                    logger.info("  perturb[σ=%.1f]: %s", s, txt[:100])

                # --- 4. Random z (prior sampling) ---
                z_random = torch.randn(3, cfg.latent_dim, device=device)
                random_texts = _sample_decode(z_random)
                for j, txt in enumerate(random_texts):
                    logger.info("  random[%d]: %s", j, txt[:100])

                # --- 5. Length distribution (autoregressive decode) ---
                # Decode a batch of random z to check EOS behavior and
                # length variation — the key metric for LayerNorm fix.
                z_length_test = torch.randn(32, cfg.latent_dim, device=device)
                length_texts = _sample_decode(z_length_test)
                word_counts = [len(t.split()) for t in length_texts]
                char_counts = [len(t) for t in length_texts]
                logger.info(
                    "  length_dist (32 random z): "
                    "words: min=%d max=%d mean=%.1f std=%.1f | "
                    "chars: min=%d max=%d mean=%.1f std=%.1f",
                    min(word_counts), max(word_counts),
                    sum(word_counts) / len(word_counts),
                    (sum((w - sum(word_counts) / len(word_counts)) ** 2
                         for w in word_counts) / len(word_counts)) ** 0.5,
                    min(char_counts), max(char_counts),
                    sum(char_counts) / len(char_counts),
                    (sum((c - sum(char_counts) / len(char_counts)) ** 2
                         for c in char_counts) / len(char_counts)) ** 0.5,
                )

                # --- 6. Structural health metrics ---
                all_eval_texts = recon_texts + interp_texts + perturb_texts + random_texts

                def _struct_metrics(texts: list[str]) -> dict[str, float]:
                    ttrs, rep_rates, word_lens = [], [], []
                    for t in texts:
                        words = t.split()
                        if not words:
                            continue
                        # Type-token ratio
                        ttrs.append(len(set(words)) / len(words))
                        # Bigram repetition rate
                        bigrams = [
                            (words[i], words[i + 1])
                            for i in range(len(words) - 1)
                        ]
                        if bigrams:
                            rep_rates.append(
                                1.0 - len(set(bigrams)) / len(bigrams)
                            )
                        # Mean word length (IPA chars)
                        word_lens.extend(len(w) for w in words)
                    return {
                        "ttr": sum(ttrs) / max(len(ttrs), 1),
                        "rep_rate": sum(rep_rates) / max(len(rep_rates), 1),
                        "mean_word_len": (
                            sum(word_lens) / max(len(word_lens), 1)
                        ),
                    }

                m = _struct_metrics(all_eval_texts)
                logger.info(
                    "  struct: TTR=%.3f rep_rate=%.3f "
                    "mean_word_len=%.1f",
                    m["ttr"],
                    m["rep_rate"],
                    m["mean_word_len"],
                )

            # Save best checkpoint (decoder only)
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_metrics = {
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                    "val_ce": val_ce,
                    "val_kl": val_kl,
                    "num_samples": float(len(token_ids_list)),
                    "epoch": float(epoch + 1),
                }

                output_path = Path(cfg.output_path)
                output_path.parent.mkdir(parents=True, exist_ok=True)
                # Save decoder-only checkpoint (for inference)
                torch.save(
                    {
                        "latent_dim": cfg.latent_dim,
                        "vocab_size": vocab_size,
                        "decoder_hidden_dim": cfg.decoder_hidden_dim,
                        "decoder_num_layers": cfg.decoder_num_layers,
                        "decoder_num_heads": cfg.decoder_num_heads,
                        "max_seq_len": cfg.max_seq_len,
                        "latent_norm": modules[
                            "latent_norm"
                        ].state_dict(),
                        "latent_to_decoder": modules[
                            "latent_to_decoder"
                        ].state_dict(),
                        "token_embedding": modules[
                            "dec_token_embedding"
                        ].state_dict(),
                        "pos_embedding": modules[
                            "dec_pos_embedding"
                        ].state_dict(),
                        "decoder": modules["decoder"].state_dict(),
                        "output_head": modules["output_head"].state_dict(),
                        "train_loss": train_loss,
                        "val_loss": val_loss,
                    },
                    output_path,
                )
                logger.info("Saved best decoder checkpoint to %s", output_path)

            # Save full training state for resume (every epoch)
            resume_path = Path(output_dir) / "vae_resume.pt"
            torch.save(
                {
                    "epoch": epoch + 1,
                    "global_step": global_step,
                    "best_val_loss": best_val_loss,
                    "modules": {
                        k: m.state_dict()
                        for k, m in modules.items()
                        if isinstance(m, nn.Module)
                    },
                    "optimizer": optimizer.state_dict(),
                    "scaler": scaler.state_dict(),
                },
                resume_path,
            )

        return best_metrics

    @staticmethod
    def _build_model(
        cfg: VAEPretrainConfig,
        hidden: int,
        full_vocab: int,
        device: torch.device,
    ) -> dict[str, nn.Module]:
        """Build all encoder and decoder components.

        Returns a dict of named modules for clean forwarding via ``**kwargs``.
        """
        # Encoder
        enc_token_embedding = nn.Embedding(full_vocab, hidden).to(device)
        enc_pos_embedding = nn.Embedding(cfg.max_seq_len, hidden).to(device)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden,
            nhead=cfg.decoder_num_heads,
            dim_feedforward=hidden * 4,
            dropout=cfg.decoder_dropout,
            batch_first=True,
        )
        encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=cfg.encoder_num_layers,
            enable_nested_tensor=False,
        ).to(device)
        enc_to_latent = nn.Linear(hidden, cfg.latent_dim * 2).to(device)

        # Decoder (linguistic architecture — must match GeneratorConfig)
        from lfm.generator.layers import (
            LinguisticDecoder,
            multiscale_causal_mask,
            precompute_rope_freqs,
        )

        latent_norm = nn.LayerNorm(cfg.latent_dim).to(device)
        latent_to_decoder = nn.Linear(cfg.latent_dim, hidden).to(device)
        dec_token_embedding = nn.Embedding(full_vocab, hidden).to(device)

        # Positional embedding: only used when RoPE is disabled
        dec_pos_embedding: nn.Module
        if cfg.use_rope:
            # Dummy — not used, but kept in module dict for checkpoint compat
            dec_pos_embedding = nn.Identity()
        else:
            dec_pos_embedding = nn.Embedding(cfg.max_seq_len, hidden).to(device)

        decoder = LinguisticDecoder(
            d_model=hidden,
            nhead=cfg.decoder_num_heads,
            num_layers=cfg.decoder_num_layers,
            dim_feedforward=hidden * 4,
            dropout=cfg.decoder_dropout,
            share_layers=cfg.share_decoder_layers,
        ).to(device)
        output_head = nn.Linear(hidden, full_vocab).to(device)

        # Precompute RoPE frequencies
        rope_freqs = None
        if cfg.use_rope:
            head_dim = hidden // cfg.decoder_num_heads
            rope_freqs = precompute_rope_freqs(
                head_dim, cfg.max_seq_len, device=device
            )

        # Precompute multi-scale causal mask (constant for fixed seq_len)
        cached_mask = multiscale_causal_mask(
            cfg.max_seq_len,
            num_heads=cfg.decoder_num_heads,
            head_windows=cfg.attention_head_windows,
            global_every=cfg.attention_global_every,
            device=device,
        )

        return {
            "enc_token_embedding": enc_token_embedding,
            "enc_pos_embedding": enc_pos_embedding,
            "encoder": encoder,
            "enc_to_latent": enc_to_latent,
            "latent_norm": latent_norm,
            "latent_to_decoder": latent_to_decoder,
            "dec_token_embedding": dec_token_embedding,
            "dec_pos_embedding": dec_pos_embedding,
            "decoder": decoder,
            "output_head": output_head,
            "_rope_freqs": rope_freqs,
            "_cached_mask": cached_mask,
            "_cfg": cfg,
        }


def pretrain_vae_decoder(config: VAEPretrainConfig) -> dict[str, float]:
    """Convenience function: create ``VAEPretrainer`` and run.

    Args:
        config: Pretraining configuration.

    Returns:
        Metrics dict with ``train_loss``, ``val_loss``,
        ``active_latent_dims``, and ``num_samples``.
    """
    return VAEPretrainer(config).pretrain()

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
    latent_dim: int = 128
    encoder_num_layers: int = 2
    decoder_hidden_dim: int = 256
    decoder_num_layers: int = 2
    decoder_num_heads: int = 4
    decoder_dropout: float = 0.2
    max_seq_len: int = 64
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
    topo_weight: float = 0.1
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
    encoder: nn.TransformerEncoder,
    enc_to_latent: nn.Linear,
    latent_to_decoder: nn.Linear,
    dec_token_embedding: nn.Embedding,
    dec_pos_embedding: nn.Embedding,
    decoder: nn.TransformerDecoder,
    output_head: nn.Linear,
    bos_id: int,
    full_vocab: int,
    kl_free_bits: float,
    compute_kl: bool = True,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
    """Run one VAE forward pass (teacher-forced).

    Args:
        compute_kl: If ``False``, skip KL computation entirely and return
            zeros.  Saves compute when ``kl_weight=0``.

    Returns:
        Tuple of ``(ce_loss, kl_loss, kl_per_dim, z, dec_hidden)`` where:

        - ``kl_per_dim``: ``(batch, latent_dim)`` for active-dims analysis.
        - ``z``: ``(batch, latent_dim)`` sampled latent codes.
        - ``dec_hidden``: ``(batch, seq_len, hidden)`` decoder output states.
    """
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

    # Decode (teacher-forced)
    memory = latent_to_decoder(z).unsqueeze(1)
    bos_col = torch.full((b, 1), bos_id, dtype=torch.long, device=device)
    dec_input_ids = torch.cat([bos_col, batch_tokens[:, :-1]], dim=1)
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

    memory = latent_to_decoder(z).unsqueeze(1)  # (B, 1, H)
    bos_embed = dec_token_embedding(
        torch.full((batch, 1), bos_id, dtype=torch.long, device=device)
    )
    generated_embeds = bos_embed

    all_probs: list[Tensor] = []

    for _t in range(max_len):
        seq_len = generated_embeds.size(1)
        pos = torch.arange(seq_len, device=device).unsqueeze(0)
        tgt = generated_embeds + dec_pos_embedding(pos)
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

            # 3. Tokenize
            token_ids_list = []
            for line in lines:
                ids = sp.encode(line, out_type=int)
                if ids:
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

        # 6. Training loop
        import time

        best_val_loss = float("inf")
        best_metrics: dict[str, float] = {}
        global_step = 0
        accum = cfg.gradient_accumulation_steps
        log_every = 50  # log every N batches
        num_batches = len(train_loader)

        # Parameter count for logging
        total_params = sum(p.numel() for m in modules.values() for p in m.parameters())
        logger.info(
            "Model: %d params (%.1fM), %d train batches/epoch",
            total_params,
            total_params / 1e6,
            num_batches,
        )

        for epoch in range(cfg.num_epochs):
            # -- Train --
            for m in modules.values():
                m.train()

            train_ce_sum = 0.0
            train_kl_sum = 0.0
            train_topo_sum = 0.0
            train_count = 0
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
                    nn.utils.clip_grad_norm_(all_params, max_norm=10.0)
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
                    raw_kl = kl_per_dim_train.detach()
                    active = int(
                        (raw_kl.mean(dim=0) > 0.1).sum().item()
                    )
                    # GPU memory
                    if device.type == "cuda":
                        mem_mb = torch.cuda.memory_allocated(device) / 1e6
                    else:
                        mem_mb = 0.0

                    adv_str = ""
                    if disc is not None and global_step >= cfg.adv_warmup_steps:
                        adv_str = (
                            f" D_r={d_real_val:.2f} D_f={d_fake_val:.2f}"
                            f" adv={adv_loss_val:.3f}"
                        )

                    logger.info(
                        "  [%d/%d] step=%d CE=%.3f KL=%.3f topo=%.4f "
                        "kl_scale=%.4f active=%d/%d "
                        "%.0f tok/s %.0fMB%s",
                        i + 1,
                        num_batches,
                        global_step,
                        ce_loss.item(),
                        kl_loss.item(),
                        topo_loss.item(),
                        kl_scale,
                        active,
                        cfg.latent_dim,
                        tokens_per_sec,
                        mem_mb,
                        adv_str,
                    )
                    batch_start = time.time()

            train_ce = train_ce_sum / max(train_count, 1)
            train_kl = train_kl_sum / max(train_count, 1)
            train_topo = train_topo_sum / max(train_count, 1)
            train_loss = train_ce + cfg.kl_weight * train_kl + cfg.topo_weight * train_topo

            # -- Validate --
            for m in modules.values():
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

            # Active latent dims: mean KL > threshold per dimension
            kl_cat = torch.cat(all_kl_per_dim, dim=0)  # (N, latent_dim)
            mean_kl_per_dim = kl_cat.mean(dim=0)  # (latent_dim,)
            active_dims = int((mean_kl_per_dim > 0.1).sum().item())

            epoch_time = time.time() - epoch_start

            logger.info(
                "Epoch %d/%d (%.0fs) — train: CE=%.4f KL=%.4f topo=%.4f "
                "total=%.4f | val: CE=%.4f KL=%.4f total=%.4f | "
                "active_dims=%d/%d",
                epoch + 1,
                cfg.num_epochs,
                epoch_time,
                train_ce,
                train_kl,
                train_topo,
                train_loss,
                val_ce,
                val_kl,
                val_loss,
                active_dims,
                cfg.latent_dim,
            )

            # -- Epoch-end evaluation: reconstruction, interpolation, perturbation, random --
            with torch.no_grad():

                def _greedy_decode(z: Tensor) -> list[str]:
                    """Greedy decode z -> token ids -> text."""
                    n = z.size(0)
                    mem = modules["latent_to_decoder"](z).unsqueeze(1)
                    ids = torch.full(
                        (n, 1), bos_id, dtype=torch.long, device=device
                    )
                    for _t in range(cfg.max_seq_len - 1):
                        p = torch.arange(ids.size(1), device=device).unsqueeze(0)
                        tgt = modules["dec_token_embedding"](ids) + modules["dec_pos_embedding"](p)
                        cm = nn.Transformer.generate_square_subsequent_mask(
                            ids.size(1), device=device
                        )
                        out = modules["decoder"](tgt=tgt, memory=mem, tgt_mask=cm)
                        nxt = modules["output_head"](out[:, -1:]).argmax(dim=-1)
                        ids = torch.cat([ids, nxt], dim=1)
                    texts = []
                    for j in range(n):
                        toks = ids[j, 1:].cpu().tolist()
                        if eos_id in toks:
                            toks = toks[: toks.index(eos_id)]
                        toks = [x for x in toks if x < vocab_size]
                        texts.append(sp.decode(toks))
                    return texts

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

                # Grab 2 real sentences from validation set
                val_batch_tokens, val_batch_lengths = next(iter(val_loader))
                val_batch_tokens = val_batch_tokens[:2].to(device)
                val_batch_lengths = torch.as_tensor(
                    val_batch_lengths[:2], device=device
                )

                # --- 1. Reconstruction ---
                z_real = _encode_text(val_batch_tokens, val_batch_lengths)
                recon_texts = _greedy_decode(z_real)
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
                interp_texts = _greedy_decode(z_interp)
                for k, (a, txt) in enumerate(zip(alphas, interp_texts)):
                    logger.info("  interp[%.2f]: %s", a, txt[:100])

                # --- 3. Perturbation (add small noise to z_real[0]) ---
                noise_scales = [0.0, 0.1, 0.5, 1.0]
                z_perturbed = torch.stack(
                    [z_real[0] + s * torch.randn_like(z_real[0]) for s in noise_scales]
                )
                perturb_texts = _greedy_decode(z_perturbed)
                for k, (s, txt) in enumerate(zip(noise_scales, perturb_texts)):
                    logger.info("  perturb[σ=%.1f]: %s", s, txt[:100])

                # --- 4. Random z (prior sampling) ---
                z_random = torch.randn(3, cfg.latent_dim, device=device)
                random_texts = _greedy_decode(z_random)
                for j, txt in enumerate(random_texts):
                    logger.info("  random[%d]: %s", j, txt[:100])

            # Save best checkpoint (decoder only)
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_metrics = {
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                    "val_ce": val_ce,
                    "val_kl": val_kl,
                    "active_latent_dims": float(active_dims),
                    "num_samples": float(len(token_ids_list)),
                }

                output_path = Path(cfg.output_path)
                output_path.parent.mkdir(parents=True, exist_ok=True)
                torch.save(
                    {
                        "latent_dim": cfg.latent_dim,
                        "vocab_size": vocab_size,
                        "decoder_hidden_dim": cfg.decoder_hidden_dim,
                        "decoder_num_layers": cfg.decoder_num_layers,
                        "decoder_num_heads": cfg.decoder_num_heads,
                        "max_seq_len": cfg.max_seq_len,
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

        # Decoder (architecture must match GeneratorConfig)
        latent_to_decoder = nn.Linear(cfg.latent_dim, hidden).to(device)
        dec_token_embedding = nn.Embedding(full_vocab, hidden).to(device)
        dec_pos_embedding = nn.Embedding(cfg.max_seq_len, hidden).to(device)
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=hidden,
            nhead=cfg.decoder_num_heads,
            dim_feedforward=hidden * 4,
            dropout=cfg.decoder_dropout,
            batch_first=True,
        )
        decoder = nn.TransformerDecoder(
            decoder_layer, num_layers=cfg.decoder_num_layers
        ).to(device)
        output_head = nn.Linear(hidden, full_vocab).to(device)

        return {
            "enc_token_embedding": enc_token_embedding,
            "enc_pos_embedding": enc_pos_embedding,
            "encoder": encoder,
            "enc_to_latent": enc_to_latent,
            "latent_to_decoder": latent_to_decoder,
            "dec_token_embedding": dec_token_embedding,
            "dec_pos_embedding": dec_pos_embedding,
            "decoder": decoder,
            "output_head": output_head,
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

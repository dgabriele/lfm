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

import atexit
import logging
import signal
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

    dataset_path: str | None = None
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
    max_seq_len: int = 96
    encoder_pooling: str = "mean"  # "mean" or "attention"
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

    # Latent variance regularization: smooth quadratic penalty that
    # pulls per-dimension z variance toward ``z_var_target``.  Provides
    # continuous gradient signal (not just a hard floor), preventing
    # the slow variance collapse that leads to gnorm explosion.
    # Loss = weight * mean((var_per_dim - target)^2).
    z_var_weight: float = 5.0
    z_var_target: float = 0.05
    z_var_floor: float = 0.01  # legacy, unused — kept for checkpoint compat

    # Word dropout (Bowman et al. 2016): randomly zero out decoder input
    # embeddings with this probability during training.  Forces the decoder
    # to rely on z for reconstruction instead of the teacher-forced input
    # highway, preventing the encoder from driving logvar → -∞.
    # Annealed linearly from word_dropout to word_dropout_min over
    # word_dropout_anneal_epochs epochs.
    word_dropout: float = 0.3
    word_dropout_min: float = 0.05
    word_dropout_anneal_epochs: int = 20

    # DIP-VAE off-diagonal covariance penalty: encourages z dimensions
    # to be statistically independent (disentangled).  Penalizes pairwise
    # correlation between dimensions without pushing mean toward zero.
    dip_weight: float = 0.1

    # Cosine LR decay: minimum LR at end of training.
    lr_min: float = 1e-4

    # Legacy topological regularization (disabled — replaced by z_var)
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

    # Scheduled sampling: anneal from 0 to target probability over warmup
    # epochs, starting at ``scheduled_sampling_start_epoch``.  At each
    # decoder position, replace ground truth input with the model's own
    # prediction with this probability.  Prevents exposure bias and
    # generation degeneration at long sequence lengths.  Must start late
    # enough that the model produces reasonable predictions to feed back.
    scheduled_sampling_target: float = 0.0
    scheduled_sampling_start_epoch: int = 5
    scheduled_sampling_warmup_epochs: int = 5

    # Contrastive learning (InfoNCE): encourages similar source sentences
    # to have similar z vectors.  Requires pre-computed sentence-transformer
    # embeddings aligned by index with the training corpus.
    contrastive_weight: float = 0.0
    contrastive_temperature: float = 0.07
    embeddings_path: str = ""

    # Fixed KL for distribution smoothness (no warmup/cycling).
    # Applied without free bits — raw KL(q||N(0,1)).
    kl_beta: float = 0.0


# ---------------------------------------------------------------------------
# Corpus loading
# ---------------------------------------------------------------------------


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


def _info_nce_loss(
    z: Tensor,
    embeddings: Tensor,
    temperature: float = 0.07,
) -> Tensor:
    """InfoNCE contrastive loss between z vectors and source embeddings.

    Pulls z vectors of semantically similar sentences together and
    pushes dissimilar ones apart.  Similarity is defined by the
    pre-computed sentence-transformer embeddings.

    Uses a symmetric formulation: z→embed and embed→z directions
    are averaged for stable gradients.

    Args:
        z: Latent codes ``(batch, latent_dim)``.
        embeddings: Pre-computed source text embeddings ``(batch, embed_dim)``.
        temperature: Softmax temperature (lower = sharper).

    Returns:
        Scalar InfoNCE loss.
    """
    # Normalize both to unit vectors
    z_norm = F.normalize(z.float(), dim=-1)
    e_norm = F.normalize(embeddings.float(), dim=-1)

    # Cross-similarity matrix: (B, B)
    logits_ze = z_norm @ e_norm.T / temperature
    logits_ez = e_norm @ z_norm.T / temperature

    # Labels: diagonal (each z should match its own embedding)
    labels = torch.arange(z.size(0), device=z.device)

    loss_ze = F.cross_entropy(logits_ze, labels)
    loss_ez = F.cross_entropy(logits_ez, labels)
    return (loss_ze + loss_ez) / 2


def _dip_covariance_loss(z: Tensor) -> Tensor:
    """Off-diagonal covariance penalty (DIP-VAE-I style).

    Penalizes pairwise correlation between z dimensions, encouraging
    each dimension to capture independent information.  Computed in
    float32 for numerical stability under AMP.

    Args:
        z: Latent codes ``(batch, latent_dim)``.

    Returns:
        Scalar loss — mean squared off-diagonal covariance, normalized
        by ``latent_dim²`` for scale-independent weighting.
    """
    z_centered = (z - z.mean(dim=0)).float()
    cov = (z_centered.T @ z_centered) / max(z.size(0) - 1, 1)
    # Zero diagonal — only penalize off-diagonal (correlations)
    off_diag = cov - torch.diag(cov.diag())
    return off_diag.pow(2).sum() / z.size(1) ** 2


def _vae_forward(
    batch_tokens: Tensor,
    batch_lengths: Tensor,
    *,
    enc_token_embedding: nn.Embedding,
    enc_pos_embedding: nn.Embedding,
    encoder: nn.Module,
    enc_to_latent: nn.Linear,
    latent_to_decoder: nn.Linear,
    dec_token_embedding: nn.Embedding,
    dec_pos_embedding: nn.Module,
    decoder: nn.Module,
    output_head: nn.Linear,
    bos_id: int,
    full_vocab: int,
    kl_free_bits: float,
    compute_kl: bool = True,
    _rope_freqs: Tensor | None = None,
    _cached_mask: Tensor | None = None,
    _cfg: object | None = None,
    _attn_pool_query: Tensor | None = None,
    scheduled_sampling_p: float = 0.0,
    _word_dropout_p: float = 0.0,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
    """Run one VAE forward pass with optional scheduled sampling.

    When ``scheduled_sampling_p > 0``, a two-pass approach is used:
    first a full teacher-forced pass to get predictions, then a second
    pass with a mixed input where each position uses the model's own
    argmax prediction with probability ``scheduled_sampling_p``, or the
    ground truth otherwise.  This teaches the decoder to recover from
    its own outputs, preventing degeneration during free-run generation
    at long sequence lengths.

    Returns:
        Tuple of ``(ce_loss, kl_loss, kl_per_dim, z, dec_hidden, mu, logvar)``.
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

    # Pool encoder outputs to a single vector
    if _cfg is not None and getattr(_cfg, "encoder_pooling", "mean") == "attention":
        # Attention pooling: learned query attends to encoder outputs
        attn_query = _attn_pool_query  # (1, 1, hidden)
        attn_weights = torch.bmm(
            attn_query.expand(b, -1, -1), enc_out.transpose(1, 2)
        )  # (b, 1, seq_len)
        attn_weights = attn_weights.masked_fill(~src_mask.unsqueeze(1), float("-inf"))
        attn_weights = F.softmax(attn_weights, dim=-1)
        pooled = torch.bmm(attn_weights, enc_out).squeeze(1)  # (b, hidden)
    else:
        # Mean pooling (default)
        enc_masked = enc_out * src_mask.unsqueeze(-1).float()
        denom = batch_lengths.unsqueeze(-1).float().clamp(min=1)
        pooled = enc_masked.sum(dim=1) / denom

    # Latent
    h = enc_to_latent(pooled)
    mu, logvar = h.chunk(2, dim=-1)
    std = (0.5 * logvar).exp()
    z = mu + std * torch.randn_like(std)

    # Decode
    memory = latent_to_decoder(z).unsqueeze(1)
    bos_col = torch.full((b, 1), bos_id, dtype=torch.long, device=device)
    teacher_input_ids = torch.cat([bos_col, batch_tokens[:, :-1]], dim=1)

    # Precompute mask
    if isinstance(decoder, LinguisticDecoder):
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
    else:
        tgt_mask = nn.Transformer.generate_square_subsequent_mask(
            seq_len, device=device
        )

    # Word dropout rate (0 at eval, passed in from training loop)
    _word_drop_p = _word_dropout_p if dec_token_embedding.training else 0.0

    def _run_decoder(input_ids: Tensor) -> Tensor:
        if isinstance(decoder, LinguisticDecoder):
            dec_input = dec_token_embedding(input_ids)
            # Word dropout: zero out embeddings to force decoder to use z
            if _word_drop_p > 0:
                drop_mask = torch.rand(
                    dec_input.shape[:2], device=device,
                ) < _word_drop_p
                drop_mask[:, 0] = False  # never drop BOS
                dec_input = dec_input.masked_fill(
                    drop_mask.unsqueeze(-1), 0.0,
                )
            if not isinstance(dec_pos_embedding, nn.Identity):
                dec_input = dec_input + dec_pos_embedding(pos_ids)
            return decoder(
                dec_input, memory, tgt_mask=tgt_mask, rope_freqs=_rope_freqs
            )
        else:
            dec_input = (
                dec_token_embedding(input_ids) + dec_pos_embedding(pos_ids)
            )
            # Word dropout for non-linguistic decoder
            if _word_drop_p > 0:
                drop_mask = torch.rand(
                    dec_input.shape[:2], device=device,
                ) < _word_drop_p
                drop_mask[:, 0] = False
                dec_input = dec_input.masked_fill(
                    drop_mask.unsqueeze(-1), 0.0,
                )
            return decoder(tgt=dec_input, memory=memory, tgt_mask=tgt_mask)

    if scheduled_sampling_p > 0:
        # Pass 1: teacher-forced to get predictions
        with torch.no_grad():
            tf_out = _run_decoder(teacher_input_ids)
            predicted_ids = output_head(tf_out).argmax(dim=-1)  # (b, seq_len)

        # Build mixed input: for each position, use model's prediction
        # with probability p, ground truth otherwise.  BOS (position 0)
        # is always ground truth.
        mix_mask = torch.rand(b, seq_len, device=device) < scheduled_sampling_p
        mix_mask[:, 0] = False  # always use BOS
        mixed_input_ids = torch.where(
            mix_mask,
            torch.cat([bos_col, predicted_ids[:, :-1]], dim=1),
            teacher_input_ids,
        )

        # Pass 2: decode with mixed input
        dec_out = _run_decoder(mixed_input_ids)
    else:
        # Pure teacher forcing
        dec_out = _run_decoder(teacher_input_ids)

    logits = output_head(dec_out)

    # Reconstruction loss (masked CE) — always against ground truth targets
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

    return ce_loss, kl_loss, kl_per_dim, z, dec_out, mu, logvar


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

        def _file_hash(path: str | Path) -> str:
            """SHA-256 of a file for consistency checks."""
            import hashlib

            h = hashlib.sha256()
            with open(path, "rb") as f:
                for chunk in iter(lambda: f.read(8192), b""):
                    h.update(chunk)
            return h.hexdigest()[:16]

        # Cache v3: includes 'languages' and 'spm_hash' for consistency.
        _cache_valid = False
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

            # 3. Tokenize, keeping language labels aligned
            _spm_specials = {0, 1, 2, 3}
            token_ids_list = []
            languages_list = []
            for lang, ipa in labeled:
                ids = sp.encode(ipa, out_type=int)
                ids = [x for x in ids if x not in _spm_specials]
                if len(ids) >= 5:
                    token_ids_list.append(ids)
                    languages_list.append(lang)

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
            if len(emb_np) != len(dataset):
                raise ValueError(
                    f"Embeddings length {len(emb_np)} != dataset length {len(dataset)}. "
                    f"Regenerate embeddings."
                )
            corpus_embeddings = torch.from_numpy(emb_np).float().pin_memory()
            logger.info(
                "Loaded contrastive embeddings: %s (%.0f MB, pinned)",
                corpus_embeddings.shape, corpus_embeddings.nbytes / 1e6,
            )

        # Wrap dataset to return indices when contrastive is active
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
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=cfg.num_epochs, eta_min=cfg.lr_min,
        )
        # Fixed-scale GradScaler: init at 256, effectively never grow.
        # The scaler exists only to prevent fp16 gradient underflow.
        # NOTE: PyTorch requires growth_factor > 1.0 (asserts on ==1.0),
        # so we use 1.001 + growth_interval=100K to make growth negligible.
        scaler = torch.amp.GradScaler(
            enabled=cfg.use_amp,
            init_scale=2**8,         # 256 — safe for 27M param model
            growth_factor=1.001,     # effectively never grow (PyTorch requires >1.0)
            backoff_factor=0.5,      # halve on overflow (recovers if needed)
            growth_interval=100000,  # check growth every 100K steps (~never)
        )

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
        current_spm_hash = _file_hash(spm_path)
        if resume_path.exists():
            logger.info("Resuming from %s", resume_path)
            ckpt = torch.load(resume_path, map_location=device, weights_only=False)
            # Verify SPM consistency
            ckpt_spm_hash = ckpt.get("spm_hash")
            if ckpt_spm_hash and ckpt_spm_hash != current_spm_hash:
                raise RuntimeError(
                    f"SPM model mismatch: checkpoint was trained with "
                    f"spm_hash={ckpt_spm_hash} but current spm.model has "
                    f"hash={current_spm_hash}. Delete vae_resume.pt to "
                    f"start fresh, or restore the matching spm.model."
                )
            for k, m in modules.items():
                if isinstance(m, nn.Module) and k in ckpt["modules"]:
                    m.load_state_dict(ckpt["modules"][k])
            optimizer.load_state_dict(ckpt["optimizer"])
            # Skip restoring scaler state — use fresh low-scale init
            # to avoid fp16 overflow from high saved scales.
            # scaler.load_state_dict(ckpt["scaler"])
            if "scheduler" in ckpt:
                scheduler.load_state_dict(ckpt["scheduler"])
            start_epoch = ckpt["epoch"]
            global_step = ckpt["global_step"]
            best_val_loss = ckpt["best_val_loss"]
            logger.info(
                "Resumed at epoch %d, step %d, best_val=%.4f",
                start_epoch, global_step, best_val_loss,
            )

        # 7. Save frozen config snapshot for provenance
        import yaml as _yaml

        config_snapshot_path = Path(output_dir) / "config.yaml"
        if not config_snapshot_path.exists():
            Path(output_dir).mkdir(parents=True, exist_ok=True)
            with open(config_snapshot_path, "w") as _cf:
                _yaml.dump(
                    cfg.model_dump() if hasattr(cfg, "model_dump") else vars(cfg),
                    _cf,
                    default_flow_style=False,
                )
            logger.info("Saved config snapshot to %s", config_snapshot_path)

        # 8. Training history
        from lfm.generator.training_history import TrainingHistory

        history = TrainingHistory(output_dir)
        history.start_session(
            start_epoch=start_epoch,
            config=cfg,
            spm_hash=current_spm_hash,
        )

        # -- Graceful shutdown: save history on SIGTERM/SIGINT/atexit --
        _session_ended = False
        # Mutable state container so signal/atexit handlers see latest values.
        _shutdown_state: dict[str, Any] = {
            "epoch": start_epoch,
            "best_val_loss": best_val_loss,
        }

        def _end_session_once() -> None:
            nonlocal _session_ended
            if _session_ended:
                return
            _session_ended = True
            history.end_session(
                end_epoch=_shutdown_state["epoch"],
                best_val_loss=_shutdown_state["best_val_loss"],
            )

        def _signal_handler(signum: int, frame: Any) -> None:
            _end_session_once()
            # Re-raise with default handler so exit code reflects the signal.
            signal.signal(signum, signal.SIG_DFL)
            signal.raise_signal(signum)

        signal.signal(signal.SIGTERM, _signal_handler)
        signal.signal(signal.SIGINT, _signal_handler)
        atexit.register(_end_session_once)

        # 8. Training loop
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

        # Running z statistics for latent calibration.  Tracked across
        # all training steps and saved with the decoder checkpoint so the
        # agent's input projection can be calibrated at inference time.
        z_running_mean = torch.zeros(cfg.latent_dim, device=device)
        z_running_std = torch.ones(cfg.latent_dim, device=device)
        z_stats_initialized = False
        z_stats_momentum = 0.01

        for epoch in range(start_epoch, cfg.num_epochs):
            # -- Word dropout: anneal from word_dropout to word_dropout_min --
            if cfg.word_dropout > 0 and cfg.word_dropout_anneal_epochs > 0:
                wd_frac = min(epoch / cfg.word_dropout_anneal_epochs, 1.0)
                wd_p = cfg.word_dropout + wd_frac * (cfg.word_dropout_min - cfg.word_dropout)
            else:
                wd_p = cfg.word_dropout

            # -- Scheduled sampling: anneal from 0 to target --
            ss_p = 0.0
            if (
                cfg.scheduled_sampling_target > 0
                and epoch >= cfg.scheduled_sampling_start_epoch
            ):
                ss_frac = min(
                    (epoch - cfg.scheduled_sampling_start_epoch)
                    / max(cfg.scheduled_sampling_warmup_epochs, 1),
                    1.0,
                )
                ss_p = ss_frac * cfg.scheduled_sampling_target

            # -- Train --
            for m in modules.values():
                if isinstance(m, nn.Module):
                    m.train()

            train_ce_sum = 0.0
            train_kl_sum = 0.0
            train_zvar_sum = 0.0
            train_dip_sum = 0.0
            train_cl_sum = 0.0
            train_klb_sum = 0.0
            train_count = 0
            last_grad_norm = 0.0
            optimizer.zero_grad()
            epoch_start = time.time()
            batch_start = time.time()

            for i, batch_data in enumerate(train_loader):
                if _use_contrastive:
                    batch_tokens, batch_lengths, batch_indices = batch_data
                else:
                    batch_tokens, batch_lengths = batch_data
                    batch_indices = None
                batch_tokens = batch_tokens.to(device)
                batch_lengths = torch.as_tensor(batch_lengths, device=device)
                b = batch_tokens.size(0)

                with torch.amp.autocast(
                    device_type=device.type, enabled=cfg.use_amp,
                ):
                    _do_kl = cfg.kl_weight > 0 or cfg.kl_beta > 0
                    (ce_loss, kl_loss, kl_per_dim_train,
                     z_batch, dec_hidden, mu_batch, logvar_batch) = (
                        _vae_forward(
                            batch_tokens,
                            batch_lengths,
                            bos_id=bos_id,
                            full_vocab=full_vocab,
                            kl_free_bits=cfg.kl_free_bits,
                            compute_kl=_do_kl,
                            scheduled_sampling_p=ss_p,
                            _word_dropout_p=wd_p,
                            **modules,
                        )
                    )

                    # Track z distribution for latent calibration
                    with torch.no_grad():
                        batch_z_mean = z_batch.mean(dim=0)
                        batch_z_std = z_batch.std(dim=0).clamp(min=1e-6)
                        if not z_stats_initialized:
                            z_running_mean.copy_(batch_z_mean)
                            z_running_std.copy_(batch_z_std)
                            z_stats_initialized = True
                        else:
                            z_running_mean.lerp_(batch_z_mean, z_stats_momentum)
                            z_running_std.lerp_(batch_z_std, z_stats_momentum)

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

                    # Smooth variance regularization: quadratic pull toward target.
                    # Provides gradient signal at all times, not just below a floor.
                    z_var_loss = torch.tensor(0.0, device=device)
                    if cfg.z_var_weight > 0 and b >= 4:
                        z_var = z_batch.var(dim=0)
                        z_var_loss = (z_var - cfg.z_var_target).pow(2).mean()

                    # DIP covariance: encourage dimension independence
                    dip_loss = torch.tensor(0.0, device=device)
                    if cfg.dip_weight > 0 and b >= 4:
                        dip_loss = _dip_covariance_loss(z_batch)

                    # Contrastive loss (InfoNCE)
                    cl_loss = torch.tensor(0.0, device=device)
                    if _use_contrastive and batch_indices is not None:
                        batch_embs = corpus_embeddings[batch_indices].to(device)
                        cl_loss = _info_nce_loss(
                            z_batch, batch_embs, cfg.contrastive_temperature,
                        )

                    # Fixed KL-beta (raw KL without free bits)
                    kl_beta_loss = torch.tensor(0.0, device=device)
                    if cfg.kl_beta > 0:
                        raw_kl = 0.5 * (mu_batch.pow(2) + logvar_batch.exp() - 1 - logvar_batch)
                        kl_beta_loss = raw_kl.sum(dim=-1).mean()

                    loss = (
                        ce_loss + kl_scale * kl_loss
                        + cfg.z_var_weight * z_var_loss
                        + cfg.dip_weight * dip_loss
                        + cfg.contrastive_weight * cl_loss
                        + cfg.kl_beta * kl_beta_loss
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
                    last_grad_norm = nn.utils.clip_grad_norm_(
                        all_params, max_norm=5.0
                    ).item()
                    # Skip step on inf/nan gradients and reset Adam momentum
                    # to prevent contamination from the unscaled inf values.
                    if math.isfinite(last_grad_norm):
                        scaler.step(optimizer)
                    else:
                        # Detailed diagnostics on first bad gnorm per epoch
                        if not getattr(self, "_gnorm_diagnosed", False):
                            self._gnorm_diagnosed = True
                            with torch.no_grad():
                                _h = modules["enc_to_latent"](
                                    torch.zeros(1, cfg.decoder_hidden_dim, device=device)
                                )
                                _mu_dbg, _lv_dbg = _h.chunk(2, dim=-1)
                                _has_nan_params = any(
                                    torch.isnan(p).any().item()
                                    for m in modules.values()
                                    if isinstance(m, nn.Module)
                                    for p in m.parameters()
                                )
                            logger.warning(
                                "GNORM DIAGNOSTIC step=%d: "
                                "ce=%.4f z_var_loss=%.6f dip=%.6f loss=%.4f "
                                "logvar_range=[%.2f, %.2f] logvar_mean=%.2f "
                                "z_std_running=%.4f z_batch_std=%.4f "
                                "scaler_scale=%.1f nan_in_params=%s",
                                global_step,
                                ce_loss.item(), z_var_loss.item(),
                                dip_loss.item(), loss.item() * accum,
                                kl_per_dim_train.min().item() if _do_kl else -999,
                                kl_per_dim_train.max().item() if _do_kl else -999,
                                kl_per_dim_train.mean().item() if _do_kl else -999,
                                z_running_std.mean().item(),
                                z_batch.std(dim=0).mean().item(),
                                scaler.get_scale(),
                                _has_nan_params,
                            )
                        logger.warning(
                            "Skipping optimizer step: gnorm=%s at step %d "
                            "— resetting momentum",
                            last_grad_norm,
                            global_step,
                        )
                        # Reset Adam momentum buffers to prevent cascade
                        for group in optimizer.param_groups:
                            for p in group["params"]:
                                state = optimizer.state.get(p)
                                if state:
                                    state["exp_avg"].zero_()
                                    state["exp_avg_sq"].zero_()
                                    if "max_exp_avg_sq" in state:
                                        state["max_exp_avg_sq"].zero_()
                    scaler.update()
                    optimizer.zero_grad()
                    global_step += 1

                train_ce_sum += ce_loss.item() * b
                train_kl_sum += kl_loss.item() * b
                train_zvar_sum += z_var_loss.item() * b
                train_dip_sum += dip_loss.item() * b
                train_cl_sum += cl_loss.item() * b
                train_klb_sum += kl_beta_loss.item() * b
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
                    if cfg.z_var_weight > 0:
                        extra_parts.append(f"zvar={z_var_loss.item():.4f}")
                    if cfg.dip_weight > 0:
                        extra_parts.append(f"dip={dip_loss.item():.6f}")
                    if _use_contrastive:
                        extra_parts.append(f"CL={cl_loss.item():.4f}")
                    if cfg.kl_beta > 0:
                        extra_parts.append(f"KLβ={kl_beta_loss.item():.4f}")
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
            train_zvar = train_zvar_sum / max(train_count, 1)
            train_dip = train_dip_sum / max(train_count, 1)
            train_cl = train_cl_sum / max(train_count, 1)
            train_klb = train_klb_sum / max(train_count, 1)
            train_loss = (
                train_ce + cfg.kl_weight * train_kl
                + cfg.z_var_weight * train_zvar + cfg.dip_weight * train_dip
                + cfg.contrastive_weight * train_cl + cfg.kl_beta * train_klb
            )

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
                        ce_loss, kl_loss, kl_per_dim, _, _, _, _ = _vae_forward(
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
            if cfg.z_var_weight > 0:
                epoch_parts.append(f"zvar={train_zvar:.4f}")
            if cfg.dip_weight > 0:
                epoch_parts.append(f"dip={train_dip:.6f}")
            if _use_contrastive:
                epoch_parts.append(f"CL={train_cl:.4f}")
            if cfg.kl_beta > 0:
                epoch_parts.append(f"KLβ={train_klb:.4f}")
            epoch_parts.append(f"total={train_loss:.4f}")
            epoch_parts.append(f"| val: CE={val_ce:.4f}")
            if _do_kl:
                epoch_parts.append(f"KL={val_kl:.4f}")
            epoch_parts.append(f"total={val_loss:.4f}")
            # Log z distribution health + LR
            current_lr = scheduler.get_last_lr()[0]
            epoch_parts.append(
                f"| z_std={z_running_std.mean():.4f}"
                f" z_active={int((z_running_std > 0.01).sum())}/{cfg.latent_dim}"
                f" lr={current_lr:.6f}"
            )
            if ss_p > 0:
                epoch_parts.append(f"ss={ss_p:.2f}")
            if wd_p > 0:
                epoch_parts.append(f"wd={wd_p:.2f}")
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
                    mem = modules["latent_to_decoder"](z).unsqueeze(1)
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
                    # Decode single-sentence output: truncate at first EOS
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
                        # Strip trailing orphan consonants
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

                # Find language-specific samples from the validation set.
                # Use the full dataset indices (val_dataset.indices) to look
                # up language labels.  Pick 1 English + 1 non-English for
                # interpretable diagnostics.
                _val_indices = list(val_dataset.indices)
                _eng_idx = _non_eng_idx = None
                _lang_labels = ["?", "?"]
                for _vi in _val_indices:
                    _lang = languages_list[_vi] if _vi < len(languages_list) else "?"
                    if _lang == "eng" and _eng_idx is None:
                        _eng_idx = _vi
                    elif _lang != "eng" and _non_eng_idx is None:
                        _non_eng_idx = _lang
                        _non_eng_dataset_idx = _vi
                    if _eng_idx is not None and _non_eng_idx is not None:
                        break

                # Build a small batch of the selected samples
                _sample_indices = []
                if _eng_idx is not None:
                    _sample_indices.append(_eng_idx)
                    _lang_labels[0] = "eng"
                if _non_eng_idx is not None:
                    _sample_indices.append(_non_eng_dataset_idx)
                    _lang_labels[1] = _non_eng_idx
                # Fallback: just use first 2 from val loader
                if len(_sample_indices) < 2:
                    val_batch_tokens, val_batch_lengths = next(iter(val_loader))
                    val_batch_tokens = val_batch_tokens[:2].to(device)
                    val_batch_lengths = torch.as_tensor(
                        val_batch_lengths[:2], device=device
                    )
                else:
                    _toks = []
                    _lens = []
                    for _si in _sample_indices:
                        _t, _l = dataset[_si]
                        _toks.append(_t)
                        _lens.append(_l)
                    val_batch_tokens = torch.stack(_toks).to(device)
                    val_batch_lengths = torch.tensor(_lens, device=device)

                # --- 1. Reconstruction ---
                z_real = _encode_text(val_batch_tokens, val_batch_lengths)
                recon_texts = _sample_decode(z_real)
                for j in range(min(2, len(_lang_labels))):
                    orig_ids = val_batch_tokens[j].cpu().tolist()
                    orig_ids = [x for x in orig_ids[:val_batch_lengths[j]] if x < vocab_size]
                    orig = sp.decode(orig_ids)
                    logger.info(
                        "  recon[%s] orig: %s", _lang_labels[j], orig[:120]
                    )
                    logger.info(
                        "  recon[%s]  dec: %s", _lang_labels[j], recon_texts[j][:120]
                    )

                # --- 2. Interpolation (English ↔ non-English) ---
                alphas = [0.0, 0.25, 0.5, 0.75, 1.0]
                z_interp = torch.stack(
                    [z_real[0] * (1 - a) + z_real[1] * a for a in alphas]
                )
                interp_texts = _sample_decode(z_interp)
                logger.info(
                    "  interp: %s → %s", _lang_labels[0], _lang_labels[1]
                )
                for k, (a, txt) in enumerate(zip(alphas, interp_texts)):
                    logger.info("  interp[%.2f]: %s", a, txt[:120])

                # --- 3. Perturbation (around English sentence) ---
                noise_scales = [0.0, 0.1, 0.5, 1.0]
                z_perturbed = torch.stack(
                    [z_real[0] + s * z_running_std * torch.randn_like(z_real[0])
                     for s in noise_scales]
                )
                perturb_texts = _sample_decode(z_perturbed)
                logger.info("  perturb: around %s sentence", _lang_labels[0])
                for k, (s, txt) in enumerate(zip(noise_scales, perturb_texts)):
                    logger.info("  perturb[σ=%.1f]: %s", s, txt[:120])

                # --- 4. Random z near English cluster ---
                # Sample from the encoder distribution, then also sample
                # near the English z for interpretable comparison.
                z_random = (
                    torch.randn(3, cfg.latent_dim, device=device)
                    * z_running_std + z_running_mean
                )
                # 2 more near the English z (small perturbation)
                z_near_eng = torch.stack([
                    z_real[0] + 0.3 * z_running_std * torch.randn_like(z_real[0]),
                    z_real[0] + 0.3 * z_running_std * torch.randn_like(z_real[0]),
                ])
                z_all_random = torch.cat([z_random, z_near_eng], dim=0)
                random_texts = _sample_decode(z_all_random)
                for j in range(3):
                    logger.info("  random[%d]: %s", j, random_texts[j][:120])
                for j in range(3, 5):
                    logger.info(
                        "  near_eng[%d]: %s", j - 3, random_texts[j][:120]
                    )

                # --- 5. Length distribution (autoregressive decode) ---
                # Decode a batch of z sampled from the encoder distribution
                # to check EOS behavior and length variation.
                z_length_test = (
                    torch.randn(32, cfg.latent_dim, device=device)
                    * z_running_std + z_running_mean
                )
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
                # EOS rate: fraction of length-test decodes that emitted
                # EOS before max_seq_len (measures stopping ability)
                eos_rate = sum(
                    1 for t in length_texts if len(t.split()) < cfg.max_seq_len // 2
                ) / max(len(length_texts), 1)
                logger.info(
                    "  struct: TTR=%.3f rep_rate=%.3f "
                    "mean_word_len=%.1f eos_rate=%.2f",
                    m["ttr"],
                    m["rep_rate"],
                    m["mean_word_len"],
                    eos_rate,
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
                        # Latent calibration statistics — used at agent
                        # time to keep projected z in-distribution.
                        "z_mean": z_running_mean.cpu(),
                        "z_std": z_running_std.cpu(),
                        "train_loss": train_loss,
                        "val_loss": val_loss,
                        "spm_hash": _file_hash(spm_path),
                    },
                    output_path,
                )
                logger.info("Saved best decoder checkpoint to %s", output_path)

            # -- Early termination checks --
            # 1. Gradient explosion: val CE suddenly jumps (>2x best)
            if val_loss > best_val_loss * 2.0 and epoch > 5:
                logger.warning(
                    "EARLY STOP: val_loss=%.4f is >2x best=%.4f — "
                    "gradient explosion detected. Best checkpoint preserved.",
                    val_loss,
                    best_val_loss,
                )
                break

            # 2. Latent collapse: z_std drops below absolute floor
            z_std_mean = z_running_std.mean().item()
            if epoch > 5 and z_std_mean < 0.02:
                logger.warning(
                    "EARLY STOP: z_std=%.4f below 0.02 — latent space "
                    "collapse. Best checkpoint preserved.",
                    z_std_mean,
                )
                break

            # Step LR scheduler
            scheduler.step()

            # Update training history
            _shutdown_state["epoch"] = epoch + 1
            _shutdown_state["best_val_loss"] = best_val_loss
            history.update_epoch(epoch + 1, best_val_loss)

            # Save full training state for resume (every epoch)
            resume_path = Path(output_dir) / "vae_resume.pt"
            torch.save(
                {
                    "epoch": epoch + 1,
                    "global_step": global_step,
                    "best_val_loss": best_val_loss,
                    "spm_hash": _file_hash(spm_path),
                    "z_mean": z_running_mean.cpu(),
                    "z_std": z_running_std.cpu(),
                    "modules": {
                        k: m.state_dict()
                        for k, m in modules.items()
                        if isinstance(m, nn.Module)
                    },
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "scaler": scaler.state_dict(),
                },
                resume_path,
            )

        _shutdown_state["epoch"] = epoch + 1 if epoch >= start_epoch else start_epoch
        _shutdown_state["best_val_loss"] = best_val_loss
        _end_session_once()
        atexit.unregister(_end_session_once)
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
            "latent_to_decoder": latent_to_decoder,
            "dec_token_embedding": dec_token_embedding,
            "dec_pos_embedding": dec_pos_embedding,
            "decoder": decoder,
            "output_head": output_head,
            "_rope_freqs": rope_freqs,
            "_cached_mask": cached_mask,
            "_cfg": cfg,
            "_attn_pool_query": (
                nn.Parameter(torch.randn(1, 1, hidden) * 0.01).to(device)
                if cfg.encoder_pooling == "attention" else None
            ),
        }


def pretrain_vae_decoder(config: VAEPretrainConfig) -> dict[str, float]:
    """Convenience function: create ``VAEPretrainer`` and run.

    Args:
        config: Pretraining configuration.

    Returns:
        Metrics dict with ``train_loss``, ``val_loss``,
        ``active_latent_dims``, and ``num_samples``.
    """
    # Ensure logging is configured (no-op if already set up by caller).
    # force=False avoids overriding existing handlers; the handler flushes
    # after every record so output appears immediately when piped.
    if not logging.root.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(
            logging.Formatter("%(asctime)s %(name)s %(message)s")
        )
        handler.setLevel(logging.INFO)
        logging.root.addHandler(handler)
        logging.root.setLevel(logging.INFO)
    return VAEPretrainer(config).pretrain()

"""VAE forward pass and free-run decode (shared between train and val)."""

from __future__ import annotations

import torch
import torch.nn.functional as F  # noqa: N812
from torch import Tensor, nn


def _unlikelihood_ngram_loss(
    logits: Tensor,
    targets: Tensor,
    mask: Tensor,
    window: int = 4,
) -> Tensor:
    """Welleck-style unlikelihood loss for unigram reduplication.

    For each decoder position ``t``, penalize assigning high probability
    to any token that appeared in the previous ``window`` target
    positions — unless that token IS the true next target (in which case
    the repetition is legitimate and penalizing it would be wrong).

    Directly targets the ``nd nd nd`` degenerate-tail failure mode: a
    decoder that wants to repeat its last token gets pushed away from
    that choice, while EOS (a unique terminator that doesn't match any
    recent content token) is unaffected.

    Args:
        logits: ``(B, S, V)`` decoder logits.
        targets: ``(B, S)`` ground-truth token ids.
        mask: ``(B, S)`` boolean mask of valid positions.
        window: How many previous positions to treat as "recent".

    Returns:
        Scalar unlikelihood loss averaged over (position × offset) pairs
        that are both masked-valid and non-self-repeating in the target.
    """
    B, S, V = logits.shape
    # logsumexp per position: avoids materializing full (B, S, V) log_probs
    # tensor (which for v9.5 with V=30K is a ~2.6 GB float32 plus gradient
    # duplicate — blows past 24GB VRAM).  Compute log P(token) on-demand
    # via gather(logits) - logsumexp(logits).
    lse = torch.logsumexp(logits, dim=-1)              # (B, S)
    total = logits.new_tensor(0.0)
    denom = logits.new_tensor(0.0)
    for offset in range(1, window + 1):
        if offset >= S:
            break
        prev_tok = targets[:, :-offset]                # (B, S-offset)
        logit_prev = logits[:, offset:, :].gather(
            -1, prev_tok.unsqueeze(-1),
        ).squeeze(-1)                                  # (B, S-offset)
        logp_prev = logit_prev - lse[:, offset:]       # (B, S-offset)
        prob_prev = logp_prev.exp().clamp(max=0.999999)
        ul = -(1.0 - prob_prev).log()                  # (B, S-offset)
        true_tgt = targets[:, offset:]                 # (B, S-offset)
        applicable = (
            mask[:, offset:].float() * (prev_tok != true_tgt).float()
        )
        total = total + (ul * applicable).sum()
        denom = denom + applicable.sum()
    return total / denom.clamp(min=1.0)


def _info_nce_loss(
    z: Tensor,
    embeddings: Tensor,
    temperature: float = 0.07,
    projection: nn.Module | None = None,
) -> Tensor:
    """InfoNCE contrastive loss between z vectors and source embeddings.

    Pulls z vectors of semantically similar sentences together and
    pushes dissimilar ones apart.  Similarity is defined by the
    pre-computed sentence-transformer embeddings.

    When z and embeddings have different dimensions, ``projection``
    maps z into the embedding space.

    Args:
        z: Latent codes ``(batch, latent_dim)``.
        embeddings: Pre-computed source text embeddings ``(batch, embed_dim)``.
        temperature: Softmax temperature (lower = sharper).
        projection: Optional linear layer to project z → embed_dim.

    Returns:
        Scalar InfoNCE loss.
    """
    z_proj = projection(z) if projection is not None else z

    # Normalize both to unit vectors
    z_norm = F.normalize(z_proj.float(), dim=-1)
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
    _phonetic_sim: Tensor | None = None,
    _phonetic_smoothing: float = 0.0,
    _residual_vq: nn.Module | None = None,
    encoder_tokens: Tensor | None = None,
    encoder_lengths: Tensor | None = None,
    _tag_open_ids: Tensor | None = None,
    _tag_close_ids: Tensor | None = None,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor | None, Tensor, Tensor]:
    """Run one VAE forward pass with optional scheduled sampling.

    When ``scheduled_sampling_p > 0``, a two-pass approach is used:
    first a full teacher-forced pass to get predictions, then a second
    pass with a mixed input where each position uses the model's own
    argmax prediction with probability ``scheduled_sampling_p``, or the
    ground truth otherwise.  This teaches the decoder to recover from
    its own outputs, preventing degeneration during free-run generation
    at long sequence lengths.

    Returns:
        Tuple of ``(ce_loss, kl_loss, kl_per_dim, z, logits, mu, logvar, vq_loss, ...)``.
    """
    from lfm.generator.layers import PhraseDecoder, multiscale_causal_mask

    device = batch_tokens.device
    b, dec_seq_len = batch_tokens.shape

    # Phase 2: encoder may see different tokens (parent sentence) than
    # the decoder target (constituent).  When encoder_tokens is None,
    # encoder and decoder use the same tokens (phase 1 / standard VAE).
    if encoder_tokens is not None:
        enc_toks = encoder_tokens
        enc_lens = encoder_lengths
        _, enc_seq_len = enc_toks.shape
    else:
        enc_toks = batch_tokens
        enc_lens = batch_lengths
        enc_seq_len = dec_seq_len
    seq_len = dec_seq_len  # decoder sequence length for masks/positions

    # Encode
    enc_src_mask = (
        torch.arange(enc_seq_len, device=device).unsqueeze(0)
        < enc_lens.unsqueeze(1)
    )
    enc_pos_ids = torch.arange(enc_seq_len, device=device).unsqueeze(0)
    enc_input = enc_token_embedding(enc_toks) + enc_pos_embedding(enc_pos_ids)
    enc_out = encoder(enc_input, src_key_padding_mask=~enc_src_mask)

    # Pool encoder outputs to a single vector
    if _cfg is not None and getattr(_cfg, "encoder_pooling", "mean") == "attention":
        # Attention pooling: learned query attends to encoder outputs
        attn_query = _attn_pool_query  # (1, 1, hidden)
        attn_weights = torch.bmm(
            attn_query.expand(b, -1, -1), enc_out.transpose(1, 2)
        )  # (b, 1, enc_seq_len)
        attn_weights = attn_weights.masked_fill(~enc_src_mask.unsqueeze(1), float("-inf"))
        attn_weights = F.softmax(attn_weights, dim=-1)
        pooled = torch.bmm(attn_weights, enc_out).squeeze(1)  # (b, hidden)
    else:
        # Mean pooling (default)
        enc_masked = enc_out * enc_src_mask.unsqueeze(-1).float()
        denom = enc_lens.unsqueeze(-1).float().clamp(min=1)
        pooled = enc_masked.sum(dim=1) / denom

    # Decoder mask (for CE loss — uses decoder sequence lengths)
    src_mask = (
        torch.arange(dec_seq_len, device=device).unsqueeze(0)
        < batch_lengths.unsqueeze(1)
    )
    pos_ids = torch.arange(dec_seq_len, device=device).unsqueeze(0)

    # Latent
    if _residual_vq is not None:
        # VQ-VAE path: deterministic encoding → discrete quantization
        z_continuous = enc_to_latent(pooled)  # (B, latent_dim)
        z, vq_commitment_loss, _vq_indices = _residual_vq(z_continuous)
        # Noise augmentation: train decoder to handle continuous
        # perturbations around codebook entries.  At agent time, the
        # continuous residual from _input_proj stays within this envelope.
        if _cfg is not None and getattr(_cfg, "vq_noise_sigma", 0) > 0:
            if enc_token_embedding.training:
                z = z + torch.randn_like(z) * _cfg.vq_noise_sigma
        mu = z_continuous
        logvar = torch.zeros_like(z_continuous)
    else:
        # Standard VAE path
        h = enc_to_latent(pooled)
        mu, logvar = h.chunk(2, dim=-1)
        std = (0.5 * logvar).exp()
        z = mu + std * torch.randn_like(std)
        # Denoising-VAE augmentation: add extra gaussian noise to z
        # during training so the decoder learns to produce valid
        # reconstructions from z values slightly off the encoder's
        # posterior manifold.  Disabled when z_noise_sigma <= 0
        # (v7/v12 default).
        if _cfg is not None and getattr(_cfg, "z_noise_sigma", 0.0) > 0:
            if enc_token_embedding.training:
                z = z + torch.randn_like(z) * _cfg.z_noise_sigma
        vq_commitment_loss = None

    # Decode — reshape into K memory tokens for multi-token z injection
    _n_mem = getattr(_cfg, "num_memory_tokens", 1) if _cfg is not None else 1
    memory = latent_to_decoder(z).reshape(b, _n_mem, -1)

    bos_col = torch.full((b, 1), bos_id, dtype=torch.long, device=device)
    teacher_input_ids = torch.cat([bos_col, batch_tokens[:, :-1]], dim=1)

    # Precompute mask
    if isinstance(decoder, PhraseDecoder):
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
        if isinstance(decoder, PhraseDecoder):
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

    # Reconstruction loss (masked CE) — always against ground truth targets.
    # With phonetic label smoothing, the target distribution blends one-hot
    # with phonetic similarity — near-miss predictions (e.g. /b/ for /p/)
    # incur less loss than distant predictions (e.g. /ʒ/ for /p/).
    flat_logits = logits.reshape(-1, full_vocab)
    flat_targets = batch_tokens.reshape(-1)

    if _phonetic_sim is not None and _phonetic_smoothing > 0:
        # Standard CE + phonetic smoothing correction (memory-efficient).
        # The smoothing term uses detached log_probs — it modifies the
        # target distribution, not the gradient path through the model.
        ce_hard = F.cross_entropy(flat_logits, flat_targets, reduction="none")
        with torch.no_grad():
            log_probs_d = F.log_softmax(flat_logits.detach(), dim=-1)
            _chunk = 4096
            soft_ce = torch.zeros_like(ce_hard)
            for _start in range(0, len(flat_targets), _chunk):
                _end = min(_start + _chunk, len(flat_targets))
                _tgt = flat_targets[_start:_end]
                _sim_rows = _phonetic_sim[_tgt.cpu()].to(flat_logits.device)
                soft_ce[_start:_end] = -(_sim_rows * log_probs_d[_start:_end]).sum(dim=-1)
        ce = (
            (1 - _phonetic_smoothing) * ce_hard + _phonetic_smoothing * soft_ce
        ).reshape(b, seq_len)
    else:
        ce = F.cross_entropy(flat_logits, flat_targets, reduction="none").reshape(b, seq_len)

    ce_loss = (ce * src_mask.float()).sum() / src_mask.float().sum().clamp(min=1)

    # Unlikelihood regularization against unigram reduplication (Welleck 2020).
    # Cheap teacher-forced loss that suppresses the decoder's tendency to
    # fall into `tok tok tok ...` local attractors when content is
    # exhausted.  Additive to CE, weight-controlled; EOS is automatically
    # safe since it's a unique terminator that doesn't appear in recent
    # content windows.
    _ul_weight = getattr(_cfg, "unlikelihood_weight", 0.0) if _cfg is not None else 0.0
    if _ul_weight > 0:
        _ul_window = getattr(_cfg, "unlikelihood_window", 4)
        ul_loss = _unlikelihood_ngram_loss(
            logits, batch_tokens, src_mask, window=_ul_window,
        )
        ce_loss = ce_loss + _ul_weight * ul_loss

    # Bag-of-words loss: order-invariant token presence check.
    # Mean of per-position log-softmax gives a pooled prediction over
    # the vocabulary, compared against target token frequencies.
    if _cfg is not None and getattr(_cfg, "bow_weight", 0) > 0:
        # Mean-pool logits across valid positions → (b, V)
        # Uses the existing src_mask; no extra memory allocation.
        mask_f = src_mask.unsqueeze(-1).float()  # (b, S, 1)
        bow_logits = (logits * mask_f).sum(dim=1) / mask_f.sum(dim=1).clamp(min=1)
        # Target: token frequency distribution per sample
        bow_target = torch.zeros(b, full_vocab, device=device)
        for bi in range(b):
            valid_toks = batch_tokens[bi, :batch_lengths[bi]]
            bow_target[bi].scatter_add_(
                0, valid_toks, torch.ones_like(valid_toks, dtype=torch.float),
            )
        bow_target = bow_target / bow_target.sum(dim=-1, keepdim=True).clamp(min=1)
        bow_loss = F.cross_entropy(bow_logits, bow_target)
    else:
        bow_loss = torch.tensor(0.0, device=device)

    # KL loss (skipped in VQ mode or when kl_weight=0)
    if _residual_vq is not None:
        kl_per_dim = torch.zeros(b, mu.size(-1), device=device)
        kl_loss = torch.tensor(0.0, device=device)
    elif compute_kl:
        kl_per_dim = 0.5 * (mu.pow(2) + logvar.exp() - 1 - logvar)
        if kl_free_bits > 0:
            kl_per_dim = torch.clamp(kl_per_dim, min=kl_free_bits)
        kl_loss = kl_per_dim.sum(dim=-1).mean()
    else:
        kl_per_dim = torch.zeros(b, mu.size(-1), device=device)
        kl_loss = torch.tensor(0.0, device=device)

    # Tag-balance auxiliary loss: penalize mismatched open/close tag
    # expected counts over the generated distribution.  Each sample's
    # predicted softmax mass over open-tag IDs should sum to roughly
    # the same as its mass over close-tag IDs (every `<NP>` needs a
    # matching `</NP>`).  Off-by-default (tag_balance_weight=0).
    tag_balance_weight = (
        getattr(_cfg, "tag_balance_weight", 0.0) if _cfg is not None else 0.0
    )
    if (
        tag_balance_weight > 0
        and _tag_open_ids is not None
        and _tag_close_ids is not None
        and _tag_open_ids.numel() > 0
    ):
        probs = F.softmax(logits, dim=-1)  # (B, S, V)
        mask_f = src_mask.unsqueeze(-1).float()  # (B, S, 1)
        probs_masked = probs * mask_f
        p_open = probs_masked.index_select(-1, _tag_open_ids).sum(dim=(1, 2))
        p_close = probs_masked.index_select(-1, _tag_close_ids).sum(dim=(1, 2))
        tag_balance_loss = (p_open - p_close).pow(2).mean()
    else:
        tag_balance_loss = torch.tensor(0.0, device=device)

    return (
        ce_loss, kl_loss, kl_per_dim, z, logits, mu, logvar,
        vq_commitment_loss, bow_loss, tag_balance_loss,
    )


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
    num_memory_tokens: int = 1,
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

    memory = latent_to_decoder(z).reshape(z.size(0), num_memory_tokens, -1)
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

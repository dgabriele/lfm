"""VAE model construction for pretraining."""

from __future__ import annotations

import torch
from torch import nn

from .config import VAEPretrainConfig


def build_model(
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

    # Decoder (linguistic architecture — must match GeneratorConfig)
    from lfm.generator.layers import (
        LinguisticDecoder,
        multiscale_causal_mask,
        precompute_rope_freqs,
    )

    # Encoder → latent: output dim depends on VAE mode
    if cfg.use_vq:
        enc_to_latent = nn.Linear(hidden, cfg.latent_dim).to(device)

        if cfg.vq_mode == "grouped":
            from lfm.generator.quantize import GroupedVQ

            residual_vq: nn.Module | None = GroupedVQ(
                num_groups=cfg.vq_num_groups,
                codebook_size=cfg.vq_codebook_size,
                embedding_dim=cfg.latent_dim,
                commitment_weight=cfg.vq_commitment_weight,
                entropy_weight=cfg.vq_entropy_weight,
                balance_weight=cfg.vq_balance_weight,
                orthogonality_weight=cfg.vq_orthogonality_weight,
                ema_update=cfg.vq_ema_update,
                ema_decay=cfg.vq_decay,
            ).to(device)
        else:
            from lfm.generator.quantize import ResidualVQ

            residual_vq = ResidualVQ(
                num_levels=cfg.vq_num_levels,
                codebook_size=cfg.vq_codebook_size,
                embedding_dim=cfg.latent_dim,
                commitment_weight=cfg.vq_commitment_weight,
                entropy_weight=cfg.vq_entropy_weight,
                ema_update=cfg.vq_ema_update,
                ema_decay=cfg.vq_decay,
            ).to(device)
    else:
        enc_to_latent = nn.Linear(hidden, cfg.latent_dim * 2).to(device)
        residual_vq = None

    _n_mem = getattr(cfg, "num_memory_tokens", 1)
    latent_to_decoder = nn.Linear(cfg.latent_dim, _n_mem * hidden).to(device)
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
            head_dim, cfg.max_seq_len + 1, device=device
        )

    # Precompute multi-scale causal mask (+1 for BOS in KV-cached decode)
    cached_mask = multiscale_causal_mask(
        cfg.max_seq_len + 1,
        num_heads=cfg.decoder_num_heads,
        head_windows=cfg.attention_head_windows,
        global_every=cfg.attention_global_every,
        device=device,
    )

    # Length embedding for variable-length EOS control
    length_proj = None
    if getattr(cfg, "use_length_embedding", False):
        length_proj = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
        ).to(device)
        # Initialize near zero so length embedding starts as a no-op
        with torch.no_grad():
            length_proj[-1].weight.mul_(0.01)
            length_proj[-1].bias.zero_()

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
        "_residual_vq": residual_vq,
        "_length_proj": length_proj,
        "_attn_pool_query": (
            nn.Parameter(torch.randn(1, 1, hidden) * 0.01).to(device)
            if cfg.encoder_pooling == "attention" else None
        ),
    }

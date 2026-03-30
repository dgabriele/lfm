"""Differentiable decoder re-run for agent games."""

from __future__ import annotations

from torch import Tensor

from lfm.generator.layers import multiscale_causal_mask


def rerun_decoder_with_grad(
    gen,
    z: Tensor,
    tokens: Tensor,
    mask: Tensor,
) -> Tensor:
    """Re-run the frozen decoder on generated tokens WITH gradients.

    Phase 2 of the two-phase forward: takes the token sequence from
    phase 1 (no_grad generation) and runs it through the decoder in
    one parallel pass.  Gradients flow through the decoder's cross-
    attention to memory, back to z and ``_input_proj``.

    The token embeddings are detached (integer lookup) — only the
    cross-attention path to memory carries gradient signal.

    Args:
        gen: The ``MultilingualVAEGenerator`` (frozen decoder).
        z: Latent codes ``(batch, latent_dim)`` WITH gradient.
        tokens: Generated token IDs ``(batch, seq_len)`` from phase 1.
        mask: Boolean mask ``(batch, seq_len)``.

    Returns:
        Decoder hidden states ``(batch, seq_len, hidden_dim)`` with gradients.
    """
    # Calibrate / quantize z (same path as generation)
    if gen._vq_codebook is not None:
        z_dec = gen._quantize_z(z)
    elif gen._z_stats_initialized:
        z_dec = gen.calibrate_z(z)
    else:
        z_dec = z

    # z → memory (differentiable through frozen linear)
    memory = gen.latent_to_decoder(z_dec).reshape(
        z.size(0), gen._num_memory_tokens, -1,
    )

    # Embed the generated tokens (detached — no grad through token chain)
    tok_emb = gen.token_embedding(tokens)

    # Full causal mask for the sequence
    seq_len = tokens.size(1)
    causal_mask = multiscale_causal_mask(
        seq_len,
        num_heads=gen.config.decoder_num_heads,
        head_windows=gen.config.attention_head_windows,
        global_every=gen.config.attention_global_every,
        device=z.device,
    )

    # Run decoder in one parallel pass — cross-attention to memory has grad
    rope = gen._rope_freqs[:seq_len] if gen._rope_freqs is not None else None
    return gen.decoder(
        tok_emb, memory, tgt_mask=causal_mask, rope_freqs=rope,
    )

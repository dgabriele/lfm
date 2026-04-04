"""Decoder utilities for agent games: multi-segment decode and gradient re-run."""

from __future__ import annotations

import torch
from torch import nn
from torch import Tensor

from lfm.generator.layers import multiscale_causal_mask


def _calibrate_or_quantize(gen, z: Tensor) -> Tensor:
    """Apply the generator's z calibration/quantization path."""
    if gen._vq_codebook is not None:
        return gen._quantize_z(z)
    if gen._z_stats_initialized:
        return gen.calibrate_z(z)
    return z


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
    z_dec = _calibrate_or_quantize(gen, z)

    memory = gen.latent_to_decoder(z_dec).reshape(
        z.size(0), gen._num_memory_tokens, -1,
    )

    tok_emb = gen.token_embedding(tokens)

    seq_len = tokens.size(1)
    causal_mask = multiscale_causal_mask(
        seq_len,
        num_heads=gen.config.decoder_num_heads,
        head_windows=gen.config.attention_head_windows,
        global_every=gen.config.attention_global_every,
        device=z.device,
    )

    rope = gen._rope_freqs[:seq_len] if gen._rope_freqs is not None else None
    return gen.decoder(
        tok_emb, memory, tgt_mask=causal_mask, rope_freqs=rope,
    )


def rerun_decoder_multiseg_with_grad(
    gen,
    z_sequence: Tensor,
    z_weights: Tensor,
    tokens: Tensor,
    mask: Tensor,
    segment_boundaries: Tensor,
) -> Tensor:
    """Re-run decoder with gradients for multi-segment z-switching.

    Each position cross-attends only to the memory from the z that
    generated it.  All segment memories are concatenated and a
    cross-attention mask restricts each position to its segment's
    memory tokens.

    Args:
        gen: The ``MultilingualVAEGenerator`` (frozen decoder).
        z_sequence: ``(batch, max_segments, latent_dim)`` with gradients.
        z_weights: ``(batch, max_segments)`` ACT weights per segment.
        tokens: ``(batch, seq_len)`` from phase 1.
        mask: ``(batch, seq_len)`` boolean mask.
        segment_boundaries: ``(batch, max_segments)`` start position of
            each segment (0 for unused segments).

    Returns:
        Decoder hidden states ``(batch, seq_len, hidden_dim)`` with gradients.
    """
    batch, max_segs, latent_dim = z_sequence.shape
    # Trim to actual max length (avoids exceeding precomputed RoPE)
    actual_max = int(mask.float().sum(dim=1).max().item())
    tokens = tokens[:, :actual_max]
    mask = mask[:, :actual_max]
    segment_boundaries = segment_boundaries.clamp(max=actual_max)
    seq_len = actual_max
    num_mem = gen._num_memory_tokens
    nhead = gen.config.decoder_num_heads
    device = tokens.device

    # Weight each z by its ACT weight, then calibrate/quantize
    weighted_z = z_weights.unsqueeze(-1) * z_sequence  # (B, K, latent)
    z_flat = weighted_z.reshape(batch * max_segs, latent_dim)
    z_dec = _calibrate_or_quantize(gen, z_flat)

    # All memories concatenated: (B, K * num_mem, hidden)
    all_memory = gen.latent_to_decoder(z_dec).reshape(
        batch, max_segs * num_mem, -1,
    )

    # Segment assignment: which segment generated each position
    seg_idx = _compute_segment_assignment(
        segment_boundaries, seq_len, max_segs, device,
    )

    # Cross-attention mask: each position only attends to its segment's memory
    xattn_mask = _build_xattn_mask(seg_idx, max_segs, num_mem, nhead, device)

    tok_emb = gen.token_embedding(tokens)

    causal_mask = multiscale_causal_mask(
        seq_len,
        num_heads=nhead,
        head_windows=gen.config.attention_head_windows,
        global_every=gen.config.attention_global_every,
        device=device,
    )

    rope = None
    if gen._rope_freqs is not None:
        if gen._rope_freqs.size(0) < seq_len:
            from lfm.generator.layers import precompute_rope_freqs
            rope = precompute_rope_freqs(
                gen.config.decoder_hidden_dim // gen.config.decoder_num_heads,
                seq_len, device=device,
            )
        else:
            rope = gen._rope_freqs[:seq_len]

    return gen.decoder(
        tok_emb, all_memory,
        tgt_mask=causal_mask, rope_freqs=rope, xattn_mask=xattn_mask,
    )


def _compute_segment_assignment(
    segment_boundaries: Tensor,
    seq_len: int,
    max_segs: int,
    device: torch.device,
) -> Tensor:
    """Map each position to its segment index.

    Args:
        segment_boundaries: ``(batch, max_segments)`` start positions.
        seq_len: Total sequence length.
        max_segs: Maximum number of segments.
        device: Torch device.

    Returns:
        ``(batch, seq_len)`` segment index per position.
    """
    batch = segment_boundaries.size(0)
    positions = torch.arange(seq_len, device=device).unsqueeze(0).expand(batch, -1)

    # searchsorted: find which bin each position falls into
    # boundaries are sorted; searchsorted returns the index of the first
    # boundary > position, so subtract 1 to get the segment index
    seg_idx = torch.searchsorted(
        segment_boundaries.contiguous(), positions.contiguous(), right=True,
    ) - 1
    return seg_idx.clamp(0, max_segs - 1)


def _build_xattn_mask(
    seg_idx: Tensor,
    max_segs: int,
    num_mem: int,
    nhead: int,
    device: torch.device,
) -> Tensor:
    """Build cross-attention mask for position-dependent memory.

    Each position can only attend to the memory tokens belonging to
    its segment.  All other memory positions are masked with ``-inf``.

    Args:
        seg_idx: ``(batch, seq_len)`` segment index per position.
        max_segs: Maximum number of segments.
        num_mem: Memory tokens per segment.
        nhead: Number of attention heads.
        device: Torch device.

    Returns:
        ``(batch * nhead, seq_len, max_segs * num_mem)`` float mask.
    """
    batch, seq_len = seg_idx.shape
    total_mem = max_segs * num_mem

    # Memory token → segment mapping: (total_mem,)
    mem_seg = torch.arange(max_segs, device=device).repeat_interleave(num_mem)

    # Allowed: seg_idx[b, t] == mem_seg[m]
    # seg_idx: (B, S, 1), mem_seg: (1, 1, M) → (B, S, M)
    allowed = seg_idx.unsqueeze(-1) == mem_seg.unsqueeze(0).unsqueeze(0)

    # Mask: 0 where allowed, -inf where blocked
    mask = torch.where(allowed, 0.0, float("-inf"))

    # Expand for heads: (B, S, M) → (B*H, S, M)
    mask = mask.unsqueeze(1).expand(-1, nhead, -1, -1).reshape(
        batch * nhead, seq_len, total_mem,
    )
    return mask


# ---------------------------------------------------------------------------
# Reusable multi-phrase decoder
# ---------------------------------------------------------------------------


class ExpressionDecoder:
    """Decode multiple latent codes into a multi-phrase expression.

    Each latent code (phrase) is decoded autoregressively through the
    frozen decoder until EOS, with the KV cache persisting across
    phrase boundaries for coarticulation.  The result is a variable-
    length token sequence composed of multiple phrase constituents.

    This class encapsulates the Phase 1 (no-grad) decode logic shared
    by the expression game, dialogue game, and any future game type.

    Args:
        generator: The ``MultilingualVAEGenerator`` (frozen decoder).
    """

    def __init__(self, generator) -> None:
        self.gen = generator

    @torch.no_grad()
    def decode(
        self,
        z_seq: Tensor,
        z_weights: Tensor,
        max_tokens_per_phrase: int = 48,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Decode z-sequence into tokens via KV-cached autoregressive generation.

        Args:
            z_seq: ``(batch, num_phrases, latent_dim)`` latent codes.
            z_weights: ``(batch, num_phrases)`` per-phrase activity weights.
            max_tokens_per_phrase: Hard limit on tokens per phrase.

        Returns:
            tokens: ``(batch, max_total)`` token IDs.
            token_mask: ``(batch, max_total)`` boolean (True = valid).
            phrase_boundaries: ``(batch, num_phrases)`` start position
                of each phrase in the token sequence.
        """
        from lfm.generator.layers import (
            PhraseDecoder,
            multiscale_causal_mask,
            precompute_rope_freqs,
        )

        gen = self.gen
        batch, num_phrases, _ = z_seq.shape
        device = z_seq.device
        max_total = max_tokens_per_phrase * num_phrases

        # Project weighted z to decoder memory
        weighted_z = z_weights.unsqueeze(-1) * z_seq
        num_memory_tokens = gen._num_memory_tokens
        hidden_dim = gen.config.decoder_hidden_dim

        z_flat = _calibrate_or_quantize(gen, weighted_z.reshape(batch * num_phrases, -1))
        memories = gen.latent_to_decoder(z_flat).reshape(
            batch, num_phrases, num_memory_tokens, hidden_dim,
        )

        decoder = gen.decoder
        is_linguistic = isinstance(decoder, PhraseDecoder)

        # Precompute causal mask
        if gen._full_causal_mask is None or gen._full_causal_mask.size(1) < max_total + 1:
            gen._full_causal_mask = multiscale_causal_mask(
                max_total + 1,
                num_heads=gen.config.decoder_num_heads,
                head_windows=gen.config.attention_head_windows,
                global_every=gen.config.attention_global_every,
                device=device,
            )

        # Output buffers
        tokens = torch.zeros(batch, max_total, dtype=torch.long, device=device)
        token_mask = torch.zeros(batch, max_total, dtype=torch.bool, device=device)
        phrase_boundaries = torch.zeros(batch, num_phrases, dtype=torch.long, device=device)

        # Per-sample state
        current_phrase = torch.zeros(batch, dtype=torch.long, device=device)
        tokens_in_phrase = torch.zeros(batch, dtype=torch.long, device=device)
        total_position = torch.zeros(batch, dtype=torch.long, device=device)
        phrase_active = z_weights > 0.01
        finished = ~phrase_active[:, 0]

        # Extend RoPE if needed
        rope_freqs = gen._rope_freqs
        if rope_freqs is not None and rope_freqs.size(0) < max_total + 1:
            rope_freqs = precompute_rope_freqs(
                hidden_dim // gen.config.decoder_num_heads,
                max_total + 1, device=device,
            )

        # Initialize KV cache
        if is_linguistic:
            kv_cache = decoder.make_kv_cache(batch, max_total + 1, device, dtype=torch.float16)

        # BOS token
        bos_embed = gen.token_embedding(
            torch.full((batch, 1), gen.bos_id, dtype=torch.long, device=device),
        )
        batch_indices = torch.arange(batch, device=device)

        def _gather_phrase_memory() -> Tensor:
            idx = current_phrase.clamp(max=num_phrases - 1)
            mem = memories[batch_indices, idx]
            active = phrase_active[batch_indices, idx] & ~finished
            return mem * active.unsqueeze(-1).unsqueeze(-1).float()

        memory = _gather_phrase_memory()

        # Prime with BOS
        if is_linguistic:
            mask_row = gen._full_causal_mask[:, 0:1, 0:1]
            out = decoder.forward_cached(
                bos_embed, memory, kv_cache,
                rope_freqs=rope_freqs, tgt_mask_row=mask_row,
            )
            kv_cache.advance()
        else:
            out = decoder(bos_embed, memory)

        # Autoregressive decode loop
        for t in range(max_total):
            logits = gen.output_head(out[:, -1])
            next_token = logits.argmax(dim=-1)

            # Store tokens
            active = ~finished
            tokens[batch_indices, total_position] = next_token * active.long()
            token_mask[batch_indices, total_position] = active
            total_position += active.long()
            tokens_in_phrase += active.long()

            # Phrase switching on EOS or max tokens
            hit_eos = (next_token == gen.eos_id) & (tokens_in_phrase >= 1)
            hit_max = tokens_in_phrase >= max_tokens_per_phrase
            should_switch = (hit_eos | hit_max) & active

            current_phrase += should_switch.long()
            tokens_in_phrase *= ~should_switch

            # Record phrase boundaries
            switched_valid = should_switch & (current_phrase < num_phrases)
            if switched_valid.any():
                phrase_boundaries[
                    batch_indices[switched_valid],
                    current_phrase[switched_valid],
                ] = total_position[switched_valid]

            # Mark finished
            clamped = current_phrase.clamp(max=num_phrases - 1)
            next_inactive = ~phrase_active[batch_indices, clamped]
            finished = finished | (current_phrase >= num_phrases) | (should_switch & next_inactive)

            if finished.all():
                break

            memory = _gather_phrase_memory()
            new_embed = gen.token_embedding(next_token.unsqueeze(1))

            if is_linguistic:
                seq_so_far = kv_cache.seq_len + 1
                mask_row = gen._full_causal_mask[
                    :, kv_cache.seq_len:kv_cache.seq_len + 1, :seq_so_far
                ]
                out = decoder.forward_cached(
                    new_embed, memory, kv_cache,
                    rope_freqs=rope_freqs, tgt_mask_row=mask_row,
                )
                kv_cache.advance()
            else:
                # Fallback: non-linguistic decoder (no KV cache)
                cur_ids = torch.cat(
                    [torch.full((batch, 1), gen.bos_id, dtype=torch.long, device=device),
                     tokens[:, :t + 1]], dim=1,
                )
                all_embed = gen.token_embedding(cur_ids)
                tgt_mask = torch.nn.Transformer.generate_square_subsequent_mask(
                    cur_ids.size(1), device=device,
                )
                out = decoder(all_embed, memory, tgt_mask=tgt_mask)

        return tokens, token_mask, phrase_boundaries

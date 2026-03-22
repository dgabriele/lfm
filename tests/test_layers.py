"""Tests for linguistic decoder layers."""

from __future__ import annotations

import torch

from lfm.generator.layers import (
    LinguisticDecoder,
    LinguisticDecoderLayer,
    apply_rope,
    multiscale_causal_mask,
    precompute_rope_freqs,
)


def test_rope_roundtrip():
    """RoPE preserves tensor shape."""
    freqs = precompute_rope_freqs(dim=64, max_len=32)
    x = torch.randn(2, 4, 16, 64)  # (batch, heads, seq, head_dim)
    out = apply_rope(x, freqs[:16])
    assert out.shape == x.shape


def test_rope_position_invariance():
    """Same content at different positions gets different embeddings."""
    freqs = precompute_rope_freqs(dim=64, max_len=32)
    x = torch.randn(1, 1, 1, 64).expand(1, 1, 10, 64)
    out = apply_rope(x, freqs[:10])
    # Different positions should produce different outputs
    assert not torch.allclose(out[0, 0, 0], out[0, 0, 5], atol=1e-4)


def test_multiscale_mask_shape():
    """Multi-scale mask has correct shape."""
    mask = multiscale_causal_mask(
        seq_len=20, num_heads=4,
        head_windows=(3, 7, 15, 0), global_every=7,
    )
    assert mask.shape == (4, 20, 20)


def test_multiscale_mask_causal():
    """No head attends to future positions."""
    mask = multiscale_causal_mask(
        seq_len=16, num_heads=4,
        head_windows=(3, 7, 0, 0), global_every=7,
    )
    for h in range(4):
        for i in range(16):
            for j in range(i + 1, 16):
                assert mask[h, i, j] == float("-inf"), (
                    f"Head {h} pos {i} attends to future pos {j}"
                )


def test_multiscale_mask_window():
    """Small-window head doesn't attend beyond its window."""
    mask = multiscale_causal_mask(
        seq_len=20, num_heads=2,
        head_windows=(3, 0), global_every=100,  # no globals
    )
    # Head 0 (window=3): position 10 should NOT attend to position 6
    # (10 - 6 = 4 >= window=3)
    assert mask[0, 10, 6] == float("-inf")
    # But should attend to position 8 (10 - 8 = 2 < 3)
    assert mask[0, 10, 8] == 0.0


def test_multiscale_mask_global():
    """Global positions get full attention in all heads."""
    mask = multiscale_causal_mask(
        seq_len=20, num_heads=2,
        head_windows=(3, 3), global_every=7,
    )
    # Position 14 (a global position, 14 % 7 == 0) attends to everything <= 14
    for j in range(15):
        assert mask[0, 14, j] == 0.0, f"Global pos 14 can't attend to {j}"


def test_linguistic_decoder_layer_forward():
    """LinguisticDecoderLayer produces correct output shape."""
    layer = LinguisticDecoderLayer(d_model=64, nhead=4)
    tgt = torch.randn(2, 10, 64)
    memory = torch.randn(2, 1, 64)
    freqs = precompute_rope_freqs(16, 10)  # head_dim=64/4=16

    out = layer(tgt, memory, rope_freqs=freqs)
    assert out.shape == (2, 10, 64)


def test_linguistic_decoder_layer_no_rope():
    """Layer works without RoPE."""
    layer = LinguisticDecoderLayer(d_model=64, nhead=4)
    tgt = torch.randn(2, 10, 64)
    memory = torch.randn(2, 1, 64)

    out = layer(tgt, memory, rope_freqs=None)
    assert out.shape == (2, 10, 64)


def test_linguistic_decoder_shared():
    """Shared-layer decoder has fewer parameters."""
    shared = LinguisticDecoder(
        d_model=64, nhead=4, num_layers=4, share_layers=True
    )
    independent = LinguisticDecoder(
        d_model=64, nhead=4, num_layers=4, share_layers=False
    )

    shared_params = sum(p.numel() for p in shared.parameters())
    indep_params = sum(p.numel() for p in independent.parameters())
    # Shared should have roughly half the params (2 unique layers vs 4)
    assert shared_params < indep_params


def test_linguistic_decoder_forward():
    """Full linguistic decoder forward pass."""
    dec = LinguisticDecoder(
        d_model=64, nhead=4, num_layers=4, share_layers=True
    )
    tgt = torch.randn(2, 10, 64)
    memory = torch.randn(2, 1, 64)
    freqs = precompute_rope_freqs(16, 10)
    mask = multiscale_causal_mask(10, 4, (3, 7, 0, 0), global_every=5)

    out = dec(tgt, memory, tgt_mask=mask, rope_freqs=freqs)
    assert out.shape == (2, 10, 64)


def test_linguistic_decoder_gradient_flow():
    """Gradients flow through the linguistic decoder."""
    dec = LinguisticDecoder(
        d_model=64, nhead=4, num_layers=4, share_layers=True
    )
    tgt = torch.randn(2, 10, 64, requires_grad=True)
    memory = torch.randn(2, 1, 64)
    freqs = precompute_rope_freqs(16, 10)

    out = dec(tgt, memory, rope_freqs=freqs)
    out.sum().backward()
    assert tgt.grad is not None

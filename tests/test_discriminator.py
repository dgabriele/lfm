"""Tests for the structural adversarial discriminator."""

from __future__ import annotations

import torch

from lfm.generator.discriminator import StructuralDiscriminator


def test_discriminator_forward_shape():
    """Discriminator produces correct output shape."""
    disc = StructuralDiscriminator(
        vocab_size=100, embed_dim=32, hidden_dim=64
    )
    token_ids = torch.randint(0, 100, (4, 20))
    mask = torch.ones(4, 20, dtype=torch.bool)

    logits = disc(token_ids, mask)
    assert logits.shape == (4,)


def test_discriminator_masked():
    """Discriminator respects padding mask."""
    disc = StructuralDiscriminator(
        vocab_size=100, embed_dim=32, hidden_dim=64
    )
    token_ids = torch.randint(0, 100, (2, 10))
    mask_full = torch.ones(2, 10, dtype=torch.bool)
    mask_half = torch.ones(2, 10, dtype=torch.bool)
    mask_half[:, 5:] = False

    out_full = disc(token_ids, mask_full)
    out_half = disc(token_ids, mask_half)

    # Different masks should produce different outputs
    assert not torch.equal(out_full, out_half)


def test_discriminator_spectral_norm():
    """Spectral normalization is applied when configured."""
    disc = StructuralDiscriminator(
        vocab_size=100, embed_dim=32, hidden_dim=64,
        use_spectral_norm=True,
    )
    # spectral_norm adds weight_orig and weight_u attributes
    assert hasattr(disc.conv3, "weight_orig")
    assert hasattr(disc.conv3, "weight_u")


def test_discriminator_no_spectral_norm():
    """No spectral norm when disabled."""
    disc = StructuralDiscriminator(
        vocab_size=100, embed_dim=32, hidden_dim=64,
        use_spectral_norm=False,
    )
    assert not hasattr(disc.conv3, "weight_orig")


def test_discriminator_gradient_flow():
    """Gradients flow through discriminator to its parameters."""
    disc = StructuralDiscriminator(
        vocab_size=100, embed_dim=32, hidden_dim=64
    )
    token_ids = torch.randint(0, 100, (2, 10))
    mask = torch.ones(2, 10, dtype=torch.bool)

    logits = disc(token_ids, mask)
    loss = logits.mean()
    loss.backward()

    # Embedding and conv weights should have gradients
    assert disc.embedding.weight.grad is not None
    assert disc.head[0].weight.grad is not None

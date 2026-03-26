"""Tests for vector quantization modules."""

from __future__ import annotations

import torch

from lfm.generator.quantize import ResidualVQ, VectorQuantizer


class TestVectorQuantizer:
    def _make(self, **kw):
        defaults = dict(codebook_size=16, embedding_dim=8, commitment_weight=0.25)
        defaults.update(kw)
        return VectorQuantizer(**defaults)

    def test_output_shapes(self):
        vq = self._make()
        z = torch.randn(4, 8)
        quantized, loss, indices = vq(z)
        assert quantized.shape == (4, 8)
        assert loss.shape == ()
        assert indices.shape == (4,)

    def test_straight_through_gradient(self):
        vq = self._make()
        z = torch.randn(4, 8, requires_grad=True)
        quantized, loss, _ = vq(z)
        (quantized.sum() + loss).backward()
        assert z.grad is not None
        assert z.grad.abs().sum() > 0

    def test_commitment_loss_nonnegative(self):
        vq = self._make()
        z = torch.randn(4, 8)
        _, loss, _ = vq(z)
        assert loss.item() >= 0

    def test_encode_decode_roundtrip(self):
        vq = self._make()
        z = torch.randn(4, 8)
        indices = vq.encode(z)
        decoded = vq.decode(indices)
        # decoded should be codebook entries (exact lookup)
        assert decoded.shape == (4, 8)
        # re-encoding should give same indices
        indices2 = vq.encode(decoded)
        assert torch.equal(indices, indices2)

    def test_indices_in_range(self):
        vq = self._make(codebook_size=32)
        z = torch.randn(100, 8)
        _, _, indices = vq(z)
        assert indices.min() >= 0
        assert indices.max() < 32

    def test_ema_updates_codebook(self):
        vq = self._make(ema_update=True)
        vq.train()
        weights_before = vq.embedding.weight.data.clone()
        z = torch.randn(64, 8)
        vq(z)
        weights_after = vq.embedding.weight.data
        # At least some weights should have changed
        assert not torch.equal(weights_before, weights_after)

    def test_no_ema_in_eval(self):
        vq = self._make(ema_update=True)
        vq.eval()
        weights_before = vq.embedding.weight.data.clone()
        z = torch.randn(64, 8)
        vq(z)
        weights_after = vq.embedding.weight.data
        assert torch.equal(weights_before, weights_after)

    def test_gradient_update_mode(self):
        vq = self._make(ema_update=False)
        z = torch.randn(4, 8, requires_grad=True)
        quantized, loss, _ = vq(z)
        (quantized.sum() + loss).backward()
        assert z.grad is not None


class TestResidualVQ:
    def _make(self, **kw):
        defaults = dict(num_levels=4, codebook_size=16, embedding_dim=8)
        defaults.update(kw)
        return ResidualVQ(**defaults)

    def test_single_level_equivalent_to_vq(self):
        rvq = self._make(num_levels=1)
        z = torch.randn(4, 8)
        quantized, loss, all_indices = rvq(z)
        assert quantized.shape == (4, 8)
        assert len(all_indices) == 1
        assert all_indices[0].shape == (4,)

    def test_multi_level_output_shapes(self):
        rvq = self._make(num_levels=4)
        z = torch.randn(4, 8)
        quantized, loss, all_indices = rvq(z)
        assert quantized.shape == (4, 8)
        assert loss.shape == ()
        assert len(all_indices) == 4
        for idx in all_indices:
            assert idx.shape == (4,)

    def test_residual_decreases(self):
        rvq = self._make(num_levels=4)
        z = torch.randn(4, 8)
        residual = z
        residual_norms = []
        for level in rvq.levels:
            quantized, _, _ = level(residual)
            residual = residual - quantized.detach()
            residual_norms.append(residual.norm().item())
        # Each level should reduce the residual
        for i in range(1, len(residual_norms)):
            assert residual_norms[i] <= residual_norms[i - 1] + 1e-5

    def test_output_is_sum_of_levels(self):
        rvq = self._make(num_levels=3)
        z = torch.randn(4, 8)
        # Use encode+decode (no straight-through) to verify sum property
        all_indices = rvq.encode(z)
        decoded = rvq.decode(all_indices)
        manual_sum = torch.zeros_like(z)
        for level, indices in zip(rvq.levels, all_indices):
            manual_sum += level.decode(indices)
        assert torch.allclose(decoded, manual_sum, atol=1e-5)

    def test_encode_decode_deterministic(self):
        rvq = self._make(num_levels=3)
        z = torch.randn(4, 8)
        # Same input → same output (deterministic)
        all_indices1 = rvq.encode(z)
        all_indices2 = rvq.encode(z)
        for i1, i2 in zip(all_indices1, all_indices2):
            assert torch.equal(i1, i2)
        decoded1 = rvq.decode(all_indices1)
        decoded2 = rvq.decode(all_indices2)
        assert torch.allclose(decoded1, decoded2)

    def test_straight_through_gradient(self):
        rvq = self._make(num_levels=2)
        z = torch.randn(4, 8, requires_grad=True)
        quantized, loss, _ = rvq(z)
        (quantized.sum() + loss).backward()
        assert z.grad is not None
        assert z.grad.abs().sum() > 0

    def test_total_codebook_size(self):
        rvq = self._make(num_levels=4, codebook_size=512)
        assert rvq.total_codebook_size == 512 ** 4


class TestDeadCodeReset:
    def test_reset_splits_high_usage_codes(self):
        vq = VectorQuantizer(
            codebook_size=8, embedding_dim=4, ema_update=True, ema_decay=0.0,
        )
        vq.train()
        # Send all inputs to code 0 to create dead codes
        z = vq.embedding.weight.data[0:1].expand(64, -1).clone()
        z += torch.randn_like(z) * 0.001
        vq(z)
        # With decay=0, cluster_size = batch count directly.
        # Code 0 gets all 64 samples, others get 0.
        n_reset = vq.reset_dead_codes(threshold=0.5)
        assert n_reset > 0
        # The reset codes should now have embeddings near code 0
        code0 = vq.embedding.weight.data[0]
        for i in range(1, 8):
            if vq._ema_cluster_size[i] > 0:
                dist = (vq.embedding.weight.data[i] - code0).norm()
                assert dist < 1.0  # should be close to parent

    def test_reset_preserves_ema_consistency(self):
        vq = VectorQuantizer(codebook_size=8, embedding_dim=4, ema_update=True)
        vq.train()
        z = vq.embedding.weight.data[0:1].expand(64, -1).clone()
        z += torch.randn_like(z) * 0.001
        vq(z)
        vq.reset_dead_codes()
        # EMA cluster sizes should all be positive after reset
        assert (vq._ema_cluster_size >= 0).all()
        # Embeddings should not contain NaN
        assert not torch.isnan(vq.embedding.weight.data).any()

    def test_residual_vq_reset(self):
        rvq = ResidualVQ(num_levels=2, codebook_size=8, embedding_dim=4)
        rvq.train()
        z = torch.randn(32, 4)
        rvq(z)
        resets = rvq.reset_dead_codes()
        assert len(resets) == 2
        assert all(isinstance(r, int) for r in resets)

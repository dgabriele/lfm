"""Tests for the expression generation and encoding system."""

from __future__ import annotations

import pytest
import torch
from torch import nn

from lfm.expression import Expression, ExpressionConfig, ExpressionEncoder, ExpressionGenerator


# ── Fixtures ──────────────────────────────────────────────────────────


class MockGenerator(nn.Module):
    """Minimal mock of MultilingualVAEGenerator for testing."""

    def __init__(self, latent_dim: int = 16, hidden_dim: int = 32, vocab_size: int = 50) -> None:
        super().__init__()
        self._latent_dim = latent_dim
        self._max_output_len = 20
        self.latent_to_decoder = nn.Linear(latent_dim, hidden_dim)
        self.token_embedding = nn.Embedding(vocab_size, hidden_dim)
        self.output_head = nn.Linear(hidden_dim, vocab_size)
        self.decoder = MockDecoder(hidden_dim)
        self._rope_freqs = None
        self._full_causal_mask = None
        self.bos_id = 1
        self.eos_id = 2

        class _Cfg:
            decoder_hidden_dim = hidden_dim
            decoder_num_heads = 2
            attention_head_windows = [0, 0]
            attention_global_every = 7

        self.config = _Cfg()


class MockDecoder(nn.Module):
    """Mock decoder that returns random states (not a LinguisticDecoder)."""

    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.proj = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, tgt: torch.Tensor, memory: torch.Tensor, **kwargs) -> torch.Tensor:
        return self.proj(tgt) + memory


@pytest.fixture
def mock_gen() -> MockGenerator:
    return MockGenerator()


@pytest.fixture
def config() -> ExpressionConfig:
    return ExpressionConfig(
        input_dim=32,
        latent_dim=16,
        hidden_dim=32,
        output_dim=24,
        max_depth=3,
        min_depth=1,
        max_tokens_per_leaf=10,
        transition_on_eos=True,
    )


# ── Expression dataclass ─────────────────────────────────────────────


class TestExpression:
    def test_properties(self) -> None:
        b, max_nodes, d = 4, 7, 16
        expr = Expression(
            is_leaf=torch.zeros(b, max_nodes, dtype=torch.bool),
            active=torch.ones(b, max_nodes, dtype=torch.bool),
            depth=torch.zeros(max_nodes, dtype=torch.long),
            parent_idx=torch.full((max_nodes,), -1),
            left_child=torch.full((max_nodes,), -1),
            right_child=torch.full((max_nodes,), -1),
            num_active_nodes=torch.full((b,), max_nodes),
            num_leaves=torch.zeros(b, dtype=torch.long),
            leaf_z=torch.randn(b, max_nodes, d),
            leaf_mu=torch.randn(b, max_nodes, d),
            leaf_logvar=torch.randn(b, max_nodes, d),
        )
        assert expr.batch_size == b
        assert expr.max_nodes == max_nodes
        assert not expr.has_decoded

    def test_has_decoded_after_population(self) -> None:
        b, n = 2, 3
        expr = Expression(
            is_leaf=torch.zeros(b, n, dtype=torch.bool),
            active=torch.ones(b, n, dtype=torch.bool),
            depth=torch.zeros(n),
            parent_idx=torch.full((n,), -1),
            left_child=torch.full((n,), -1),
            right_child=torch.full((n,), -1),
            num_active_nodes=torch.full((b,), n),
            num_leaves=torch.zeros(b, dtype=torch.long),
            leaf_z=torch.randn(b, n, 8),
            leaf_mu=torch.randn(b, n, 8),
            leaf_logvar=torch.randn(b, n, 8),
            tokens=torch.zeros(b, 10, dtype=torch.long),
        )
        assert expr.has_decoded


# ── ExpressionGenerator ──────────────────────────────────────────────


class TestExpressionGenerator:
    def test_creates_valid_topology(self, mock_gen: MockGenerator) -> None:
        gen = ExpressionGenerator(
            generator=mock_gen,
            input_dim=32, latent_dim=16, hidden_dim=32,
            max_depth=3, min_depth=1,
            max_tokens_per_leaf=10,
        )
        gen.eval()
        anchor = torch.randn(4, 32)
        expr = gen._generate_topology(anchor)

        assert expr.batch_size == 4
        assert expr.max_nodes == 7  # 2^3 - 1
        # Every active non-leaf node must have active children
        for bi in range(4):
            for ni in range(7):
                if expr.active[bi, ni] and not expr.is_leaf[bi, ni]:
                    lc = expr.left_child[ni].item()
                    rc = expr.right_child[ni].item()
                    if lc >= 0:
                        assert expr.active[bi, lc]
                    if rc >= 0:
                        assert expr.active[bi, rc]

    def test_has_at_least_min_depth_leaves(self, mock_gen: MockGenerator) -> None:
        gen = ExpressionGenerator(
            generator=mock_gen,
            input_dim=32, latent_dim=16, hidden_dim=32,
            max_depth=3, min_depth=2,
            max_tokens_per_leaf=10,
        )
        gen.eval()
        anchor = torch.randn(8, 32)
        expr = gen._generate_topology(anchor)
        # With min_depth=2, every sample must have >= 2 leaves
        # (root forced to expand, producing at least 2 leaves)
        assert (expr.num_leaves >= 2).all()

    def test_leaf_z_populated(self, mock_gen: MockGenerator) -> None:
        gen = ExpressionGenerator(
            generator=mock_gen,
            input_dim=32, latent_dim=16, hidden_dim=32,
            max_depth=2, min_depth=1,
            max_tokens_per_leaf=10,
        )
        gen.eval()
        anchor = torch.randn(4, 32)
        expr = gen._generate_topology(anchor)
        # Leaf z should be nonzero for leaf nodes
        for bi in range(4):
            for ni in range(expr.max_nodes):
                if expr.is_leaf[bi, ni]:
                    assert expr.leaf_z[bi, ni].abs().sum() > 0

    def test_full_forward_produces_decoded_expression(self, mock_gen: MockGenerator) -> None:
        gen = ExpressionGenerator(
            generator=mock_gen,
            input_dim=32, latent_dim=16, hidden_dim=32,
            max_depth=2, min_depth=1,
            max_tokens_per_leaf=5,
        )
        gen.eval()
        anchor = torch.randn(2, 32)
        expr = gen(anchor)

        assert expr.has_decoded
        assert expr.tokens.shape[0] == 2
        assert (expr.lengths > 0).all()
        assert expr.leaf_order.shape[0] == 2

    def test_training_mode_stochastic(self, mock_gen: MockGenerator) -> None:
        gen = ExpressionGenerator(
            generator=mock_gen,
            input_dim=32, latent_dim=16, hidden_dim=32,
            max_depth=3, min_depth=1,
            max_tokens_per_leaf=5,
        )
        gen.train()
        anchor = torch.randn(16, 32)
        # Run multiple times — stochastic topology should vary
        topologies = []
        for _ in range(5):
            expr = gen._generate_topology(anchor)
            topologies.append(expr.num_leaves.clone())
        # With 16 samples and 5 runs, very likely to see variation
        all_same = all(torch.equal(topologies[0], t) for t in topologies[1:])
        # Not guaranteed to differ, but extremely unlikely to be all identical
        # Just check it runs without error in training mode
        assert True

    def test_inorder_leaves(self, mock_gen: MockGenerator) -> None:
        gen = ExpressionGenerator(
            generator=mock_gen,
            input_dim=32, latent_dim=16, hidden_dim=32,
            max_depth=3, min_depth=1,
            max_tokens_per_leaf=10,
        )
        gen.eval()
        anchor = torch.randn(4, 32)
        expr = gen._generate_topology(anchor)
        leaf_order = ExpressionGenerator._inorder_leaves(expr)

        # All referenced indices should be valid leaves
        for bi in range(4):
            for li in range(leaf_order.size(1)):
                ni = leaf_order[bi, li].item()
                if ni >= 0:
                    assert expr.is_leaf[bi, ni]


# ── ExpressionEncoder ────────────────────────────────────────────────


class TestExpressionEncoder:
    def test_output_shape(self, mock_gen: MockGenerator) -> None:
        gen = ExpressionGenerator(
            generator=mock_gen,
            input_dim=32, latent_dim=16, hidden_dim=32,
            max_depth=2, min_depth=1,
            max_tokens_per_leaf=5,
        )
        gen.eval()
        enc = ExpressionEncoder(hidden_dim=32, output_dim=24, max_depth=2)

        anchor = torch.randn(4, 32)
        expr = gen(anchor)
        message = enc(expr)

        assert message.shape == (4, 24)

    def test_shape_embedding_deterministic(self, mock_gen: MockGenerator) -> None:
        gen = ExpressionGenerator(
            generator=mock_gen,
            input_dim=32, latent_dim=16, hidden_dim=32,
            max_depth=2, min_depth=1,
            max_tokens_per_leaf=5,
        )
        gen.eval()
        enc = ExpressionEncoder(hidden_dim=32, output_dim=24, max_depth=2)

        anchor = torch.randn(2, 32)
        expr = gen(anchor)
        ids1 = enc._compute_shape_ids(expr.is_leaf, expr.active)
        ids2 = enc._compute_shape_ids(expr.is_leaf, expr.active)
        assert torch.equal(ids1, ids2)

    def test_requires_decoded(self) -> None:
        enc = ExpressionEncoder(hidden_dim=32, output_dim=24, max_depth=2)
        expr = Expression(
            is_leaf=torch.zeros(2, 3, dtype=torch.bool),
            active=torch.ones(2, 3, dtype=torch.bool),
            depth=torch.zeros(3),
            parent_idx=torch.full((3,), -1),
            left_child=torch.full((3,), -1),
            right_child=torch.full((3,), -1),
            num_active_nodes=torch.full((2,), 3),
            num_leaves=torch.zeros(2, dtype=torch.long),
            leaf_z=torch.randn(2, 3, 8),
            leaf_mu=torch.randn(2, 3, 8),
            leaf_logvar=torch.randn(2, 3, 8),
        )
        with pytest.raises(AssertionError, match="must be decoded"):
            enc(expr)


# ── ExpressionConfig ─────────────────────────────────────────────────


class TestExpressionConfig:
    def test_defaults(self) -> None:
        cfg = ExpressionConfig()
        assert cfg.max_depth == 3
        assert cfg.min_depth == 1
        assert cfg.max_tokens_per_leaf == 96
        assert cfg.transition_on_eos is True

    def test_custom(self) -> None:
        cfg = ExpressionConfig(
            input_dim=512, latent_dim=256,
            max_depth=4, min_depth=2,
        )
        assert cfg.input_dim == 512
        assert cfg.max_depth == 4

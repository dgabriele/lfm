"""Tests for tree-structured communication."""

from __future__ import annotations

import torch

from lfm.communication.tree import (
    TreeMessageEncoder,
    TreeSender,
    _compute_tree_layout,
)


class TestTreeLayout:
    def test_binary_depth2(self):
        max_nodes, depths, parents = _compute_tree_layout(2, 2)
        assert max_nodes == 3
        assert depths.tolist() == [0, 1, 1]
        assert parents.tolist() == [-1, 0, 0]

    def test_binary_depth3(self):
        max_nodes, depths, parents = _compute_tree_layout(3, 2)
        assert max_nodes == 7
        assert depths.tolist() == [0, 1, 1, 2, 2, 2, 2]

    def test_ternary_depth2(self):
        max_nodes, depths, parents = _compute_tree_layout(2, 3)
        assert max_nodes == 4
        assert depths.tolist() == [0, 1, 1, 1]
        assert parents.tolist() == [-1, 0, 0, 0]


class TestTreeSender:
    def _make_mock_generator(self):
        """Create a minimal mock generator for testing."""

        class MockGenerator(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self._max_output_len = 8
                self._latent_dim = 16

            def _decode(self, z):
                b = z.size(0)
                s = self._max_output_len
                h = 32
                token_ids = torch.zeros(b, s, dtype=torch.long)
                token_probs = torch.zeros(b, s, 50)
                decoder_states = torch.randn(b, s, h)
                lengths = torch.full((b,), s)
                masks = torch.ones(b, s, dtype=torch.bool)
                return token_ids, token_probs, decoder_states, lengths, masks

        return MockGenerator()

    def test_output_shapes(self):
        gen = self._make_mock_generator()
        sender = TreeSender(
            gen, input_dim=32, latent_dim=16, hidden_dim=32,
            max_depth=2, max_children=2, min_depth=1,
        )
        sender.eval()
        anchor = torch.randn(4, 32)
        tree = sender(anchor)

        assert tree.z.shape == (4, 3, 16)
        assert tree.active.shape == (4, 3)
        assert tree.mu.shape == (4, 3, 16)
        assert tree.active[:, 0].all()  # root always active
        assert tree.decoder_states.shape[0] == 4
        assert tree.decoder_states.shape[1] == 3

    def test_variable_branching(self):
        gen = self._make_mock_generator()
        sender = TreeSender(
            gen, input_dim=32, latent_dim=16, hidden_dim=32,
            max_depth=2, max_children=3, min_depth=1,
        )
        sender.train()
        anchor = torch.randn(8, 32)
        tree = sender(anchor)

        # Root always active
        assert tree.active[:, 0].all()
        # Variable children — not all samples have same branching
        assert tree.num_children.shape == (8, 4)  # max_nodes for depth=2, K=3

    def test_gradient_flows(self):
        gen = self._make_mock_generator()
        sender = TreeSender(
            gen, input_dim=32, latent_dim=16, hidden_dim=32,
            max_depth=2, max_children=2, min_depth=1,
        )
        anchor = torch.randn(2, 32, requires_grad=True)
        tree = sender(anchor)
        loss = tree.mu[tree.active].sum()
        loss.backward()
        assert anchor.grad is not None


class TestTreeMessageEncoder:
    def test_output_shape(self):
        max_nodes = 7
        encoder = TreeMessageEncoder(
            hidden_dim=32, output_dim=16, max_nodes=max_nodes,
            max_depth=3, num_heads=2, num_layers=1,
        )

        from lfm.communication.tree import TreeMessage

        tree = TreeMessage(
            z=torch.randn(4, max_nodes, 8),
            mu=torch.randn(4, max_nodes, 8),
            logvar=torch.randn(4, max_nodes, 8),
            active=torch.ones(4, max_nodes, dtype=torch.bool),
            depth=torch.tensor([0, 1, 1, 2, 2, 2, 2]),
            parent_idx=torch.tensor([-1, 0, 0, 1, 1, 2, 2]),
            num_children=torch.full((4, max_nodes), 2),
            decoder_states=torch.randn(4, max_nodes, 8, 32),
            token_ids=torch.zeros(4, max_nodes, 8, dtype=torch.long),
            node_lengths=torch.full((4, max_nodes), 8),
            node_masks=torch.ones(4, max_nodes, 8, dtype=torch.bool),
        )

        msg = encoder(tree)
        assert msg.shape == (4, 16)

    def test_inactive_nodes_masked(self):
        max_nodes = 3
        encoder = TreeMessageEncoder(
            hidden_dim=32, output_dim=16, max_nodes=max_nodes,
            max_depth=2, num_heads=2, num_layers=1,
        )

        from lfm.communication.tree import TreeMessage

        active = torch.tensor([[True, True, False], [True, False, False]])
        tree = TreeMessage(
            z=torch.randn(2, max_nodes, 8),
            mu=torch.randn(2, max_nodes, 8),
            logvar=torch.randn(2, max_nodes, 8),
            active=active,
            depth=torch.tensor([0, 1, 1]),
            parent_idx=torch.tensor([-1, 0, 0]),
            num_children=torch.zeros(2, max_nodes, dtype=torch.long),
            decoder_states=torch.randn(2, max_nodes, 8, 32),
            token_ids=torch.zeros(2, max_nodes, 8, dtype=torch.long),
            node_lengths=torch.full((2, max_nodes), 8),
            node_masks=torch.ones(2, max_nodes, 8, dtype=torch.bool),
        )

        msg = encoder(tree)
        assert msg.shape == (2, 16)

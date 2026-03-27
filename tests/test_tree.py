"""Tests for constituency-style tree communication."""

from __future__ import annotations

import torch

from lfm.communication.tree import (
    TreeMessage,
    TreeMessageEncoder,
    TreeSender,
    _max_nodes_for_depth,
)


class TestTreeLayout:
    def test_max_nodes(self):
        assert _max_nodes_for_depth(1) == 1
        assert _max_nodes_for_depth(2) == 3
        assert _max_nodes_for_depth(3) == 7
        assert _max_nodes_for_depth(4) == 15


class TestTreeSender:
    def _make_mock_generator(self):
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
            max_depth=3, min_depth=1,
        )
        sender.eval()
        tree = sender(torch.randn(4, 32))

        assert tree.active.shape == (4, 7)  # max_nodes for depth 3
        assert tree.is_leaf.shape == (4, 7)
        assert tree.leaf_z.shape == (4, 7, 16)
        assert tree.active[:, 0].all()  # root always active

    def test_root_always_active(self):
        gen = self._make_mock_generator()
        sender = TreeSender(
            gen, input_dim=32, latent_dim=16, hidden_dim=32,
            max_depth=2, min_depth=1,
        )
        sender.eval()
        tree = sender(torch.randn(8, 32))
        assert tree.active[:, 0].all()

    def test_leaves_have_z(self):
        gen = self._make_mock_generator()
        sender = TreeSender(
            gen, input_dim=32, latent_dim=16, hidden_dim=32,
            max_depth=2, min_depth=1,
        )
        sender.eval()
        tree = sender(torch.randn(4, 32))
        # All leaves should have nonzero z
        for i in range(3):
            leaf_mask = tree.is_leaf[:, i]
            if leaf_mask.any():
                leaf_norms = tree.leaf_z[leaf_mask, i].norm(dim=-1)
                assert (leaf_norms > 0).all()

    def test_leaves_have_decoded_content(self):
        gen = self._make_mock_generator()
        sender = TreeSender(
            gen, input_dim=32, latent_dim=16, hidden_dim=32,
            max_depth=2, min_depth=1,
        )
        sender.eval()
        tree = sender(torch.randn(4, 32))
        # Leaf nodes should have nonzero decoder states
        assert tree.leaf_states.shape[0] == 4
        assert (tree.leaf_lengths[tree.is_leaf] > 0).all()

    def test_internal_nodes_not_decoded(self):
        gen = self._make_mock_generator()
        sender = TreeSender(
            gen, input_dim=32, latent_dim=16, hidden_dim=32,
            max_depth=2, min_depth=1,
        )
        sender.eval()
        tree = sender(torch.randn(4, 32))
        # Internal (non-leaf) nodes should have zero decoder states
        internal = tree.active & ~tree.is_leaf
        if internal.any():
            for i in range(3):
                int_mask = internal[:, i]
                if int_mask.any():
                    assert (tree.leaf_lengths[int_mask, i] == 0).all()

    def test_gradient_flows_through_leaf_mu(self):
        gen = self._make_mock_generator()
        sender = TreeSender(
            gen, input_dim=32, latent_dim=16, hidden_dim=32,
            max_depth=2, min_depth=1,
        )
        anchor = torch.randn(2, 32, requires_grad=True)
        tree = sender(anchor)
        loss = tree.leaf_mu[tree.is_leaf].sum()
        loss.backward()
        assert anchor.grad is not None
        assert anchor.grad.abs().sum() > 0

    def test_variable_depth(self):
        gen = self._make_mock_generator()
        sender = TreeSender(
            gen, input_dim=32, latent_dim=16, hidden_dim=32,
            max_depth=3, min_depth=1,
        )
        sender.train()
        tree = sender(torch.randn(16, 32))
        # Different samples may have different numbers of active nodes
        node_counts = tree.num_active_nodes
        assert node_counts.min() >= 1  # at least root
        # With training randomness, not all trees should be identical
        # (probabilistic — may rarely fail)
        assert node_counts.float().std() > 0 or node_counts[0] > 1


class TestTreeMessageEncoder:
    def _make_tree(self, batch: int, max_nodes: int, hidden_dim: int):
        """Create a simple fully-expanded binary tree for testing."""
        depths = [0] * max_nodes
        parents = [-1] * max_nodes
        left_children = [-1] * max_nodes
        right_children = [-1] * max_nodes
        for i in range(max_nodes):
            if i > 0:
                depths[i] = depths[(i - 1) // 2] + 1
                parents[i] = (i - 1) // 2
            lc, rc = 2 * i + 1, 2 * i + 2
            if lc < max_nodes:
                left_children[i] = lc
            if rc < max_nodes:
                right_children[i] = rc

        # Leaves are nodes with no children
        is_leaf = torch.tensor(
            [left_children[i] == -1 for i in range(max_nodes)],
        ).unsqueeze(0).expand(batch, -1)

        return TreeMessage(
            is_leaf=is_leaf,
            active=torch.ones(batch, max_nodes, dtype=torch.bool),
            depth=torch.tensor(depths),
            parent_idx=torch.tensor(parents),
            left_child=torch.tensor(left_children),
            right_child=torch.tensor(right_children),
            num_active_nodes=torch.full((batch,), max_nodes),
            num_leaves=is_leaf.sum(dim=1),
            leaf_z=torch.randn(batch, max_nodes, 8),
            leaf_mu=torch.randn(batch, max_nodes, 8),
            leaf_logvar=torch.randn(batch, max_nodes, 8),
            leaf_states=torch.randn(batch, max_nodes, 8, hidden_dim),
            leaf_token_ids=torch.zeros(batch, max_nodes, 8, dtype=torch.long),
            leaf_lengths=torch.full((batch, max_nodes), 8),
            leaf_masks=torch.ones(batch, max_nodes, 8, dtype=torch.bool),
        )

    def test_output_shape(self):
        encoder = TreeMessageEncoder(
            hidden_dim=32, output_dim=16, max_nodes=7, max_depth=3,
        )
        tree = self._make_tree(batch=4, max_nodes=7, hidden_dim=32)
        msg = encoder(tree)
        assert msg.shape == (4, 16)

    def test_gradient_flows(self):
        encoder = TreeMessageEncoder(
            hidden_dim=32, output_dim=16, max_nodes=3, max_depth=2,
        )
        tree = self._make_tree(batch=2, max_nodes=3, hidden_dim=32)
        tree.leaf_states.requires_grad_(True)
        msg = encoder(tree)
        msg.sum().backward()
        assert tree.leaf_states.grad is not None

    def test_different_trees_different_messages(self):
        encoder = TreeMessageEncoder(
            hidden_dim=32, output_dim=16, max_nodes=3, max_depth=2,
        )
        tree1 = self._make_tree(batch=1, max_nodes=3, hidden_dim=32)
        tree2 = self._make_tree(batch=1, max_nodes=3, hidden_dim=32)
        msg1 = encoder(tree1)
        msg2 = encoder(tree2)
        # Different random leaf states → different messages
        assert not torch.allclose(msg1, msg2)

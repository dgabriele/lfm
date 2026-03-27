"""Constituency-style tree communication for agent games.

The agent produces a binary tree where only terminal (leaf) nodes carry
IPA expressions decoded through the frozen decoder.  Internal nodes are
compositional operators that combine their children's representations
bottom-up.  The tree topology itself is part of the message — different
structures compose the same leaf expressions into different meanings.

    ○ (root: composed representation → receiver)
   / \\
  ○   z₃ (leaf: decoded to IPA)
 / \\
z₁  z₂ (leaves: decoded to IPA)

The sender makes two orthogonal decisions:
1. Tree shape (topology via learned branching)
2. Leaf content (z vectors → frozen decoder)

The receiver composes bottom-up: decode leaves → compose internal
nodes recursively → root representation → score candidates.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import torch
from torch import Tensor, nn

logger = logging.getLogger(__name__)


@dataclass
class TreeMessage:
    """A batch of constituency-style tree messages.

    Nodes indexed in BFS order.  Only leaf nodes have decoded content.
    Internal nodes have composed representations (filled by the encoder).
    """

    # Tree topology
    is_leaf: Tensor          # (batch, max_nodes) bool
    active: Tensor           # (batch, max_nodes) bool
    depth: Tensor            # (max_nodes,) int
    parent_idx: Tensor       # (max_nodes,) int — -1 for root
    left_child: Tensor       # (max_nodes,) int — -1 if leaf/absent
    right_child: Tensor      # (max_nodes,) int — -1 if leaf/absent
    num_active_nodes: Tensor # (batch,) int
    num_leaves: Tensor       # (batch,) int

    # Leaf z vectors
    leaf_z: Tensor           # (batch, max_nodes, latent_dim) — only leaves populated
    leaf_mu: Tensor          # (batch, max_nodes, latent_dim)
    leaf_logvar: Tensor      # (batch, max_nodes, latent_dim)

    # Decoded leaf content (filled after decoding)
    leaf_states: Tensor = field(default_factory=lambda: torch.empty(0))
    leaf_token_ids: Tensor = field(default_factory=lambda: torch.empty(0))
    leaf_lengths: Tensor = field(default_factory=lambda: torch.empty(0))
    leaf_masks: Tensor = field(default_factory=lambda: torch.empty(0))


def _max_nodes_for_depth(max_depth: int) -> int:
    """Max nodes in a complete binary tree of given depth."""
    return 2 ** max_depth - 1


class TreeSender(nn.Module):
    """Produce a constituency tree: topology + leaf z vectors.

    Top-down generation: at each internal node, decide whether to
    expand (create two children) or stop (make this a leaf).  Leaf
    nodes get z vectors decoded through the frozen decoder.

    Args:
        generator: Frozen generator (provides _decode).
        input_dim: Input embedding dimension.
        latent_dim: Latent z dimension.
        hidden_dim: Decoder hidden dimension.
        max_depth: Maximum tree depth (root = 0).
        min_depth: Minimum forced depth before halting is allowed.
    """

    def __init__(
        self,
        generator: nn.Module,
        input_dim: int,
        latent_dim: int,
        hidden_dim: int,
        max_depth: int = 3,
        min_depth: int = 1,
    ) -> None:
        super().__init__()
        self.generator = generator
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.max_depth = max_depth
        self.min_depth = min_depth
        self.max_nodes = _max_nodes_for_depth(max_depth)

        # Precompute BFS binary tree layout
        depths = [0] * self.max_nodes
        parents = [-1] * self.max_nodes
        left_children = [-1] * self.max_nodes
        right_children = [-1] * self.max_nodes
        for i in range(self.max_nodes):
            if i > 0:
                depths[i] = depths[(i - 1) // 2] + 1
                parents[i] = (i - 1) // 2
            lc = 2 * i + 1
            rc = 2 * i + 2
            if lc < self.max_nodes:
                left_children[i] = lc
            if rc < self.max_nodes:
                right_children[i] = rc

        self.register_buffer("_depth", torch.tensor(depths), persistent=False)
        self.register_buffer("_parent", torch.tensor(parents), persistent=False)
        self.register_buffer("_left", torch.tensor(left_children), persistent=False)
        self.register_buffer("_right", torch.tensor(right_children), persistent=False)

        # Root context: input → hidden representation for tree decisions
        self.root_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # Expand decision: hidden → (expand_logit, left_ctx, right_ctx)
        # expand_logit: scalar — probability of expanding (vs being a leaf)
        # left_ctx, right_ctx: hidden vectors passed to children
        self.expand_head = nn.Linear(hidden_dim, 1 + hidden_dim * 2)

        # Leaf z projection: hidden → (mu, logvar)
        self.leaf_proj = nn.Linear(hidden_dim, latent_dim * 2)

        # Initialize near-zero for stable start
        with torch.no_grad():
            self.expand_head.weight.mul_(0.01)
            self.expand_head.bias.zero_()
            # Bias expand_logit toward expanding at start
            self.expand_head.bias.data[0] = 1.0

    def forward(self, anchor: Tensor) -> TreeMessage:
        """Generate tree topology and leaf z vectors.

        Args:
            anchor: (batch, input_dim) input embeddings.

        Returns:
            TreeMessage with topology and leaf z (not yet decoded).
        """
        b = anchor.size(0)
        device = anchor.device
        max_seq = self.generator._max_output_len

        # Buffers
        active = torch.zeros(b, self.max_nodes, dtype=torch.bool, device=device)
        is_leaf = torch.zeros(b, self.max_nodes, dtype=torch.bool, device=device)
        leaf_z = torch.zeros(b, self.max_nodes, self.latent_dim, device=device)
        leaf_mu = torch.zeros_like(leaf_z)
        leaf_logvar = torch.zeros_like(leaf_z)

        # Hidden context per node — stored as a dict to avoid in-place
        # tensor modifications that break autograd.
        node_ctx: dict[int, Tensor] = {}

        # Root
        active[:, 0] = True
        node_ctx[0] = self.root_proj(anchor)

        # Top-down: decide expand vs leaf at each active node
        for i in range(self.max_nodes):
            depth = self._depth[i].item()
            lc = self._left[i].item()
            rc = self._right[i].item()

            node_active = active[:, i]  # (B,)
            if not node_active.any():
                continue

            ctx = node_ctx.get(i, torch.zeros(b, self.hidden_dim, device=device))

            # At max depth - 1 (last internal level) or if no children possible: leaf
            if lc == -1 or depth >= self.max_depth - 1:
                is_leaf[:, i] = node_active
                h = self.leaf_proj(ctx)
                mu, logvar = h.chunk(2, dim=-1)
                z = self._reparameterize(mu, logvar)
                leaf_z[:, i] = z * node_active.unsqueeze(-1).float()
                leaf_mu[:, i] = mu * node_active.unsqueeze(-1).float()
                leaf_logvar[:, i] = logvar * node_active.unsqueeze(-1).float()
                continue

            # Decide: expand or leaf?
            head_out = self.expand_head(ctx)  # (B, 1 + 2H)
            expand_logit = head_out[:, 0]     # (B,)
            child_ctx = head_out[:, 1:]       # (B, 2H)
            left_ctx = child_ctx[:, :self.hidden_dim]
            right_ctx = child_ctx[:, self.hidden_dim:]

            # Force expand at shallow depths
            if depth < self.min_depth:
                expand = node_active
            elif self.training:
                expand_prob = torch.sigmoid(expand_logit)
                expand = (torch.rand_like(expand_prob) < expand_prob) & node_active
            else:
                expand = (expand_logit > 0) & node_active

            # Nodes that expand → activate children
            if lc < self.max_nodes:
                active[:, lc] = expand
                node_ctx[lc] = left_ctx * expand.unsqueeze(-1).float()
            if rc < self.max_nodes:
                active[:, rc] = expand
                node_ctx[rc] = right_ctx * expand.unsqueeze(-1).float()

            # Nodes that don't expand → become leaves
            not_expand = node_active & ~expand
            is_leaf[:, i] = not_expand
            if not_expand.any():
                h = self.leaf_proj(ctx)
                mu, logvar = h.chunk(2, dim=-1)
                z = self._reparameterize(mu, logvar)
                leaf_z[:, i] = z * not_expand.unsqueeze(-1).float()
                leaf_mu[:, i] = mu * not_expand.unsqueeze(-1).float()
                leaf_logvar[:, i] = logvar * not_expand.unsqueeze(-1).float()

        # Decode all leaves in one batched call
        leaf_states = torch.zeros(b, self.max_nodes, max_seq, self.hidden_dim, device=device)
        leaf_token_ids = torch.zeros(b, self.max_nodes, max_seq, dtype=torch.long, device=device)
        leaf_lengths = torch.zeros(b, self.max_nodes, dtype=torch.long, device=device)
        leaf_masks = torch.zeros(b, self.max_nodes, max_seq, dtype=torch.bool, device=device)

        # Gather all leaf z vectors into a flat batch
        leaf_flat_mask = is_leaf.reshape(-1)  # (B*N,)
        leaf_z_flat = leaf_z.reshape(-1, self.latent_dim)  # (B*N, D)

        if leaf_flat_mask.any():
            active_leaf_z = leaf_z_flat[leaf_flat_mask]  # (num_leaves, D)
            with torch.no_grad():
                tids, _probs, states, lens, masks = self.generator._decode(active_leaf_z)

            # Scatter back into buffers
            leaf_idx = leaf_flat_mask.nonzero(as_tuple=True)[0]
            batch_idx = leaf_idx // self.max_nodes
            node_idx = leaf_idx % self.max_nodes
            seq_len = tids.size(1)

            for k in range(len(leaf_idx)):
                bi, ni = batch_idx[k], node_idx[k]
                leaf_states[bi, ni, :seq_len] = states[k]
                leaf_token_ids[bi, ni, :seq_len] = tids[k]
                leaf_lengths[bi, ni] = lens[k]
                leaf_masks[bi, ni, :seq_len] = masks[k]

        return TreeMessage(
            is_leaf=is_leaf,
            active=active,
            depth=self._depth,
            parent_idx=self._parent,
            left_child=self._left,
            right_child=self._right,
            num_active_nodes=active.sum(dim=1),
            num_leaves=is_leaf.sum(dim=1),
            leaf_z=leaf_z,
            leaf_mu=leaf_mu,
            leaf_logvar=leaf_logvar,
            leaf_states=leaf_states,
            leaf_token_ids=leaf_token_ids,
            leaf_lengths=leaf_lengths,
            leaf_masks=leaf_masks,
        )

    def _reparameterize(self, mu: Tensor, logvar: Tensor) -> Tensor:
        if self.training:
            std = (0.5 * logvar).exp()
            return mu + std * torch.randn_like(std)
        return mu


class TreeMessageEncoder(nn.Module):
    """Bottom-up composition of a constituency tree into a message vector.

    Leaf nodes: pool decoder hidden states → leaf representation.
    Internal nodes: compose children's representations via learned Merge.
    Root representation → receiver scoring.

    This mirrors syntactic Merge: the meaning of a phrase is a function
    of the meanings of its constituents and how they combine.

    Args:
        hidden_dim: Decoder hidden dimension (for leaf pooling).
        output_dim: Output message vector dimension.
        max_nodes: Maximum nodes in the tree.
        max_depth: Maximum tree depth.
    """

    def __init__(
        self,
        hidden_dim: int,
        output_dim: int,
        max_nodes: int,
        max_depth: int,
    ) -> None:
        super().__init__()
        self.max_nodes = max_nodes
        self.output_dim = output_dim

        # Leaf encoder: pool decoder states → representation
        self.leaf_enc = nn.Sequential(
            nn.Linear(hidden_dim, output_dim),
            nn.GELU(),
            nn.Linear(output_dim, output_dim),
        )

        # Merge: compose two children → parent representation
        # This is the core compositional operation.
        self.merge = nn.Sequential(
            nn.Linear(output_dim * 2, output_dim),
            nn.GELU(),
            nn.Linear(output_dim, output_dim),
        )

        # Depth embedding — different composition behavior at different levels
        self.depth_embed = nn.Embedding(max_depth, output_dim)

        # Shape embedding — explicit encoding of tree topology.
        # Each distinct tree shape (pattern of active/leaf nodes) gets
        # a learned embedding, separating structural information from
        # compositional content.  With max_depth=3, there are ~14 valid
        # binary tree shapes; we allocate generously.
        max_shapes = 2 ** max_nodes  # upper bound (most won't be used)
        # Practical cap — Catalan numbers grow slowly
        self.max_shape_ids = min(max_shapes, 256)
        self.shape_embed = nn.Embedding(self.max_shape_ids, output_dim)

    def forward(self, tree: TreeMessage) -> Tensor:
        """Compose tree bottom-up into a root message vector.

        Args:
            tree: TreeMessage from TreeSender (with decoded leaf states).

        Returns:
            (batch, output_dim) message vector (root composition).
        """
        b = tree.active.size(0)
        device = tree.active.device

        # Node representations (filled bottom-up)
        node_repr = torch.zeros(
            b, self.max_nodes, self.output_dim, device=device,
        )

        # 1. Encode leaves: pool decoder states → representation
        for i in range(self.max_nodes):
            leaf_mask = tree.is_leaf[:, i]  # (B,)
            if not leaf_mask.any():
                continue

            states = tree.leaf_states[:, i]   # (B, S, H)
            masks = tree.leaf_masks[:, i]     # (B, S)
            mask_f = masks.unsqueeze(-1).float()
            pooled = (states * mask_f).sum(dim=1) / mask_f.sum(dim=1).clamp(min=1)
            encoded = self.leaf_enc(pooled)   # (B, output_dim)
            depth = tree.depth[i]
            encoded = encoded + self.depth_embed(depth)
            node_repr[:, i] = encoded * leaf_mask.unsqueeze(-1).float()

        # 2. Bottom-up composition: from deepest internal nodes to root
        # Process nodes in reverse BFS order (leaves first, root last)
        for i in range(self.max_nodes - 1, -1, -1):
            lc = tree.left_child[i].item()
            rc = tree.right_child[i].item()

            # Skip leaves and inactive nodes
            is_internal = tree.active[:, i] & ~tree.is_leaf[:, i]
            if not is_internal.any() or lc == -1:
                continue

            left_repr = node_repr[:, lc]   # (B, output_dim)
            right_repr = node_repr[:, rc]  # (B, output_dim)
            merged = self.merge(torch.cat([left_repr, right_repr], dim=-1))
            depth = tree.depth[i]
            merged = merged + self.depth_embed(depth)
            node_repr[:, i] = merged * is_internal.unsqueeze(-1).float()

        # 3. Add explicit shape embedding to root
        # Compute a topology signature per sample: hash the is_leaf pattern
        # into a shape ID. This makes tree structure an independent,
        # disentangled signal separate from compositional content.
        shape_ids = self._compute_shape_ids(tree.is_leaf, tree.active)
        root = node_repr[:, 0] + self.shape_embed(shape_ids)

        return root  # (B, output_dim)

    def _compute_shape_ids(self, is_leaf: Tensor, active: Tensor) -> Tensor:
        """Hash tree topology into shape embedding indices.

        Encodes the (is_leaf, active) pattern as a binary number modulo
        max_shape_ids.  Deterministic for the same topology.

        Args:
            is_leaf: (batch, max_nodes) bool
            active: (batch, max_nodes) bool

        Returns:
            (batch,) long tensor of shape IDs.
        """
        # Encode as 2-bit per node: active*2 + is_leaf
        pattern = active.long() * 2 + is_leaf.long()  # (B, N) values in {0,1,2,3}
        # Hash: weighted sum with prime multipliers
        weights = torch.tensor(
            [31 ** i for i in range(pattern.size(1))],
            device=pattern.device, dtype=torch.long,
        )
        hashed = (pattern * weights.unsqueeze(0)).sum(dim=1)
        return hashed.abs() % self.max_shape_ids

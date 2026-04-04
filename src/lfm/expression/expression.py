"""Expression data structure for tree-structured linguistic output."""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
from torch import Tensor


@dataclass
class Expression:
    """A batch of tree-structured linguistic expressions.

    Each expression is a binary constituency tree where the topology is
    learned and leaf nodes carry latent z vectors decoded through the
    frozen LFM decoder.  The decoded output is a single continuous token
    sequence with z-switching at phrase boundaries.

    Nodes are indexed in BFS order (root=0, left child of i = 2i+1,
    right child of i = 2i+2).
    """

    # ── Tree topology ────────────────────────────────────────────────
    is_leaf: Tensor          # (batch, max_nodes) bool
    active: Tensor           # (batch, max_nodes) bool
    depth: Tensor            # (max_nodes,) int — precomputed per-node depth
    parent_idx: Tensor       # (max_nodes,) int — -1 for root
    left_child: Tensor       # (max_nodes,) int — -1 if leaf/absent
    right_child: Tensor      # (max_nodes,) int — -1 if leaf/absent
    num_active_nodes: Tensor  # (batch,) int
    num_leaves: Tensor       # (batch,) int

    # ── Leaf latent vectors ──────────────────────────────────────────
    leaf_z: Tensor           # (batch, max_nodes, latent_dim) — only leaves populated
    leaf_mu: Tensor          # (batch, max_nodes, latent_dim)
    leaf_logvar: Tensor      # (batch, max_nodes, latent_dim)

    # ── Continuous decoded output ────────────────────────────────────
    # Populated by ExpressionGenerator after continuous z-switching decode.
    tokens: Tensor = field(default_factory=lambda: torch.empty(0))
    states: Tensor = field(default_factory=lambda: torch.empty(0))
    lengths: Tensor = field(default_factory=lambda: torch.empty(0))
    mask: Tensor = field(default_factory=lambda: torch.empty(0))
    phrase_boundaries: Tensor = field(default_factory=lambda: torch.empty(0))
    # leaf_order[b, i] = node index of the i-th leaf in left-to-right order
    leaf_order: Tensor = field(default_factory=lambda: torch.empty(0))

    @property
    def batch_size(self) -> int:
        return self.active.size(0)

    @property
    def max_nodes(self) -> int:
        return self.active.size(1)

    @property
    def has_decoded(self) -> bool:
        """Whether continuous decode has been run."""
        return self.tokens.numel() > 0

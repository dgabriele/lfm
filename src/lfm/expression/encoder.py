"""ExpressionEncoder: compose a decoded expression into a fixed-size message.

Operates on the continuous decoded output from ExpressionGenerator.
Segments (defined by z-switch boundaries) are pooled individually,
then composed bottom-up following the tree topology via learned Merge —
the same operation as syntactic Merge in generative linguistics.

A shape embedding disentangles tree topology from compositional content,
allowing the receiver to use structure as an independent signal.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn

from lfm.expression.expression import Expression


class ExpressionEncoder(nn.Module):
    """Encode a decoded Expression into a fixed-size message vector.

    Pipeline:
      1. Pool each segment's decoder hidden states → segment representation.
      2. Map segments to leaf positions in the tree.
      3. Bottom-up Merge: compose children → parent, leaf to root.
      4. Add shape embedding (topology signature) to root.

    Args:
        hidden_dim: Decoder hidden state dimension.
        output_dim: Output message vector dimension.
        max_depth: Maximum tree depth (determines max nodes).
        max_shape_ids: Number of distinct topology embeddings.
    """

    def __init__(
        self,
        hidden_dim: int,
        output_dim: int,
        max_depth: int = 3,
        max_shape_ids: int = 256,
    ) -> None:
        super().__init__()
        self.max_nodes = 2 ** max_depth - 1
        self.output_dim = output_dim

        # Segment encoder: pool decoder states → segment representation
        self.segment_enc = nn.Sequential(
            nn.Linear(hidden_dim, output_dim),
            nn.GELU(),
            nn.Linear(output_dim, output_dim),
        )

        # Merge: compose two children → parent representation
        self.merge = nn.Sequential(
            nn.Linear(output_dim * 2, output_dim),
            nn.GELU(),
            nn.Linear(output_dim, output_dim),
        )

        # Depth embedding — different composition behavior at different levels
        self.depth_embed = nn.Embedding(max_depth, output_dim)

        # Shape embedding — topology signature (disentangled from content)
        self.max_shape_ids = max_shape_ids
        self.shape_embed = nn.Embedding(max_shape_ids, output_dim)

    def forward(self, expr: Expression) -> Tensor:
        """Compose expression bottom-up into a root message vector.

        Args:
            expr: Expression from ExpressionGenerator (must be decoded).

        Returns:
            (batch, output_dim) message vector.
        """
        assert expr.has_decoded, "Expression must be decoded before encoding"

        b = expr.batch_size
        device = expr.active.device

        # Node representations (filled bottom-up)
        node_repr = torch.zeros(
            b, self.max_nodes, self.output_dim, device=device,
        )

        # 1. Encode segments → leaf representations
        # Map each segment (from continuous decode) back to its leaf node.
        max_leaves = expr.leaf_order.size(1)
        for seg_idx in range(max_leaves):
            # Get segment boundaries
            seg_start = expr.segment_boundaries[:, seg_idx]  # (B,)
            if seg_idx + 1 < max_leaves:
                seg_end = expr.segment_boundaries[:, seg_idx + 1]
            else:
                seg_end = expr.lengths

            # Pool segment hidden states
            for bi in range(b):
                node_idx = expr.leaf_order[bi, seg_idx].item()
                if node_idx < 0 or not expr.is_leaf[bi, node_idx]:
                    continue

                s = seg_start[bi].item()
                e = seg_end[bi].item()
                if e <= s:
                    continue

                seg_states = expr.states[bi, s:e]  # (seg_len, H)
                seg_mask = expr.mask[bi, s:e]       # (seg_len,)
                if not seg_mask.any():
                    continue

                mask_f = seg_mask.unsqueeze(-1).float()
                pooled = (seg_states * mask_f).sum(0) / mask_f.sum(0).clamp(min=1)
                encoded = self.segment_enc(pooled.unsqueeze(0)).squeeze(0)
                depth = expr.depth[node_idx]
                encoded = encoded + self.depth_embed(depth)
                node_repr[bi, node_idx] = encoded

        # 2. Bottom-up Merge: from deepest internal nodes to root
        for i in range(self.max_nodes - 1, -1, -1):
            lc = expr.left_child[i].item()
            rc = expr.right_child[i].item()

            is_internal = expr.active[:, i] & ~expr.is_leaf[:, i]
            if not is_internal.any() or lc == -1:
                continue

            left_repr = node_repr[:, lc]
            right_repr = node_repr[:, rc]
            merged = self.merge(torch.cat([left_repr, right_repr], dim=-1))
            depth = expr.depth[i]
            merged = merged + self.depth_embed(depth)
            node_repr[:, i] = merged * is_internal.unsqueeze(-1).float()

        # 3. Add shape embedding to root
        shape_ids = self._compute_shape_ids(expr.is_leaf, expr.active)
        root = node_repr[:, 0] + self.shape_embed(shape_ids)

        return root

    def _compute_shape_ids(self, is_leaf: Tensor, active: Tensor) -> Tensor:
        """Hash tree topology into shape embedding indices.

        Deterministic: same topology → same ID.
        """
        pattern = active.long() * 2 + is_leaf.long()
        weights = torch.tensor(
            [31 ** i for i in range(pattern.size(1))],
            device=pattern.device, dtype=torch.long,
        )
        hashed = (pattern * weights.unsqueeze(0)).sum(dim=1)
        return hashed.abs() % self.max_shape_ids

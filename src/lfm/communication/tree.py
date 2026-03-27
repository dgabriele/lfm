"""Tree-structured communication for agent games.

The agent produces a tree of z vectors, each decoded through the frozen
decoder into a variable-length IPA statement.  The tree topology itself
is part of the message — different inputs produce trees of different
shapes, depths, and branching patterns.

Key design choices:
- Variable branching: categorical over 0..K children per node
- Sibling-aware generation: each child conditioned on parent + prior siblings
- Top-down level-by-level expansion (batched per depth level)
- The frozen decoder is called once per active node (unchanged)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import torch
import torch.nn.functional as F  # noqa: N812
from torch import Tensor, nn

logger = logging.getLogger(__name__)


@dataclass
class TreeMessage:
    """A batch of tree-structured messages.

    All tensors are pre-allocated to max_nodes and masked by ``active``.
    Nodes are indexed in BFS order within each depth level.

    For a tree with max_depth=D and max_children=K, max_nodes is
    computed as (K^D - 1) / (K - 1) for K > 1, or D for K = 1.
    """

    z: Tensor                    # (batch, max_nodes, latent_dim)
    mu: Tensor                   # (batch, max_nodes, latent_dim)
    logvar: Tensor               # (batch, max_nodes, latent_dim)
    active: Tensor               # (batch, max_nodes) bool
    depth: Tensor                # (max_nodes,) int — depth of each slot
    parent_idx: Tensor           # (max_nodes,) int — parent index (-1 for root)
    num_children: Tensor         # (batch, max_nodes) int — children chosen per node
    # Filled after decoding:
    decoder_states: Tensor = field(default_factory=lambda: torch.empty(0))
    token_ids: Tensor = field(default_factory=lambda: torch.empty(0))
    node_lengths: Tensor = field(default_factory=lambda: torch.empty(0))
    node_masks: Tensor = field(default_factory=lambda: torch.empty(0))


def _compute_tree_layout(
    max_depth: int, max_children: int,
) -> tuple[int, Tensor, Tensor]:
    """Precompute BFS node indices, depths, and parent pointers.

    Returns:
        (max_nodes, depth_per_slot, parent_per_slot)
    """
    # Build tree level by level
    depths: list[int] = []
    parents: list[int] = []
    nodes_at_depth: list[list[int]] = []

    idx = 0
    current_level = [idx]
    depths.append(0)
    parents.append(-1)
    nodes_at_depth.append([idx])
    idx += 1

    for d in range(1, max_depth):
        next_level = []
        level_nodes = []
        for parent in current_level:
            for _ in range(max_children):
                depths.append(d)
                parents.append(parent)
                level_nodes.append(idx)
                idx += 1
            next_level.extend(level_nodes[-max_children:])
        nodes_at_depth.append(level_nodes)
        current_level = next_level

    max_nodes = idx
    return (
        max_nodes,
        torch.tensor(depths, dtype=torch.long),
        torch.tensor(parents, dtype=torch.long),
    )


class TreeSender(nn.Module):
    """Produce a tree of z vectors from an input embedding.

    Each node's z is decoded through the frozen decoder.  Children are
    conditioned on their parent's pooled decoder hidden states AND
    previous siblings' hidden states (sibling-aware generation).

    The branching decision at each node is a categorical over
    0..max_children (0 = leaf, K = fully expand).  Trained via
    REINFORCE alongside z selection.

    Args:
        generator: Frozen MultilingualVAEGenerator (provides _decode).
        input_dim: Input embedding dimension (e.g., 384).
        latent_dim: Latent z dimension (e.g., 256).
        hidden_dim: Decoder hidden dimension (e.g., 512).
        max_depth: Maximum tree depth (root = depth 0).
        max_children: Maximum children per node.
        min_depth: Minimum forced depth before halt is allowed.
    """

    def __init__(
        self,
        generator: nn.Module,
        input_dim: int,
        latent_dim: int,
        hidden_dim: int,
        max_depth: int = 3,
        max_children: int = 3,
        min_depth: int = 1,
    ) -> None:
        super().__init__()
        self.generator = generator
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.max_depth = max_depth
        self.max_children = max_children
        self.min_depth = min_depth

        # Precompute tree layout
        self.max_nodes, depth_buf, parent_buf = _compute_tree_layout(
            max_depth, max_children,
        )
        self.register_buffer("_depth", depth_buf, persistent=False)
        self.register_buffer("_parent_idx", parent_buf, persistent=False)

        # Root projection: input → (mu, logvar) for root z
        self.root_proj = nn.Linear(input_dim, latent_dim * 2)

        # Child projection: parent_hidden + sibling_context → child (mu, logvar)
        # + branching logits (how many children: 0..max_children)
        child_input_dim = hidden_dim * 2  # parent_pooled + sibling_context
        child_output_dim = latent_dim * 2 + (max_children + 1)  # mu,logvar + branch logits
        self.child_proj = nn.Sequential(
            nn.Linear(child_input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, child_output_dim),
        )

        # Sibling aggregator: running summary of siblings at each depth
        self.sibling_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # Initialize projections near zero for stable start
        with torch.no_grad():
            self.child_proj[-1].weight.mul_(0.01)
            self.child_proj[-1].bias.zero_()

    def _pool_decoder_states(self, states: Tensor, masks: Tensor) -> Tensor:
        """Mean-pool decoder hidden states per node.

        Args:
            states: (N, seq_len, hidden_dim)
            masks: (N, seq_len) bool

        Returns:
            (N, hidden_dim) pooled vectors.
        """
        mask_f = masks.unsqueeze(-1).float()
        return (states * mask_f).sum(dim=1) / mask_f.sum(dim=1).clamp(min=1)

    def forward(self, anchor: Tensor) -> TreeMessage:
        """Generate a tree of z vectors and decode each node.

        Args:
            anchor: (batch, input_dim) input embeddings.

        Returns:
            TreeMessage with all node z vectors, decoded states, and topology.
        """
        b = anchor.size(0)
        device = anchor.device

        # Allocate buffers
        all_z = torch.zeros(b, self.max_nodes, self.latent_dim, device=device)
        all_mu = torch.zeros_like(all_z)
        all_logvar = torch.zeros_like(all_z)
        all_active = torch.zeros(b, self.max_nodes, dtype=torch.bool, device=device)
        all_num_children = torch.zeros(b, self.max_nodes, dtype=torch.long, device=device)

        # Decoder output buffers (filled as we decode)
        max_seq = self.generator._max_output_len
        all_dec_states = torch.zeros(b, self.max_nodes, max_seq, self.hidden_dim, device=device)
        all_token_ids = torch.zeros(b, self.max_nodes, max_seq, dtype=torch.long, device=device)
        all_node_lengths = torch.zeros(b, self.max_nodes, dtype=torch.long, device=device)
        all_node_masks = torch.zeros(b, self.max_nodes, max_seq, dtype=torch.bool, device=device)

        # --- Root node (always active) ---
        h_root = self.root_proj(anchor)  # (B, latent_dim * 2)
        mu_root, logvar_root = h_root.chunk(2, dim=-1)
        z_root = self._reparameterize(mu_root, logvar_root)

        all_z[:, 0] = z_root
        all_mu[:, 0] = mu_root
        all_logvar[:, 0] = logvar_root
        all_active[:, 0] = True

        # Decode root
        self._decode_nodes(z_root, 0, all_dec_states, all_token_ids,
                           all_node_lengths, all_node_masks)

        # --- Level-by-level expansion ---
        # Track which node indices belong to each depth level
        node_idx = 1  # next available slot (0 = root)

        for d in range(self.max_depth - 1):
            # Find active parents at depth d
            parent_slots = (self._depth == d).nonzero(as_tuple=True)[0]
            active_parents = all_active[:, parent_slots]  # (B, num_parents_at_d)

            if not active_parents.any():
                break

            # Pool each parent's decoder states
            parent_states = all_dec_states[:, parent_slots]  # (B, P, S, H)
            parent_masks = all_node_masks[:, parent_slots]   # (B, P, S)

            # Flatten for pooling: (B*P, S, H)
            bp = b * parent_slots.size(0)
            parent_pooled = self._pool_decoder_states(
                parent_states.reshape(bp, max_seq, self.hidden_dim),
                parent_masks.reshape(bp, max_seq),
            ).reshape(b, parent_slots.size(0), self.hidden_dim)  # (B, P, H)

            # For each parent, decide branching and generate children
            for p_idx_local, p_slot in enumerate(parent_slots):
                parent_h = parent_pooled[:, p_idx_local]  # (B, H)
                parent_active = all_active[:, p_slot]      # (B,)

                # Sibling context starts as zeros (first child has no siblings)
                sibling_ctx = torch.zeros(b, self.hidden_dim, device=device)

                # Determine branching (how many children)
                child_input = torch.cat([parent_h, sibling_ctx], dim=-1)  # (B, 2H)
                child_out = self.child_proj(child_input)  # (B, latent*2 + K+1)

                branch_logits = child_out[:, -self.max_children - 1:]  # (B, K+1)

                # Force expansion at shallow depths
                if d < self.min_depth - 1:
                    # Mask out 0-children option
                    branch_logits[:, 0] = float("-inf")

                # Sample number of children (categorical)
                if self.training:
                    branch_probs = F.softmax(branch_logits, dim=-1)
                    n_children = torch.multinomial(branch_probs, 1).squeeze(-1)
                else:
                    n_children = branch_logits.argmax(dim=-1)

                n_children = n_children * parent_active.long()  # inactive parents get 0
                all_num_children[:, p_slot] = n_children

                # Generate each child
                for c in range(self.max_children):
                    if node_idx >= self.max_nodes:
                        break

                    child_active = parent_active & (n_children > c)

                    if not child_active.any():
                        node_idx += 1
                        continue

                    # Child z from child_proj (conditioned on parent + siblings)
                    child_input = torch.cat([parent_h, sibling_ctx], dim=-1)
                    child_out = self.child_proj(child_input)
                    mu_c = child_out[:, :self.latent_dim]
                    logvar_c = child_out[:, self.latent_dim:self.latent_dim * 2]
                    z_c = self._reparameterize(mu_c, logvar_c)

                    all_z[:, node_idx] = z_c
                    all_mu[:, node_idx] = mu_c
                    all_logvar[:, node_idx] = logvar_c
                    all_active[:, node_idx] = child_active

                    # Decode this child
                    self._decode_nodes(
                        z_c * child_active.unsqueeze(-1).float(),
                        node_idx, all_dec_states, all_token_ids,
                        all_node_lengths, all_node_masks,
                    )

                    # Update sibling context with this child's pooled states
                    child_pooled = self._pool_decoder_states(
                        all_dec_states[:, node_idx],
                        all_node_masks[:, node_idx],
                    )  # (B, H)
                    sibling_ctx = self.sibling_proj(
                        sibling_ctx + child_pooled * child_active.unsqueeze(-1).float()
                    )

                    node_idx += 1

        return TreeMessage(
            z=all_z,
            mu=all_mu,
            logvar=all_logvar,
            active=all_active,
            depth=self._depth,
            parent_idx=self._parent_idx,
            num_children=all_num_children,
            decoder_states=all_dec_states,
            token_ids=all_token_ids,
            node_lengths=all_node_lengths,
            node_masks=all_node_masks,
        )

    def _decode_nodes(
        self,
        z: Tensor,
        slot: int,
        dec_states: Tensor,
        token_ids: Tensor,
        lengths: Tensor,
        masks: Tensor,
    ) -> None:
        """Decode z through the frozen generator and store in buffers."""
        with torch.no_grad():
            tids, _probs, states, lens, omask = self.generator._decode(z)

        dec_states[:, slot, :states.size(1), :] = states
        token_ids[:, slot, :tids.size(1)] = tids
        lengths[:, slot] = lens
        masks[:, slot, :omask.size(1)] = omask

    def _reparameterize(self, mu: Tensor, logvar: Tensor) -> Tensor:
        """Sample z via reparameterization trick."""
        if self.training:
            std = (0.5 * logvar).exp()
            return mu + std * torch.randn_like(std)
        return mu


class TreeMessageEncoder(nn.Module):
    """Encode a tree of decoded node states into a single message vector.

    Uses tree-positional encoding (depth + BFS position) with a small
    transformer for cross-node attention, plus edge features encoding
    parent-child relationships.

    Args:
        hidden_dim: Decoder hidden dimension.
        output_dim: Output message vector dimension.
        max_nodes: Maximum nodes in the tree.
        max_depth: Maximum tree depth.
        num_heads: Transformer attention heads.
        num_layers: Transformer layers.
        dropout: Dropout rate.
    """

    def __init__(
        self,
        hidden_dim: int,
        output_dim: int,
        max_nodes: int,
        max_depth: int,
        num_heads: int = 4,
        num_layers: int = 2,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.max_nodes = max_nodes

        # Node projection
        self.node_proj = nn.Linear(hidden_dim, output_dim)

        # Tree-positional embeddings
        self.depth_embed = nn.Embedding(max_depth, output_dim)
        self.position_embed = nn.Embedding(max_nodes, output_dim)

        # Edge features: encode parent-child relationship
        self.edge_proj = nn.Sequential(
            nn.Linear(output_dim * 2, output_dim),
            nn.GELU(),
            nn.Linear(output_dim, output_dim),
        )

        # Transformer for cross-node attention
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=output_dim, nhead=num_heads,
            batch_first=True, dropout=dropout,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers,
        )

        # Aggregation via learned query
        self.agg_query = nn.Parameter(torch.randn(1, 1, output_dim) * 0.02)
        self.agg_attn = nn.MultiheadAttention(
            output_dim, num_heads=num_heads, batch_first=True,
        )

    def forward(self, tree: TreeMessage) -> Tensor:
        """Encode tree into a single message vector.

        Args:
            tree: TreeMessage from TreeSender.

        Returns:
            (batch, output_dim) message vector.
        """
        b = tree.active.size(0)
        device = tree.active.device

        # 1. Pool each node's decoder states
        states = tree.decoder_states  # (B, N, S, H)
        masks = tree.node_masks       # (B, N, S)
        mask_f = masks.unsqueeze(-1).float()
        node_pooled = (
            (states * mask_f).sum(dim=2) / mask_f.sum(dim=2).clamp(min=1)
        )  # (B, N, H)

        # 2. Project + tree positional encoding
        node_repr = self.node_proj(node_pooled)  # (B, N, output_dim)
        node_repr = node_repr + self.depth_embed(tree.depth).unsqueeze(0)
        positions = torch.arange(self.max_nodes, device=device)
        node_repr = node_repr + self.position_embed(positions).unsqueeze(0)

        # 3. Add edge features (parent-child relationship encoding)
        parent_idx = tree.parent_idx  # (N,)
        for i in range(1, self.max_nodes):
            p = parent_idx[i].item()
            if p >= 0:
                edge_input = torch.cat([
                    node_repr[:, p, :], node_repr[:, i, :],
                ], dim=-1)  # (B, 2*output_dim)
                edge_feat = self.edge_proj(edge_input)  # (B, output_dim)
                node_repr[:, i] = node_repr[:, i] + edge_feat

        # 4. Transformer with inactive nodes masked
        src_key_padding_mask = ~tree.active  # (B, N)
        encoded = self.transformer(
            node_repr, src_key_padding_mask=src_key_padding_mask,
        )

        # 5. Aggregate via learned query
        query = self.agg_query.expand(b, -1, -1)
        agg, _ = self.agg_attn(
            query, encoded, encoded,
            key_padding_mask=src_key_padding_mask,
        )
        return agg.squeeze(1)  # (B, output_dim)

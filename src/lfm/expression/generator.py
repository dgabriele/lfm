"""ExpressionGenerator: learn tree topology and decode via continuous z-switching.

Unifies topology generation (learned expand/leaf decisions) with continuous
autoregressive decoding through the frozen LFM decoder.  The KV cache
persists across leaf z-switch boundaries, producing phonotactically
coherent output.

    anchor embedding
      → root context MLP
      → top-down tree expansion (stochastic expand/leaf via REINFORCE)
      → leaf z projection (μ, σ) → reparameterize
      → continuous AR decode: z₁ → z₂ → z₃ (KV cache carries through)
      → Expression
"""

from __future__ import annotations

import logging

import torch
from torch import Tensor, nn

from lfm.expression.expression import Expression

logger = logging.getLogger(__name__)


def _max_nodes_for_depth(max_depth: int) -> int:
    """Max nodes in a complete binary tree of given depth."""
    return 2 ** max_depth - 1


class ExpressionGenerator(nn.Module):
    """Generate tree-structured expressions with continuous AR decoding.

    The generator learns two things via REINFORCE:
      1. Tree topology — when to expand (create children) vs. stop (leaf).
      2. Leaf content — z vectors decoded through the frozen decoder.

    Decoding uses continuous z-switching: one AR pass where the
    cross-attention memory switches between leaf z vectors at phrase
    transitions.  The KV cache carries across boundaries, enabling
    natural coarticulation.

    Args:
        generator: Frozen MultilingualVAEGenerator (provides decoder).
        input_dim: Dimension of input anchor embeddings.
        latent_dim: Latent z dimension per leaf.
        hidden_dim: Internal hidden dimension for tree decisions.
        max_depth: Maximum tree depth (root = 0).
        min_depth: Minimum forced expansion depth.
        max_tokens_per_leaf: Max tokens per phrase before z-switch.
        transition_on_eos: Switch z when decoder emits EOS.
    """

    def __init__(
        self,
        generator: nn.Module,
        input_dim: int,
        latent_dim: int,
        hidden_dim: int,
        max_depth: int = 3,
        min_depth: int = 1,
        max_tokens_per_leaf: int = 96,
        transition_on_eos: bool = True,
    ) -> None:
        super().__init__()
        self.generator = generator
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.max_depth = max_depth
        self.min_depth = min_depth
        self.max_tokens_per_leaf = max_tokens_per_leaf
        self.transition_on_eos = transition_on_eos
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
        self.expand_head = nn.Linear(hidden_dim, 1 + hidden_dim * 2)

        # Leaf z projection: hidden → (mu, logvar)
        self.leaf_proj = nn.Linear(hidden_dim, latent_dim * 2)

        # Initialize for stable start
        with torch.no_grad():
            self.expand_head.weight.mul_(0.01)
            self.expand_head.bias.zero_()
            self.expand_head.bias.data[0] = 1.0  # bias toward expanding

    def forward(self, anchor: Tensor) -> Expression:
        """Generate an expression: tree topology + continuous AR decode.

        Args:
            anchor: (batch, input_dim) input embeddings.

        Returns:
            Expression with topology, leaf z, and decoded output.
        """
        expr = self._generate_topology(anchor)
        self._continuous_decode(expr)
        return expr

    def _generate_topology(self, anchor: Tensor) -> Expression:
        """Top-down tree generation: decide expand vs. leaf at each node.

        Args:
            anchor: (batch, input_dim) input embeddings.

        Returns:
            Expression with topology and leaf z (not yet decoded).
        """
        b = anchor.size(0)
        device = anchor.device

        active = torch.zeros(b, self.max_nodes, dtype=torch.bool, device=device)
        is_leaf = torch.zeros(b, self.max_nodes, dtype=torch.bool, device=device)
        leaf_z = torch.zeros(b, self.max_nodes, self.latent_dim, device=device)
        leaf_mu = torch.zeros_like(leaf_z)
        leaf_logvar = torch.zeros_like(leaf_z)

        # Hidden context per node (dict to avoid in-place tensor ops)
        node_ctx: dict[int, Tensor] = {}

        # Root
        active[:, 0] = True
        node_ctx[0] = self.root_proj(anchor)

        # Top-down expansion
        for i in range(self.max_nodes):
            depth = self._depth[i].item()
            lc = self._left[i].item()
            rc = self._right[i].item()

            node_active = active[:, i]
            if not node_active.any():
                continue

            ctx = node_ctx.get(
                i, torch.zeros(b, self.hidden_dim, device=device),
            )

            # At max depth or no children possible: must be leaf
            if lc == -1 or depth >= self.max_depth - 1:
                is_leaf[:, i] = node_active
                self._project_leaf(
                    ctx, node_active, leaf_z, leaf_mu, leaf_logvar, i,
                )
                continue

            # Decide: expand or leaf?
            head_out = self.expand_head(ctx)
            expand_logit = head_out[:, 0]
            left_ctx = head_out[:, 1 : 1 + self.hidden_dim]
            right_ctx = head_out[:, 1 + self.hidden_dim :]

            if depth < self.min_depth:
                expand = node_active
            elif self.training:
                expand_prob = torch.sigmoid(expand_logit)
                expand = (torch.rand_like(expand_prob) < expand_prob) & node_active
            else:
                expand = (expand_logit > 0) & node_active

            # Expanding nodes → activate children
            if lc < self.max_nodes:
                active[:, lc] = expand
                node_ctx[lc] = left_ctx * expand.unsqueeze(-1).float()
            if rc < self.max_nodes:
                active[:, rc] = expand
                node_ctx[rc] = right_ctx * expand.unsqueeze(-1).float()

            # Non-expanding nodes → become leaves
            not_expand = node_active & ~expand
            is_leaf[:, i] = not_expand
            if not_expand.any():
                self._project_leaf(
                    ctx, not_expand, leaf_z, leaf_mu, leaf_logvar, i,
                )

        return Expression(
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
        )

    def _project_leaf(
        self,
        ctx: Tensor,
        mask: Tensor,
        leaf_z: Tensor,
        leaf_mu: Tensor,
        leaf_logvar: Tensor,
        node_idx: int,
    ) -> None:
        """Project hidden context to leaf z via reparameterization."""
        h = self.leaf_proj(ctx)
        mu, logvar = h.chunk(2, dim=-1)
        z = self._reparameterize(mu, logvar)
        m = mask.unsqueeze(-1).float()
        leaf_z[:, node_idx] = z * m
        leaf_mu[:, node_idx] = mu * m
        leaf_logvar[:, node_idx] = logvar * m

    def _reparameterize(self, mu: Tensor, logvar: Tensor) -> Tensor:
        if self.training:
            std = (0.5 * logvar).exp()
            return mu + std * torch.randn_like(std)
        return mu

    @torch.no_grad()
    def _continuous_decode(self, expr: Expression) -> None:
        """Continuous AR decode with z-switching at leaf boundaries.

        Populates expr.tokens, expr.states, expr.lengths, expr.mask,
        expr.phrase_boundaries, and expr.leaf_order in-place.
        """
        from lfm.generator.layers import PhraseDecoder

        gen = self.generator
        device = expr.leaf_z.device
        b = expr.batch_size

        # Collect leaves in tree-order (left-to-right in-order traversal)
        leaf_order = self._inorder_leaves(expr)
        max_leaves = leaf_order.size(1)
        expr.leaf_order = leaf_order

        # Gather z vectors in leaf order
        leaf_z_ordered = torch.zeros(
            b, max_leaves, gen._latent_dim, device=device,
        )
        leaf_active = torch.zeros(
            b, max_leaves, dtype=torch.bool, device=device,
        )
        for i in range(max_leaves):
            node_idx = leaf_order[:, i]
            for bi in range(b):
                ni = node_idx[bi].item()
                if ni >= 0 and expr.is_leaf[bi, ni]:
                    leaf_z_ordered[bi, i] = expr.leaf_z[bi, ni]
                    leaf_active[bi, i] = True

        # Precompute memory vectors: z → latent_to_decoder → (B, L, K, H)
        _n_mem = getattr(gen, "_num_memory_tokens", 1)
        memories = gen.latent_to_decoder(
            leaf_z_ordered.reshape(b * max_leaves, -1),
        ).reshape(b, max_leaves, _n_mem, -1)

        # Decoder components
        decoder = gen.decoder
        dec_tok = gen.token_embedding
        output_head = gen.output_head
        rope_freqs = gen._rope_freqs
        bos_id = gen.bos_id
        eos_id = gen.eos_id
        hidden_dim = gen.config.decoder_hidden_dim

        max_total = self.max_tokens_per_leaf * max_leaves
        is_linguistic = isinstance(decoder, PhraseDecoder)

        # Precompute causal mask if needed
        if gen._full_causal_mask is None or gen._full_causal_mask.size(1) < max_total + 1:
            from lfm.generator.layers import multiscale_causal_mask

            gen._full_causal_mask = multiscale_causal_mask(
                max_total + 1,
                num_heads=gen.config.decoder_num_heads,
                head_windows=gen.config.attention_head_windows,
                global_every=gen.config.attention_global_every,
                device=device,
            )

        # Output buffers
        all_tokens = torch.zeros(b, max_total, dtype=torch.long, device=device)
        all_states = torch.zeros(b, max_total, hidden_dim, device=device)
        all_mask = torch.zeros(b, max_total, dtype=torch.bool, device=device)
        phrase_starts = torch.zeros(b, max_leaves, dtype=torch.long, device=device)

        # Per-sample state
        cur_leaf = torch.zeros(b, dtype=torch.long, device=device)
        tokens_in_seg = torch.zeros(b, dtype=torch.long, device=device)
        finished = torch.zeros(b, dtype=torch.bool, device=device)
        total_pos = torch.zeros(b, dtype=torch.long, device=device)

        # KV cache — continuous across z-switch boundaries
        if is_linguistic:
            kv_cache = decoder.make_kv_cache(
                b, max_total + 1, device, dtype=torch.float16,
            )

        # BOS
        cur_ids = torch.full((b, 1), bos_id, dtype=torch.long, device=device)
        cur_embed = dec_tok(cur_ids)

        def _get_memory() -> Tensor:
            mem = torch.zeros(b, _n_mem, hidden_dim, device=device)
            for bi in range(b):
                li = cur_leaf[bi].item()
                if li < max_leaves and leaf_active[bi, li]:
                    mem[bi] = memories[bi, li]
            return mem

        memory = _get_memory()

        # Prime KV cache with BOS
        if is_linguistic:
            mask_row = gen._full_causal_mask[:, 0:1, 0:1]
            out = decoder.forward_cached(
                cur_embed, memory, kv_cache,
                rope_freqs=rope_freqs, tgt_mask_row=mask_row,
            )
            kv_cache.advance()
        else:
            tgt_mask = nn.Transformer.generate_square_subsequent_mask(
                1, device=device,
            )
            out = decoder(cur_embed, memory, tgt_mask=tgt_mask)

        # Autoregressive loop with z-switching
        for t in range(max_total):
            logits = output_head(out[:, -1])
            next_token = logits.argmax(dim=-1)

            # Store output
            for bi in range(b):
                if not finished[bi]:
                    pos = total_pos[bi].item()
                    all_tokens[bi, pos] = next_token[bi]
                    all_states[bi, pos] = out[bi, -1]
                    all_mask[bi, pos] = True
                    total_pos[bi] += 1
                    tokens_in_seg[bi] += 1

            # Check for phrase transitions (z-switch points)
            for bi in range(b):
                if finished[bi]:
                    continue

                should_switch = False
                if self.transition_on_eos and next_token[bi] == eos_id:
                    should_switch = True
                if tokens_in_seg[bi] >= self.max_tokens_per_leaf:
                    should_switch = True

                if should_switch:
                    cur_leaf[bi] += 1
                    tokens_in_seg[bi] = 0
                    if cur_leaf[bi] < max_leaves:
                        li = cur_leaf[bi].item()
                        phrase_starts[bi, li] = total_pos[bi]
                    cl = min(cur_leaf[bi].item(), max_leaves - 1)
                    if cur_leaf[bi] >= max_leaves or not leaf_active[bi, cl]:
                        finished[bi] = True

            if finished.all():
                break

            # Update memory for z-switched samples
            memory = _get_memory()

            # Next decoder step (KV cache carries through)
            new_embed = dec_tok(next_token.unsqueeze(1))
            if is_linguistic:
                seq_so_far = kv_cache.seq_len + 1
                mask_row = gen._full_causal_mask[
                    :, kv_cache.seq_len : kv_cache.seq_len + 1, :seq_so_far
                ]
                out = decoder.forward_cached(
                    new_embed, memory, kv_cache,
                    rope_freqs=rope_freqs, tgt_mask_row=mask_row,
                )
                kv_cache.advance()
            else:
                all_embed = dec_tok(
                    torch.cat([cur_ids, next_token.unsqueeze(1)], dim=1),
                )
                cur_ids = torch.cat(
                    [cur_ids, next_token.unsqueeze(1)], dim=1,
                )
                tgt_mask = nn.Transformer.generate_square_subsequent_mask(
                    cur_ids.size(1), device=device,
                )
                out = decoder(all_embed, memory, tgt_mask=tgt_mask)

        # Populate Expression fields
        expr.tokens = all_tokens
        expr.states = all_states
        expr.lengths = total_pos
        expr.mask = all_mask
        expr.phrase_boundaries = phrase_starts

    @staticmethod
    def _inorder_leaves(expr: Expression) -> Tensor:
        """Get leaf node indices in left-to-right (in-order) traversal.

        Returns:
            (batch, max_leaves) tensor of node indices. -1 for padding.
        """
        b = expr.batch_size
        max_nodes = expr.max_nodes

        def _inorder(node: int, order: list[int]) -> None:
            if node >= max_nodes:
                return
            _inorder(2 * node + 1, order)
            order.append(node)
            _inorder(2 * node + 2, order)

        traversal: list[int] = []
        _inorder(0, traversal)

        max_leaves = expr.is_leaf.sum(dim=1).max().item()
        result = torch.full(
            (b, max_leaves), -1,
            dtype=torch.long, device=expr.active.device,
        )

        for bi in range(b):
            li = 0
            for node in traversal:
                if node < max_nodes and expr.is_leaf[bi, node]:
                    result[bi, li] = node
                    li += 1
                    if li >= max_leaves:
                        break

        return result

"""Continuous z-switching tree decode for phonotactically coherent output.

Instead of decoding each leaf independently (producing disjoint IPA
islands), this module performs ONE continuous autoregressive decode
where the cross-attention memory switches between leaf z vectors at
transition points.  The decoder's KV cache carries across z boundaries,
producing natural coarticulation, sandhi, and prosodic bridging.

    Tree:        ○
                / \\
               ○   z₃
              / \\
            z₁   z₂

    Decode:  [BOS ðʌ kwɪk braʊn | fɑks dʒʌmpt | oʊvɝ ðʌ leɪzi dɑɡ EOS]
                  memory=z₁       memory=z₂     memory=z₃
                  (continuous KV cache — no breaks)

The tree determines: leaf order (in-order traversal), segment count,
transition points (per-segment EOS or max tokens).  The receiver still
sees tree structure via Merge composition.
"""

from __future__ import annotations

import logging

import torch
from torch import Tensor, nn

from lfm.communication.tree import TreeMessage

logger = logging.getLogger(__name__)


class ContinuousTreeDecoder(nn.Module):
    """Decode a tree's leaf z vectors as one continuous IPA sequence.

    Wraps the frozen generator's decoder and manages z-switching
    mid-sequence.  The KV cache is continuous across segment boundaries,
    producing phonotactically coherent output.

    Args:
        generator: Frozen MultilingualVAEGenerator.
        max_tokens_per_leaf: Max tokens before forcing z-switch.
        transition_on_eos: Switch z when decoder emits EOS for a segment.
    """

    def __init__(
        self,
        generator: nn.Module,
        max_tokens_per_leaf: int = 96,
        transition_on_eos: bool = True,
    ) -> None:
        super().__init__()
        self.generator = generator
        self.max_tokens_per_leaf = max_tokens_per_leaf
        self.transition_on_eos = transition_on_eos

    @torch.no_grad()
    def decode(self, tree: TreeMessage) -> dict[str, Tensor]:
        """Continuous decode through the tree's leaf z vectors.

        Args:
            tree: TreeMessage with leaf z vectors and topology.

        Returns:
            Dict with:
              - tokens: (batch, total_seq_len) token IDs
              - states: (batch, total_seq_len, hidden_dim) decoder states
              - lengths: (batch,) total valid tokens
              - mask: (batch, total_seq_len) bool
              - segment_boundaries: (batch, max_leaves) int — token position
                where each leaf's segment starts
        """
        from lfm.generator.layers import LinguisticDecoder

        gen = self.generator
        device = tree.leaf_z.device
        b = tree.active.size(0)

        # Collect leaf z vectors in tree-order (left-to-right in-order traversal)
        leaf_order = self._inorder_leaves(tree)  # (batch, max_leaves) indices
        max_leaves = leaf_order.size(1)

        # Prepare memories for each leaf
        # leaf_order[b, i] gives the node index of the i-th leaf for sample b
        # Gather z vectors in leaf order
        leaf_z_ordered = torch.zeros(
            b, max_leaves, gen._latent_dim, device=device,
        )
        leaf_active = torch.zeros(b, max_leaves, dtype=torch.bool, device=device)
        for i in range(max_leaves):
            node_idx = leaf_order[:, i]  # (B,)
            for bi in range(b):
                ni = node_idx[bi].item()
                if ni >= 0 and tree.is_leaf[bi, ni]:
                    leaf_z_ordered[bi, i] = tree.leaf_z[bi, ni]
                    leaf_active[bi, i] = True

        # Precompute memories: z → latent_to_decoder → (B, 1, H)
        memories = gen.latent_to_decoder(
            leaf_z_ordered.reshape(b * max_leaves, -1)
        ).reshape(b, max_leaves, 1, -1)  # (B, L, 1, H)

        # Decoder components
        decoder = gen.decoder
        dec_tok = gen.token_embedding
        output_head = gen.output_head
        rope_freqs = gen._rope_freqs
        bos_id = gen.bos_id
        eos_id = gen.eos_id
        hidden_dim = gen.config.decoder_hidden_dim

        max_total = self.max_tokens_per_leaf * max_leaves
        is_linguistic = isinstance(decoder, LinguisticDecoder)

        # Precompute causal mask
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
        segment_starts = torch.zeros(b, max_leaves, dtype=torch.long, device=device)

        # Per-sample state tracking
        cur_leaf = torch.zeros(b, dtype=torch.long, device=device)  # current leaf index
        tokens_in_seg = torch.zeros(b, dtype=torch.long, device=device)
        finished = torch.zeros(b, dtype=torch.bool, device=device)
        total_pos = torch.zeros(b, dtype=torch.long, device=device)

        # KV cache — continuous across segments
        if is_linguistic:
            kv_cache = decoder.make_kv_cache(
                b, max_total + 1, device, dtype=torch.float16,
            )

        # BOS token
        cur_ids = torch.full((b, 1), bos_id, dtype=torch.long, device=device)
        cur_embed = dec_tok(cur_ids)

        # Get current memory for each sample
        def _get_memory() -> Tensor:
            """(B, 1, H) — current leaf's memory per sample."""
            mem = torch.zeros(b, 1, hidden_dim, device=device)
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
            tgt_mask = nn.Transformer.generate_square_subsequent_mask(1, device=device)
            out = decoder(cur_embed, memory, tgt_mask=tgt_mask)

        # Autoregressive loop
        for t in range(max_total):
            logits = output_head(out[:, -1])  # (B, V)
            next_token = logits.argmax(dim=-1)  # greedy

            # Store
            for bi in range(b):
                if not finished[bi]:
                    pos = total_pos[bi].item()
                    all_tokens[bi, pos] = next_token[bi]
                    all_states[bi, pos] = out[bi, -1]
                    all_mask[bi, pos] = True
                    total_pos[bi] += 1
                    tokens_in_seg[bi] += 1

            # Check for segment transitions
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
                        segment_starts[bi, li] = total_pos[bi]
                    cl = min(cur_leaf[bi].item(), max_leaves - 1)
                    if cur_leaf[bi] >= max_leaves or not leaf_active[bi, cl]:
                        finished[bi] = True

            if finished.all():
                break

            # Update memory for switched samples
            memory = _get_memory()

            # Next step
            new_embed = dec_tok(next_token.unsqueeze(1))
            if is_linguistic:
                seq_so_far = kv_cache.seq_len + 1
                mask_row = gen._full_causal_mask[
                    :, kv_cache.seq_len:kv_cache.seq_len + 1, :seq_so_far
                ]
                out = decoder.forward_cached(
                    new_embed, memory, kv_cache,
                    rope_freqs=rope_freqs, tgt_mask_row=mask_row,
                )
                kv_cache.advance()
            else:
                # Fallback — no cache
                all_embed = dec_tok(
                    torch.cat([cur_ids, next_token.unsqueeze(1)], dim=1)
                )
                cur_ids = torch.cat([cur_ids, next_token.unsqueeze(1)], dim=1)
                tgt_mask = nn.Transformer.generate_square_subsequent_mask(
                    cur_ids.size(1), device=device,
                )
                out = decoder(all_embed, memory, tgt_mask=tgt_mask)

        return {
            "tokens": all_tokens,
            "states": all_states,
            "lengths": total_pos,
            "mask": all_mask,
            "segment_boundaries": segment_starts,
        }

    @staticmethod
    def _inorder_leaves(tree: TreeMessage) -> Tensor:
        """Get leaf node indices in left-to-right (in-order) traversal.

        Returns:
            (batch, max_leaves) tensor of node indices. -1 for padding.
        """
        b = tree.active.size(0)
        max_nodes = tree.active.size(1)

        # For a BFS-indexed binary tree, in-order traversal gives
        # left-to-right leaf ordering.
        def _inorder(node: int, order: list[int]) -> None:
            if node >= max_nodes:
                return
            left = 2 * node + 1
            right = 2 * node + 2
            _inorder(left, order)
            order.append(node)
            _inorder(right, order)

        traversal: list[int] = []
        _inorder(0, traversal)

        # For each sample, collect only the leaf nodes in traversal order
        max_leaves = tree.is_leaf.sum(dim=1).max().item()
        result = torch.full((b, max_leaves), -1, dtype=torch.long, device=tree.active.device)

        for bi in range(b):
            li = 0
            for node in traversal:
                if node < max_nodes and tree.is_leaf[bi, node]:
                    result[bi, li] = node
                    li += 1
                    if li >= max_leaves:
                        break

        return result

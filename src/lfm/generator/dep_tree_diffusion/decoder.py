"""Tree-structured diffusion decoder for dependency-parsed content generation.

Generates all content tokens simultaneously via iterative denoising,
conditioned on the dependency skeleton and z_content. The noise schedule
follows tree depth: root-level tokens are denoised first (stable
structure), leaf-level tokens last (fine detail).

This eliminates the autoregressive repetition loops that plague
sequential decoders, since each position is resolved independently
conditioned on z and already-resolved ancestors.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from lfm.generator.dep_tree_vae.config import NUM_DEP_RELATIONS

# Empirical per-role mean token counts (from 50K sample analysis)
ROLE_MEAN_TOKENS = {
    "det": 1.0, "cc": 1.0, "aux": 1.0, "cop": 1.0, "aux:pass": 1.0,
    "case": 1.1, "mark": 1.1, "expl": 1.1,
    "nsubj": 1.6, "nsubj:pass": 1.6, "nmod:poss": 1.1,
    "root": 2.0, "obj": 2.1, "obl": 2.1, "advmod": 2.0,
    "amod": 2.6, "compound": 2.4, "nmod": 2.4, "conj": 2.4,
    "advcl": 2.1, "xcomp": 2.0, "acl": 2.3, "acl:relcl": 1.9,
    "ccomp": 2.0, "csubj": 2.0, "flat": 2.0,
}
_DEFAULT_MEAN = 1.8


class RoleLengthPredictor(nn.Module):
    """Predict number of content tokens per dependency role.

    A small transformer that sees the full skeleton context (all roles
    in the sentence) cross-attending to z. Each role's predicted length
    is conditioned on neighboring roles and the semantic content of z.

    Trained with Poisson NLL against actual token counts from data.
    At generation time, predicts per-role lengths to allocate positions.
    """

    def __init__(
        self, z_dim: int, hidden_dim: int = 128, num_heads: int = 4,
        num_layers: int = 2, max_len: int = 12,
    ) -> None:
        super().__init__()
        self.role_embedding = nn.Embedding(NUM_DEP_RELATIONS, hidden_dim)
        self.pos_embedding = nn.Embedding(64, hidden_dim)
        self.z_proj = nn.Linear(z_dim, hidden_dim)

        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=hidden_dim, nhead=num_heads,
                dim_feedforward=hidden_dim * 2,
                dropout=0.1, batch_first=True, norm_first=True,
            )
            for _ in range(num_layers)
        ])
        self.out_head = nn.Linear(hidden_dim, 1)
        self.max_len = max_len

    def forward(self, z: Tensor, role_ids: Tensor, role_mask: Tensor | None = None) -> Tensor:
        """Predict log-rate per role, conditioned on full skeleton + z.

        Args:
            z: (B, z_dim) latent vector.
            role_ids: (B, R) role ids for the skeleton.
            role_mask: (B, R) True = valid role.

        Returns:
            log_rate: (B, R) predicted log token count.
        """
        b, r = role_ids.shape
        device = role_ids.device

        pos = torch.arange(r, device=device).unsqueeze(0)
        h = self.role_embedding(role_ids.clamp(max=NUM_DEP_RELATIONS - 1))
        h = h + self.pos_embedding(pos.clamp(max=63))
        h = h + self.z_proj(z).unsqueeze(1)

        padding_mask = ~role_mask if role_mask is not None else None
        for layer in self.layers:
            h = layer(h, src_key_padding_mask=padding_mask)

        return self.out_head(h).squeeze(-1)

    def predict_lengths(self, z: Tensor, role_ids: Tensor, role_mask: Tensor | None = None) -> Tensor:
        """Predict integer lengths for generation."""
        log_rate = self.forward(z, role_ids, role_mask)
        return log_rate.exp().round().long().clamp(min=1, max=self.max_len)

    def loss(
        self, z: Tensor, role_ids: Tensor,
        target_lengths: Tensor, role_mask: Tensor,
    ) -> Tensor:
        """Poisson NLL loss against actual token counts per role."""
        log_rate = self.forward(z, role_ids, role_mask)
        rate = log_rate.exp().clamp(min=1e-6)
        nll = rate - target_lengths.float() * torch.log(rate + 1e-8)
        return (nll * role_mask.float()).sum() / role_mask.float().sum().clamp(min=1)


class TimestepEmbedding(nn.Module):
    """Sinusoidal timestep embedding → MLP projection."""

    def __init__(self, d_model: int) -> None:
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.GELU(),
            nn.Linear(d_model * 2, d_model),
        )
        self.d_model = d_model

    def forward(self, t: Tensor) -> Tensor:
        """t: (B,) or (B, S) float in [0, 1] → (B, d_model) or (B, S, d_model)."""
        half = self.d_model // 2
        freqs = torch.exp(
            -math.log(10000.0) * torch.arange(half, device=t.device).float() / half
        )
        if t.dim() == 1:
            args = t.unsqueeze(-1) * freqs.unsqueeze(0)
        else:
            args = t.unsqueeze(-1) * freqs
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        return self.proj(emb)


class TreeDenoiserBlock(nn.Module):
    """Transformer block for tree-structured denoising.

    Self-attention between all token positions (bidirectional — not causal,
    since we're denoising simultaneously). Cross-attention to z_content
    memory. Timestep conditioning via adaptive layer norm.
    """

    def __init__(self, d_model: int, num_heads: int = 8, dropout: float = 0.1) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.self_attn = nn.MultiheadAttention(
            d_model, num_heads, dropout=dropout, batch_first=True,
        )
        self.norm2 = nn.LayerNorm(d_model)
        self.cross_attn = nn.MultiheadAttention(
            d_model, num_heads, dropout=dropout, batch_first=True,
        )
        self.norm3 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model),
            nn.Dropout(dropout),
        )
        # Adaptive scale+shift from timestep
        self.t_proj = nn.Linear(d_model, d_model * 2)

    def forward(
        self, x: Tensor, memory: Tensor, t_emb: Tensor,
        key_padding_mask: Tensor | None = None,
    ) -> Tensor:
        """
        Args:
            x: (B, S, H) noisy token embeddings.
            memory: (B, R, H) z_content role memory.
            t_emb: (B, H) or (B, S, H) timestep embedding.
            key_padding_mask: (B, S) True = ignore (for padded positions).
        """
        # Timestep-adaptive scale and shift
        if t_emb.dim() == 2:
            t_emb = t_emb.unsqueeze(1)
        scale, shift = self.t_proj(t_emb).chunk(2, dim=-1)

        # Self-attention
        h = self.norm1(x) * (1 + scale) + shift
        x = x + self.self_attn(h, h, h, key_padding_mask=key_padding_mask, need_weights=False)[0]

        # Cross-attention to z memory
        h = self.norm2(x)
        x = x + self.cross_attn(h, memory, memory, need_weights=False)[0]

        # FFN
        x = x + self.ffn(self.norm3(x))
        return x


class TreeDiffusionDecoder(nn.Module):
    """Non-autoregressive decoder that generates content tokens via diffusion.

    Architecture:
      1. Each token position gets: noisy embedding + role embedding + depth
         embedding + position embedding.
      2. Stack of TreeDenoiserBlocks: bidirectional self-attention between
         positions + cross-attention to z_content memory.
      3. Output: predicted clean token embeddings → project to vocab logits.

    Tree-structured noise schedule:
      - Each position has a depth in the dep tree (0=root, higher=leaves).
      - The noise level per position: t_pos = t_global * (depth / max_depth) ^ depth_scale.
      - Root positions are nearly clean at all t_global > 0.
      - Leaf positions have full noise until late in denoising.
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 512,
        num_layers: int = 6,
        num_heads: int = 8,
        dropout: float = 0.1,
        max_positions: int = 256,
        max_depth: int = 10,
        max_word_position: int = 8,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.max_depth = max_depth
        self.max_word_pos = max_word_position

        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.role_embedding = nn.Embedding(NUM_DEP_RELATIONS, d_model)
        self.depth_embedding = nn.Embedding(max_depth + 1, d_model)
        self.pos_embedding = nn.Embedding(max_positions, d_model)
        self.word_pos_embedding = nn.Embedding(max_word_position + 1, d_model)
        self.timestep_embedding = TimestepEmbedding(d_model)

        self.input_proj = nn.Linear(d_model, d_model)
        self.self_cond_proj = nn.Linear(d_model, d_model)

        self.layers = nn.ModuleList([
            TreeDenoiserBlock(d_model, num_heads, dropout)
            for _ in range(num_layers)
        ])
        self.final_norm = nn.LayerNorm(d_model)
        self.output_head = nn.Linear(d_model, vocab_size)

    def tree_noise_schedule(
        self, t_global: Tensor, depths: Tensor,
        depth_scale: float = 1.0, min_noise: float = 0.3,
        invert: bool = False,
    ) -> Tensor:
        """Compute per-position noise level from global t and tree depth.

        All positions get at least ``min_noise`` fraction of the full
        schedule. When ``invert=False`` (default), root positions get
        ``min_noise`` and leaves get full noise. When ``invert=True``,
        leaves get ``min_noise`` and roots get full noise — hypothesis
        that deep positions are simpler and should resolve first.

        Args:
            t_global: (B,) global diffusion time in [0, 1].
            depths: (B, S) integer depth per position.
            depth_scale: exponent controlling depth→noise mapping.
            min_noise: minimum noise fraction for quietest positions.
            invert: if True, deep positions get less noise.

        Returns:
            t_per_pos: (B, S) noise level per position in [0, 1].
        """
        normalized_depth = depths.float() / max(self.max_depth, 1)
        depth_factor = normalized_depth.pow(depth_scale)
        if invert:
            depth_factor = 1.0 - depth_factor
        noise_factor = min_noise + (1.0 - min_noise) * depth_factor
        return t_global.unsqueeze(1) * noise_factor

    def add_noise(
        self, x0: Tensor, t_per_pos: Tensor, noise: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        """Flow-matching forward process: x_t = (1-t)*x0 + t*noise.

        Args:
            x0: (B, S, H) clean token embeddings.
            t_per_pos: (B, S) noise level per position.
            noise: optional pre-sampled noise.

        Returns:
            x_t: (B, S, H) noisy embeddings.
            noise: (B, S, H) the noise used.
        """
        if noise is None:
            noise = torch.randn_like(x0)
        t = t_per_pos.unsqueeze(-1)  # (B, S, 1)
        x_t = (1 - t) * x0 + t * noise
        return x_t, noise

    @staticmethod
    def compute_word_positions(tokens: Tensor, role_offset: int) -> Tensor:
        """Position within word, reset at each role marker.

        Role markers (token >= role_offset) get position 0.
        Subsequent content tokens get 1, 2, 3, ...
        The model learns that most words end by position 2-3.
        """
        b, s = tokens.shape
        device = tokens.device
        is_boundary = tokens >= role_offset
        positions = torch.arange(s, device=device).unsqueeze(0).expand(b, -1)
        boundary_pos = torch.where(is_boundary, positions, torch.zeros_like(positions))
        boundary_pos, _ = boundary_pos.cummax(dim=1)
        return positions - boundary_pos

    @staticmethod
    def word_positions_from_roles(role_ids: Tensor) -> Tensor:
        """Approximate word positions from forward-filled role IDs.

        Used at inference step 0 when tokens aren't available yet.
        Detects span boundaries where role_id changes.
        """
        b, s = role_ids.shape
        device = role_ids.device
        positions = torch.arange(s, device=device).unsqueeze(0).expand(b, -1)
        changes = torch.cat([
            torch.ones(b, 1, dtype=torch.bool, device=device),
            role_ids[:, 1:] != role_ids[:, :-1],
        ], dim=1)
        change_pos = torch.where(changes, positions, torch.zeros_like(positions))
        change_pos, _ = change_pos.cummax(dim=1)
        return positions - change_pos

    def forward(
        self,
        x_t: Tensor,
        t_per_pos: Tensor,
        role_ids: Tensor,
        depths: Tensor,
        memory: Tensor,
        padding_mask: Tensor | None = None,
        word_positions: Tensor | None = None,
        self_cond: Tensor | None = None,
    ) -> Tensor:
        """Predict clean token embeddings from noisy input.

        Args:
            x_t: (B, S, H) noisy token embeddings.
            t_per_pos: (B, S) per-position noise level.
            role_ids: (B, S) dependency role per position.
            depths: (B, S) tree depth per position.
            memory: (B, R, H) z_content role memory.
            padding_mask: (B, S) True = padded position.
            word_positions: (B, S) position within word (0=boundary).
            self_cond: (B, S, H) previous step's x0 prediction (detached).

        Returns:
            x0_pred: (B, S, H) predicted clean embeddings.
        """
        b, s, _ = x_t.shape
        device = x_t.device

        # Build input: noisy embedding + role + depth + position + word position
        pos_ids = torch.arange(s, device=device).unsqueeze(0).expand(b, -1)
        h = self.input_proj(x_t)
        h = h + self.role_embedding(role_ids.clamp(max=NUM_DEP_RELATIONS - 1))
        h = h + self.depth_embedding(depths.clamp(max=self.max_depth))
        h = h + self.pos_embedding(pos_ids.clamp(max=self.pos_embedding.num_embeddings - 1))
        if word_positions is not None:
            h = h + self.word_pos_embedding(word_positions.clamp(max=self.max_word_pos))
        if self_cond is not None:
            h = h + self.self_cond_proj(self_cond)

        # Per-position timestep embedding
        t_emb = self.timestep_embedding(t_per_pos)  # (B, S, H)

        # Denoiser stack
        for layer in self.layers:
            h = layer(h, memory, t_emb, key_padding_mask=padding_mask)

        return self.final_norm(h)

    @torch.no_grad()
    def sample(
        self,
        seq_len: int,
        role_ids: Tensor,
        depths: Tensor,
        memory: Tensor,
        num_steps: int = 8,
        depth_scale: float = 1.0,
        min_noise: float = 0.3,
        padding_mask: Tensor | None = None,
        role_offset: int | None = None,
        ref_tokens: Tensor | None = None,
        invert_depth_noise: bool = False,
    ) -> Tensor:
        """Generate tokens via iterative denoising from pure noise.

        Word positions are refined iteratively: step 0 approximates from
        role_ids, subsequent steps derive from predicted tokens.

        Returns:
            token_ids: (B, S) predicted token IDs.
        """
        b = memory.size(0)
        device = memory.device

        # Start from noise
        x_t = torch.randn(b, seq_len, self.d_model, device=device)

        # Clamp role markers to clean embeddings if ref_tokens provided
        role_mask = None
        if ref_tokens is not None and role_offset is not None:
            role_mask = ref_tokens >= role_offset
            role_emb_clean = self.token_embedding(
                ref_tokens.clamp(max=self.token_embedding.num_embeddings - 1)
            )
            x_t = torch.where(role_mask.unsqueeze(-1), role_emb_clean, x_t)

        # Initial word positions from role structure
        word_positions = self.word_positions_from_roles(role_ids)
        word_positions = word_positions.clamp(max=self.max_word_pos)

        # Self-conditioning: chain x0 predictions across steps
        x0_prev = None

        # Reverse diffusion: t goes from 1 → 0
        for step in range(num_steps):
            t_global = torch.full((b,), 1.0 - step / num_steps, device=device)
            t_per_pos = self.tree_noise_schedule(t_global, depths, depth_scale, min_noise, invert=invert_depth_noise)

            x0_pred = self(x_t, t_per_pos, role_ids, depths, memory, padding_mask,
                           word_positions=word_positions, self_cond=x0_prev)

            x0_prev = x0_pred

            # Update word positions from predicted tokens
            if role_offset is not None and step < num_steps - 1:
                pred_tokens = self.output_head(x0_pred).argmax(dim=-1)
                word_positions = self.compute_word_positions(pred_tokens, role_offset)
                word_positions = word_positions.clamp(max=self.max_word_pos)

            if step < num_steps - 1:
                # Re-noise to next step level
                t_next = torch.full((b,), 1.0 - (step + 1) / num_steps, device=device)
                t_per_pos_next = self.tree_noise_schedule(t_next, depths, depth_scale, min_noise, invert=invert_depth_noise)
                noise = torch.randn_like(x0_pred)
                t_n = t_per_pos_next.unsqueeze(-1)
                x_t = (1 - t_n) * x0_pred + t_n * noise
                # Clamp role markers back to clean embeddings
                if role_mask is not None:
                    x_t = torch.where(role_mask.unsqueeze(-1), role_emb_clean, x_t)
            else:
                x_t = x0_pred

        # Project to token IDs
        logits = self.output_head(x_t)
        return logits.argmax(dim=-1)

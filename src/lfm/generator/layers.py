"""Linguistically-motivated transformer decoder layers.

Provides custom decoder components that encode linguistic universals as
architectural inductive biases:

- **Multi-scale hierarchical attention mask**: per-head window sizes
  creating a multi-resolution filter bank (phonotactic → morpheme →
  word → clause).
- **Rotary Positional Embeddings (RoPE)**: relative position encoding
  for translation-invariant pattern learning (morphemes work the same
  way regardless of absolute position).
- **Weight-shared recursive decoder**: the same layers applied multiple
  times, implementing literal recursion.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import torch
import torch.nn.functional as F  # noqa: N812
from torch import Tensor, nn

# ---------------------------------------------------------------------------
# Rotary Positional Embeddings (RoPE)
# ---------------------------------------------------------------------------


def precompute_rope_freqs(
    dim: int,
    max_len: int,
    theta: float = 10000.0,
    device: torch.device | str = "cpu",
) -> Tensor:
    """Precompute RoPE rotation frequency matrix.

    Args:
        dim: Head dimension (must be even).
        max_len: Maximum sequence length.
        theta: Base frequency scaling parameter.
        device: Target device.

    Returns:
        Complex tensor of shape ``(max_len, dim // 2)`` containing
        rotation frequencies for each position.
    """
    freqs = 1.0 / (
        theta ** (torch.arange(0, dim, 2, device=device).float() / dim)
    )
    positions = torch.arange(max_len, device=device).float()
    angles = torch.outer(positions, freqs)  # (max_len, dim//2)
    return torch.polar(torch.ones_like(angles), angles)  # complex64


def apply_rope(x: Tensor, freqs: Tensor) -> Tensor:
    """Apply rotary positional embeddings to query or key tensor.

    Args:
        x: Input tensor of shape ``(batch, num_heads, seq_len, head_dim)``.
        freqs: Precomputed frequency tensor from ``precompute_rope_freqs``,
            sliced to ``(seq_len, head_dim // 2)``.

    Returns:
        Rotated tensor of same shape as ``x``.
    """
    # Reshape x to pairs: (batch, heads, seq, dim//2, 2) -> complex
    x_reshape = x.float().reshape(*x.shape[:-1], -1, 2)
    x_complex = torch.view_as_complex(x_reshape)
    # Apply rotation (broadcast over batch and heads)
    freqs = freqs[: x_complex.size(-2)]  # slice to seq_len
    rotated = x_complex * freqs.unsqueeze(0).unsqueeze(0)
    # Back to real
    return torch.view_as_real(rotated).flatten(-2).type_as(x)


# ---------------------------------------------------------------------------
# KV Cache for autoregressive decoding
# ---------------------------------------------------------------------------


@dataclass
class KVCache:
    """Key-value cache for incremental autoregressive decoding.

    Stores K and V tensors for each decoder layer application (not per
    unique layer — weight-shared layers get separate cache slots).

    Pre-allocates tensors at ``init()`` to avoid repeated concatenation.
    """

    batch_size: int = 0
    max_len: int = 0
    num_heads: int = 0
    head_dim: int = 0
    num_applications: int = 0  # len(layer_order), not len(unique_layers)
    device: torch.device | str = "cpu"
    dtype: torch.dtype = torch.float16

    # Pre-allocated: (num_applications, B, H, max_len, D)
    k_cache: Tensor = field(default_factory=lambda: torch.empty(0))
    v_cache: Tensor = field(default_factory=lambda: torch.empty(0))
    # Cross-attention cache (computed once from memory)
    cross_cache: Tensor = field(default_factory=lambda: torch.empty(0))
    seq_len: int = 0  # Current position in the cache

    def init(
        self,
        batch_size: int,
        max_len: int,
        num_heads: int,
        head_dim: int,
        num_applications: int,
        device: torch.device | str,
        dtype: torch.dtype = torch.float16,
    ) -> None:
        """Pre-allocate cache tensors."""
        self.batch_size = batch_size
        self.max_len = max_len
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.num_applications = num_applications
        self.device = device
        self.dtype = dtype
        self.seq_len = 0

        self.k_cache = torch.zeros(
            num_applications, batch_size, num_heads, max_len, head_dim,
            device=device, dtype=dtype,
        )
        self.v_cache = torch.zeros(
            num_applications, batch_size, num_heads, max_len, head_dim,
            device=device, dtype=dtype,
        )
        self.cross_cache = torch.empty(0)

    def update(self, app_idx: int, k: Tensor, v: Tensor) -> tuple[Tensor, Tensor]:
        """Write new K, V at current position and return full cached K, V.

        Args:
            app_idx: Layer application index (0..num_applications-1).
            k: New key ``(B, H, 1, D)``.
            v: New value ``(B, H, 1, D)``.

        Returns:
            Cached (k, v) sliced to ``[:seq_len+1]``.
        """
        pos = self.seq_len
        self.k_cache[app_idx, :, :, pos : pos + 1, :] = k.to(self.dtype)
        self.v_cache[app_idx, :, :, pos : pos + 1, :] = v.to(self.dtype)
        # Return cached up to current position (inclusive)
        return (
            self.k_cache[app_idx, :, :, : pos + 1, :],
            self.v_cache[app_idx, :, :, : pos + 1, :],
        )

    def advance(self) -> None:
        """Advance the sequence position after processing one token."""
        self.seq_len += 1


# ---------------------------------------------------------------------------
# Multi-scale hierarchical causal mask
# ---------------------------------------------------------------------------


def multiscale_causal_mask(
    seq_len: int,
    num_heads: int,
    head_windows: tuple[int, ...],
    global_every: int = 7,
    device: torch.device | str = "cpu",
) -> Tensor:
    """Per-head hierarchical causal attention mask.

    Each head gets a different sliding window size, creating a
    multi-resolution filter bank for linguistic processing:

    - Small windows (3): phonotactic / sub-morpheme scale
    - Medium windows (7): morpheme scale
    - Large windows (15): word / short phrase scale
    - Window 0: full causal (clause / sentence level)

    Global positions (every ``global_every`` tokens) get full attention
    in all heads — these serve as natural composition/boundary points.

    Args:
        seq_len: Sequence length.
        num_heads: Number of attention heads.
        head_windows: Tuple of window sizes, one per head.  ``0`` means
            full causal attention.
        global_every: Spacing of global attention positions.
        device: Target device.

    Returns:
        Float mask of shape ``(num_heads, seq_len, seq_len)`` where
        ``-inf`` = masked (no attention) and ``0`` = attend.
    """
    assert len(head_windows) == num_heads, (
        f"head_windows length {len(head_windows)} != num_heads {num_heads}"
    )

    # Vectorized construction — no Python loops over seq positions
    # Row indices (i) and column indices (j)
    rows = torch.arange(seq_len, device=device).unsqueeze(1)  # (S, 1)
    cols = torch.arange(seq_len, device=device).unsqueeze(0)  # (1, S)
    # Causal: future positions always masked
    causal = cols > rows  # (S, S) True = future

    # Global positions mask: cols or rows that are multiples of global_every
    is_global_col = (cols % global_every) == 0  # (1, S)
    is_global_row = (rows % global_every) == 0  # (S, 1)

    mask = torch.zeros(num_heads, seq_len, seq_len, device=device)

    neg_inf = torch.tensor(float("-inf"), device=device)
    zero = torch.tensor(0.0, device=device)

    for h, window in enumerate(head_windows):
        if window == 0:
            # Full causal
            mask[h] = torch.where(causal, neg_inf, zero)
        else:
            # Outside window OR future → masked
            outside_window = (rows - cols) >= window  # (S, S)
            blocked = causal | outside_window
            # Unblock: global columns (attend TO globals) and
            # global rows (attend FROM globals to everything causal)
            unblocked_by_global_col = is_global_col & ~causal
            unblocked_by_global_row = is_global_row & ~causal
            blocked = blocked & ~unblocked_by_global_col & ~unblocked_by_global_row
            mask[h] = torch.where(blocked, neg_inf, zero)

    return mask


# ---------------------------------------------------------------------------
# Custom Decoder Layer with RoPE + Multi-Scale Attention
# ---------------------------------------------------------------------------


class PhraseDecoderLayer(nn.Module):
    """Transformer decoder layer with RoPE and multi-scale attention.

    Replaces ``nn.TransformerDecoderLayer`` with a custom implementation
    that applies Rotary Positional Embeddings to self-attention queries
    and keys, enabling translation-invariant pattern learning.

    The standard cross-attention (to the VAE latent code) uses no
    positional encoding — the latent is position-free.

    Args:
        d_model: Model dimensionality.
        nhead: Number of attention heads.
        dim_feedforward: FFN inner dimensionality.
        dropout: Dropout rate.
    """

    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.nhead = nhead
        self.head_dim = d_model // nhead

        # Self-attention projections (manual for RoPE access to Q, K)
        self.self_attn_qkv = nn.Linear(d_model, 3 * d_model)
        self.self_attn_out = nn.Linear(d_model, d_model)

        # Cross-attention (standard — no RoPE needed for latent code)
        self.cross_attn = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=True
        )

        # FFN
        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
            nn.Dropout(dropout),
        )

        # Layer norms (pre-norm architecture)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        tgt: Tensor,
        memory: Tensor,
        tgt_mask: Tensor | None = None,
        rope_freqs: Tensor | None = None,
        capture_attention: bool = False,
        xattn_mask: Tensor | None = None,
    ) -> Tensor:
        """Forward pass with RoPE self-attention.

        Args:
            tgt: Target sequence ``(batch, seq_len, d_model)``.
            memory: Encoder/latent memory ``(batch, mem_len, d_model)``.
            tgt_mask: Self-attention mask ``(num_heads, seq_len, seq_len)``
                or ``(seq_len, seq_len)``.
            rope_freqs: Precomputed RoPE frequencies for self-attention.
            capture_attention: If ``True``, store the post-softmax self-attention
                weights on ``self._last_self_attn`` for visualization.
            xattn_mask: Optional cross-attention mask for position-dependent
                memory access ``(batch * num_heads, seq_len, mem_len)``.

        Returns:
            Output tensor ``(batch, seq_len, d_model)``.
        """
        # --- Self-attention with RoPE ---
        x = self.norm1(tgt)
        b, s, d = x.shape
        qkv = self.self_attn_qkv(x).reshape(b, s, 3, self.nhead, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, B, H, S, D)
        q, k, v = qkv[0], qkv[1], qkv[2]

        # Apply RoPE to Q and K
        if rope_freqs is not None:
            q = apply_rope(q, rope_freqs[:s])
            k = apply_rope(k, rope_freqs[:s])

        # Scaled dot-product attention with mask
        scale = math.sqrt(self.head_dim)
        attn = torch.matmul(q, k.transpose(-2, -1)) / scale  # (B, H, S, S)

        if tgt_mask is not None:
            if tgt_mask.dim() == 2:
                # (S, S) -> broadcast over heads
                attn = attn + tgt_mask.unsqueeze(0).unsqueeze(0)
            elif tgt_mask.dim() == 3:
                # (H, S, S) -> broadcast over batch
                attn = attn + tgt_mask.unsqueeze(0)

        attn = F.softmax(attn, dim=-1)

        if capture_attention:
            self._last_self_attn = attn.detach()

        attn = self.dropout(attn)

        self_attn_out = torch.matmul(attn, v)  # (B, H, S, D)
        self_attn_out = self_attn_out.transpose(1, 2).reshape(b, s, d)
        self_attn_out = self.self_attn_out(self_attn_out)

        tgt = tgt + self.dropout(self_attn_out)

        # --- Cross-attention (standard, no RoPE) ---
        x = self.norm2(tgt)
        cross_out, _ = self.cross_attn(
            x, memory, memory, attn_mask=xattn_mask,
        )
        tgt = tgt + self.dropout(cross_out)

        # --- FFN ---
        x = self.norm3(tgt)
        tgt = tgt + self.ffn(x)

        return tgt


    def forward_cached(
        self,
        tgt: Tensor,
        memory: Tensor,
        kv_cache: KVCache,
        app_idx: int,
        rope_freqs: Tensor | None = None,
        tgt_mask_row: Tensor | None = None,
        xattn_mask: Tensor | None = None,
    ) -> Tensor:
        """Cached forward pass: process only the newest token.

        Uses cached K/V from previous steps for self-attention,
        computing Q/K/V only for the new position.

        Args:
            tgt: New token embedding ``(batch, 1, d_model)``.
            memory: Latent memory ``(batch, 1, d_model)``.
            kv_cache: Shared KV cache.
            app_idx: This layer's application index in the cache.
            rope_freqs: Full RoPE frequencies (indexed at current pos).
            tgt_mask_row: Attention mask row for current position
                ``(num_heads, 1, seq_len)`` — which past positions
                this token can attend to.
            xattn_mask: Optional cross-attention mask
                ``(batch * num_heads, 1, mem_len)``.

        Returns:
            Output for the new position ``(batch, 1, d_model)``.
        """
        pos = kv_cache.seq_len

        # --- Self-attention (cached) ---
        x = self.norm1(tgt)
        b, _, d = x.shape
        qkv = self.self_attn_qkv(x).reshape(b, 1, 3, self.nhead, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, B, H, 1, D)
        q, k_new, v_new = qkv[0], qkv[1], qkv[2]

        # Apply RoPE only to current position
        if rope_freqs is not None:
            q = apply_rope(q, rope_freqs[pos : pos + 1])
            k_new = apply_rope(k_new, rope_freqs[pos : pos + 1])

        # Update cache and get full K, V
        k_full, v_full = kv_cache.update(app_idx, k_new, v_new)

        # Attention: Q(1) @ K(seq_len)^T -> (B, H, 1, seq_len)
        scale = math.sqrt(self.head_dim)
        attn = torch.matmul(q, k_full.transpose(-2, -1).to(q.dtype)) / scale

        if tgt_mask_row is not None:
            attn = attn + tgt_mask_row.unsqueeze(0)  # (1, H, 1, S) broadcast

        attn = F.softmax(attn, dim=-1)
        # No dropout during inference (eval mode)

        self_attn_out = torch.matmul(attn, v_full.to(q.dtype))  # (B, H, 1, D)
        self_attn_out = self_attn_out.transpose(1, 2).reshape(b, 1, d)
        self_attn_out = self.self_attn_out(self_attn_out)

        tgt = tgt + self_attn_out

        # --- Cross-attention (to latent — same every step) ---
        x = self.norm2(tgt)
        cross_out, _ = self.cross_attn(x, memory, memory, attn_mask=xattn_mask)
        tgt = tgt + cross_out

        # --- FFN ---
        x = self.norm3(tgt)
        tgt = tgt + self.ffn(x)

        return tgt


class PhraseDecoder(nn.Module):
    """Decoder stack with optional weight sharing for recursive application.

    When ``share_layers=True``, creates ``num_layers // 2`` unique layers
    and applies each twice, implementing literal recursion. The same
    transformation at each depth mirrors how syntactic Merge operates at
    every level of recursive embedding.

    Args:
        layer: A ``PhraseDecoderLayer`` instance (or compatible).
        num_layers: Total number of layer applications.
        share_layers: If ``True``, use ``num_layers // 2`` unique layers
            each applied twice.
    """

    def __init__(
        self,
        d_model: int,
        nhead: int,
        num_layers: int,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        share_layers: bool = True,
    ) -> None:
        super().__init__()
        self.num_layers = num_layers
        self.share_layers = share_layers

        if share_layers:
            n_unique = max(1, num_layers // 2)
            unique_layers = nn.ModuleList([
                PhraseDecoderLayer(d_model, nhead, dim_feedforward, dropout)
                for _ in range(n_unique)
            ])
            # Build application order: [0, 1, 0, 1, ...] for n_unique=2
            self.layers = unique_layers
            self.layer_order = []
            for i in range(num_layers):
                self.layer_order.append(i % n_unique)
        else:
            self.layers = nn.ModuleList([
                PhraseDecoderLayer(d_model, nhead, dim_feedforward, dropout)
                for _ in range(num_layers)
            ])
            self.layer_order = list(range(num_layers))

        self.final_norm = nn.LayerNorm(d_model)

    def forward(
        self,
        tgt: Tensor,
        memory: Tensor,
        tgt_mask: Tensor | None = None,
        rope_freqs: Tensor | None = None,
        capture_attention: bool = False,
        xattn_mask: Tensor | None = None,
    ) -> Tensor:
        """Forward pass through all decoder layers.

        Args:
            tgt: Target ``(batch, seq_len, d_model)``.
            memory: Latent code ``(batch, 1, d_model)``.
            tgt_mask: Multi-scale causal mask.
            rope_freqs: Precomputed RoPE frequencies.
            capture_attention: If ``True``, each layer stores its
                post-softmax attention on ``layer._last_self_attn``.
            xattn_mask: Optional cross-attention mask for position-dependent
                memory access ``(batch * num_heads, seq_len, mem_len)``.

        Returns:
            Decoded output ``(batch, seq_len, d_model)``.
        """
        x = tgt
        for idx in self.layer_order:
            x = self.layers[idx](
                x, memory,
                tgt_mask=tgt_mask,
                rope_freqs=rope_freqs,
                capture_attention=capture_attention,
                xattn_mask=xattn_mask,
            )
        return self.final_norm(x)

    def make_kv_cache(
        self,
        batch_size: int,
        max_len: int,
        device: torch.device | str,
        dtype: torch.dtype = torch.float16,
    ) -> KVCache:
        """Create a pre-allocated KV cache for autoregressive decoding."""
        layer0 = self.layers[0]
        cache = KVCache()
        cache.init(
            batch_size=batch_size,
            max_len=max_len,
            num_heads=layer0.nhead,
            head_dim=layer0.head_dim,
            num_applications=len(self.layer_order),
            device=device,
            dtype=dtype,
        )
        return cache

    def forward_cached(
        self,
        tgt: Tensor,
        memory: Tensor,
        kv_cache: KVCache,
        rope_freqs: Tensor | None = None,
        tgt_mask_row: Tensor | None = None,
        xattn_mask: Tensor | None = None,
    ) -> Tensor:
        """Cached forward: process one new token through all layers.

        Args:
            tgt: New token embedding ``(batch, 1, d_model)``.
            memory: Latent memory ``(batch, 1, d_model)``.
            kv_cache: Pre-allocated KV cache.
            rope_freqs: Full precomputed RoPE frequencies.
            tgt_mask_row: Mask row for current position
                ``(num_heads, 1, past_len+1)``.
            xattn_mask: Optional cross-attention mask
                ``(batch * num_heads, 1, mem_len)``.

        Returns:
            Output for the new token ``(batch, 1, d_model)``.
        """
        x = tgt
        for app_idx, layer_idx in enumerate(self.layer_order):
            x = self.layers[layer_idx].forward_cached(
                x, memory,
                kv_cache=kv_cache,
                app_idx=app_idx,
                rope_freqs=rope_freqs,
                tgt_mask_row=tgt_mask_row,
                xattn_mask=xattn_mask,
            )
        return self.final_norm(x)

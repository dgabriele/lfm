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


class LinguisticDecoderLayer(nn.Module):
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
        cross_out, _ = self.cross_attn(x, memory, memory)
        tgt = tgt + self.dropout(cross_out)

        # --- FFN ---
        x = self.norm3(tgt)
        tgt = tgt + self.ffn(x)

        return tgt


class LinguisticDecoder(nn.Module):
    """Decoder stack with optional weight sharing for recursive application.

    When ``share_layers=True``, creates ``num_layers // 2`` unique layers
    and applies each twice, implementing literal recursion. The same
    transformation at each depth mirrors how syntactic Merge operates at
    every level of recursive embedding.

    Args:
        layer: A ``LinguisticDecoderLayer`` instance (or compatible).
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
                LinguisticDecoderLayer(d_model, nhead, dim_feedforward, dropout)
                for _ in range(n_unique)
            ])
            # Build application order: [0, 1, 0, 1, ...] for n_unique=2
            self.layers = unique_layers
            self.layer_order = []
            for i in range(num_layers):
                self.layer_order.append(i % n_unique)
        else:
            self.layers = nn.ModuleList([
                LinguisticDecoderLayer(d_model, nhead, dim_feedforward, dropout)
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
    ) -> Tensor:
        """Forward pass through all decoder layers.

        Args:
            tgt: Target ``(batch, seq_len, d_model)``.
            memory: Latent code ``(batch, 1, d_model)``.
            tgt_mask: Multi-scale causal mask.
            rope_freqs: Precomputed RoPE frequencies.
            capture_attention: If ``True``, each layer stores its
                post-softmax attention on ``layer._last_self_attn``.

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
            )
        return self.final_norm(x)

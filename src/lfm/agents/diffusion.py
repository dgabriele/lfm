"""Diffusion-based z-sequence generator for expression games.

Replaces the GRU autoregressive approach with a small transformer
denoiser that refines all K z positions simultaneously, conditioned
on the input embedding via cross-attention.  Uses flow matching
(linear interpolation between data and noise) for T=4 steps.

All segments see each other during refinement, enabling co-adaptation
that the sequential GRU cannot achieve.  Variable length emerges from
learned per-position activity scores rather than a halt gate.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F  # noqa: N812
from torch import Tensor, nn


def sinusoidal_embedding(t: Tensor, dim: int) -> Tensor:
    """Sinusoidal timestep embedding.

    Args:
        t: ``(batch,)`` integer or float timesteps.
        dim: Embedding dimensionality.

    Returns:
        ``(batch, dim)`` sinusoidal embedding.
    """
    half = dim // 2
    freqs = torch.exp(
        -math.log(10000.0) * torch.arange(half, device=t.device).float() / half
    )
    args = t.float().unsqueeze(-1) * freqs.unsqueeze(0)
    return torch.cat([args.sin(), args.cos()], dim=-1)


class DenoiserBlock(nn.Module):
    """Pre-norm transformer block with self-attention and cross-attention.

    Self-attention lets z positions see each other for co-adaptation.
    Cross-attention conditions on the input embedding.

    Args:
        d_model: Hidden dimensionality.
        num_heads: Number of attention heads.
        dim_feedforward: FFN inner dimension.
        dropout: Dropout rate.
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int = 8,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
    ) -> None:
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
            nn.Linear(d_model, dim_feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: Tensor, cond: Tensor) -> Tensor:
        """Forward pass.

        Args:
            x: ``(B, K, d_model)`` z position hidden states.
            cond: ``(B, 1, d_model)`` conditioning token.

        Returns:
            ``(B, K, d_model)`` refined hidden states.
        """
        # Self-attention across K positions
        h = self.norm1(x)
        x = x + self.self_attn(h, h, h, need_weights=False)[0]

        # Cross-attention to conditioning
        h = self.norm2(x)
        x = x + self.cross_attn(h, cond, cond, need_weights=False)[0]

        # FFN
        x = x + self.ffn(self.norm3(x))

        return x


class DiffusionZGenerator(nn.Module):
    """Generate z-sequences via iterative denoising in latent space.

    Starts from noise drawn from the pretrained z distribution, then
    refines all K positions simultaneously over T steps.  A per-position
    activity head determines which segments contribute to the expression.

    Uses flow matching: forward ``z_t = (1-t)*z_0 + t*ε``, model predicts
    the clean ``z_0`` directly.  At T=4 steps the full reverse process is
    differentiable and trains end-to-end through the game loss.

    Args:
        input_dim: Conditioning embedding dimension (384).
        latent_dim: VAE latent dimensionality (256).
        d_model: Denoiser hidden dimension.
        max_segments: Number of z positions (K).
        num_steps: Reverse diffusion steps (T).
        num_layers: Denoiser transformer blocks.
        num_heads: Attention heads per block.
        z_mean: Per-dim latent mean from pretraining.
        z_std: Per-dim latent std from pretraining.
        target_segments: Target E[K] for length regularization.
    """

    def __init__(
        self,
        input_dim: int,
        latent_dim: int,
        d_model: int = 512,
        max_segments: int = 8,
        num_steps: int = 4,
        num_layers: int = 4,
        num_heads: int = 8,
        z_mean: Tensor | None = None,
        z_std: Tensor | None = None,
        target_segments: float = 2.5,
    ) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        self.d_model = d_model
        self.max_segments = max_segments
        self.num_steps = num_steps
        self.target_segments = target_segments

        # Project noisy z into denoiser space
        self.z_in = nn.Linear(latent_dim, d_model)

        # Conditioning projection
        self.cond_proj = nn.Linear(input_dim, d_model)

        # Timestep embedding
        self.time_embed = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )

        # Learned per-slot position embeddings
        self.pos_embed = nn.Parameter(torch.randn(max_segments, d_model) * 0.02)

        # Denoiser stack
        self.denoiser_layers = nn.ModuleList([
            DenoiserBlock(d_model, num_heads, d_model * 4)
            for _ in range(num_layers)
        ])

        # Output heads
        self.z_out = nn.Linear(d_model, latent_dim)
        self.activity_head = nn.Linear(d_model, 1)

        # Initialize activity biases for Zipfian length distribution:
        # pos 0 = always 1.0 (hardcoded), pos 1 = ~0.88, pos 2 = ~0.62,
        # pos 3 = ~0.27, pos 4+ = ~0.05
        with torch.no_grad():
            biases = torch.full((max_segments,), -3.0)
            if max_segments > 1:
                biases[1] = 2.0
            if max_segments > 2:
                biases[2] = 0.5
            if max_segments > 3:
                biases[3] = -1.0
            # Store for reference; actual bias is scalar per the linear head,
            # so we initialize the bias to the mean and let pos_embed handle
            # per-position differentiation.
            self.activity_head.bias.fill_(-1.0)
            # Encode the Zipfian prior into pos_embed's contribution to
            # the activity head by adjusting the last dim of pos_embed
            # proportionally.  Simpler: just add a learned buffer.
            self.register_buffer("_activity_bias", biases)

        # z distribution stats from pretraining
        if z_mean is not None:
            self.register_buffer("z_mean", z_mean.detach().clone())
            self.register_buffer("z_std", z_std.detach().clone())
        else:
            self.register_buffer("z_mean", torch.zeros(latent_dim))
            self.register_buffer("z_std", torch.ones(latent_dim))

        # Scale z_out to match pretrained distribution
        with torch.no_grad():
            target_std = self.z_std.mean().item()
            fan_in = self.z_out.weight.size(1)
            current_std = self.z_out.weight.data.std().item() * (fan_in ** 0.5)
            scale = target_std / max(current_std, 1e-6)
            self.z_out.weight.data.mul_(scale)
            self.z_out.bias.data.copy_(self.z_mean)

    def _denoise_step(
        self, z_t: Tensor, cond: Tensor, t_embed: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """Single denoising pass through the transformer.

        Args:
            z_t: ``(B, K, latent_dim)`` current noisy z.
            cond: ``(B, d_model)`` projected conditioning.
            t_embed: ``(B, d_model)`` timestep embedding.

        Returns:
            z_pred: ``(B, K, latent_dim)`` predicted clean z.
            activity_logits: ``(B, K)`` per-position activity.
        """
        # Project noisy z to denoiser space + add position + time
        h = self.z_in(z_t) + self.pos_embed.unsqueeze(0) + t_embed.unsqueeze(1)

        # Condition token for cross-attention
        cond_tokens = cond.unsqueeze(1)  # (B, 1, d_model)

        for block in self.denoiser_layers:
            h = block(h, cond_tokens)

        z_pred = self.z_out(h)
        activity_logits = self.activity_head(h).squeeze(-1) + self._activity_bias

        return z_pred, activity_logits

    def _reverse_sample(self, embedding: Tensor) -> tuple[Tensor, Tensor]:
        """Full T-step reverse process: noise → clean z_seq.

        Args:
            embedding: ``(B, input_dim)`` conditioning embedding.

        Returns:
            z_seq: ``(B, K, latent_dim)`` denoised z vectors.
            activity_logits: ``(B, K)`` per-position activity logits.
        """
        B = embedding.size(0)
        device = embedding.device
        T = self.num_steps

        # Start from noise scaled to pretrained z distribution
        z_t = (
            torch.randn(B, self.max_segments, self.latent_dim, device=device)
            * self.z_std
            + self.z_mean
        )

        # Project conditioning once
        cond = self.cond_proj(embedding)

        # Flow matching reverse: step from t=1 → t=0
        # At each step, predict z_0 and interpolate toward it
        activity_logits = None
        for step in range(T):
            t_val = 1.0 - step / T  # 1.0, 0.75, 0.5, 0.25
            t_next = 1.0 - (step + 1) / T  # 0.75, 0.5, 0.25, 0.0

            t_tensor = torch.full((B,), t_val, device=device)
            t_embed = self.time_embed(sinusoidal_embedding(t_tensor, self.d_model))

            z_pred, activity_logits = self._denoise_step(z_t, cond, t_embed)

            if t_next > 0:
                # Interpolate toward predicted z_0
                z_t = t_next * z_t + (1.0 - t_next / t_val) * (z_pred - z_t)
                # Add small noise for exploration (not at final step)
                noise_scale = 0.1 * t_next * self.z_std
                z_t = z_t + noise_scale * torch.randn_like(z_t)
            else:
                z_t = z_pred

        return z_t, activity_logits

    def forward(
        self, embedding: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Generate z sequence via reverse diffusion.

        Args:
            embedding: ``(B, input_dim)`` anchor embeddings.

        Returns:
            z_seq: ``(B, K, latent_dim)`` denoised z vectors.
            z_weights: ``(B, K)`` per-position activity weights.
            num_segments: ``(B,)`` soft segment count.
        """
        z_seq, activity_logits = self._reverse_sample(embedding)

        z_weights = torch.sigmoid(activity_logits)
        z_weights[:, 0] = 1.0  # First segment always active

        num_segments = z_weights.sum(dim=-1)

        return z_seq, z_weights, num_segments


def length_distribution_loss(
    z_weights: Tensor, target_mean: float = 2.5,
) -> Tensor:
    """Regularize segment count toward a target mean.

    Soft penalty on mean segment count plus entropy bonus to encourage
    variable lengths across the batch (Zipfian distribution).

    Args:
        z_weights: ``(B, K)`` per-position activity weights.
        target_mean: Target expected segment count.

    Returns:
        Scalar loss.
    """
    num_segments = z_weights.sum(dim=-1)

    # Mean penalty
    mean_loss = (num_segments.mean() - target_mean).pow(2)

    # Entropy bonus: encourage diverse lengths across the batch
    p = z_weights.mean(dim=0)  # (K,) average activity per position
    entropy = -(
        p * p.clamp(min=1e-8).log()
        + (1 - p) * (1 - p).clamp(min=1e-8).log()
    ).sum()

    return mean_loss - 0.1 * entropy

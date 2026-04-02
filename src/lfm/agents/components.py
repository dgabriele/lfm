"""Shared neural components for agent communication games."""

from __future__ import annotations

import torch
from torch import Tensor, nn


class MessageEncoder(nn.Module):
    """Encode variable-length decoder hidden states into a fixed message vector.

    Uses self-attention to process the decoder's multi-scale hidden states,
    then a learned query cross-attention readout to produce a fixed-size
    vector.  This preserves the rich per-position structure from the
    multi-head decoder rather than destroying it with mean-pooling.

    Args:
        hidden_dim: Dimensionality of decoder hidden states.
        output_dim: Dimensionality of the output message vector.
        num_heads: Number of attention heads in self-attention and readout.
        num_layers: Number of self-attention layers.
    """

    def __init__(
        self,
        hidden_dim: int,
        output_dim: int,
        num_heads: int = 8,
        num_layers: int = 2,
    ) -> None:
        super().__init__()
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=num_heads,
                dim_feedforward=hidden_dim * 4,
                batch_first=True,
                dropout=0.1,
            )
            for _ in range(num_layers)
        ])
        self.query = nn.Parameter(torch.randn(1, 1, hidden_dim) * 0.02)
        self.readout = nn.MultiheadAttention(
            embed_dim=hidden_dim, num_heads=num_heads, batch_first=True,
        )
        self.out_norm = nn.LayerNorm(hidden_dim)
        self.proj = nn.Linear(hidden_dim, output_dim)

    def forward(self, hidden_states: Tensor, mask: Tensor) -> Tensor:
        """Encode decoder hidden states into a message vector.

        Args:
            hidden_states: ``(batch, seq_len, hidden_dim)`` decoder states.
            mask: ``(batch, seq_len)`` boolean mask (``True`` = valid).

        Returns:
            ``(batch, output_dim)`` message vector.
        """
        pad_mask = ~mask
        x = hidden_states
        for layer in self.layers:
            x = layer(x, src_key_padding_mask=pad_mask)

        b = x.size(0)
        query = self.query.expand(b, -1, -1)
        readout, _ = self.readout(query, x, x, key_padding_mask=pad_mask)
        readout = self.out_norm(readout.squeeze(1))
        return self.proj(readout)


class ZDiversityLoss(nn.Module):
    """Regularize intra-expression z diversity toward a data-driven target.

    Penalizes when z vectors within an expression are more similar
    (higher cosine similarity) than typical vectors drawn from the
    pretrained latent distribution.  This prevents the GRU from
    collapsing all segments to near-identical z's while keeping them
    close enough to maintain linguistic coherence for downstream
    translation.

    The target similarity is computed empirically from the pretrained
    z distribution: sample random z vectors from N(z_mean, z_std),
    compute their pairwise cosine similarity, and use the mean as the
    baseline.  Intra-expression similarity above this target is
    penalized via a hinge loss.

    Args:
        z_mean: Per-dimension latent mean from pretraining, ``(latent_dim,)``.
        z_std: Per-dimension latent std from pretraining, ``(latent_dim,)``.
        target: Override target similarity.  ``None`` = auto-compute.
        n_calibration_samples: Samples for empirical target estimation.
    """

    def __init__(
        self,
        z_mean: Tensor,
        z_std: Tensor,
        target: float | None = None,
        n_calibration_samples: int = 1000,
    ) -> None:
        super().__init__()
        if target is not None:
            self.register_buffer("target_sim", torch.tensor(target))
        else:
            with torch.no_grad():
                zs = torch.randn(n_calibration_samples, z_mean.shape[0])
                zs = zs * z_std.cpu() + z_mean.cpu()
                normed = torch.nn.functional.normalize(zs, dim=-1)
                sim = normed @ normed.T
                mask = ~torch.eye(n_calibration_samples, dtype=torch.bool)
                self.register_buffer("target_sim", sim[mask].mean())

    def forward(
        self, z_seq: Tensor, z_weights: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """Compute diversity hinge loss and mean similarity.

        Args:
            z_seq: ``(batch, K, latent_dim)`` z vectors per segment.
            z_weights: ``(batch, K)`` segment weights from PonderNet.

        Returns:
            Tuple of ``(loss, mean_sim)`` — scalar hinge loss and
            mean intra-expression cosine similarity (detached).
        """
        K = z_seq.size(1)
        z_normed = torch.nn.functional.normalize(z_seq, dim=-1)
        sim = torch.bmm(z_normed, z_normed.transpose(1, 2))

        active = z_weights > 0.01
        active_pair = active.unsqueeze(-1) & active.unsqueeze(-2)
        diag_mask = ~torch.eye(K, dtype=torch.bool, device=z_seq.device).unsqueeze(0)
        pair_mask = active_pair & diag_mask

        n_pairs = pair_mask.float().sum(dim=(1, 2)).clamp(min=1)
        mean_sim = (sim.clamp(min=0) * pair_mask.float()).sum(dim=(1, 2)) / n_pairs

        excess = (mean_sim - self.target_sim).clamp(min=0)
        return excess.mean(), mean_sim.mean().detach()

    @staticmethod
    def compute_similarity(
        z_seq: Tensor, z_weights: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """Compute mean intra-expression cosine similarity (no loss).

        Useful as a diagnostic when diversity regularization is disabled.

        Returns:
            Tuple of ``(zero_loss, mean_sim)``.
        """
        K = z_seq.size(1)
        z_normed = torch.nn.functional.normalize(z_seq, dim=-1)
        sim = torch.bmm(z_normed, z_normed.transpose(1, 2))

        active = z_weights > 0.01
        active_pair = active.unsqueeze(-1) & active.unsqueeze(-2)
        diag_mask = ~torch.eye(K, dtype=torch.bool, device=z_seq.device).unsqueeze(0)
        pair_mask = active_pair & diag_mask

        n_pairs = pair_mask.float().sum(dim=(1, 2)).clamp(min=1)
        mean_sim = (sim.clamp(min=0) * pair_mask.float()).sum(dim=(1, 2)) / n_pairs
        return torch.tensor(0.0, device=z_seq.device), mean_sim.mean()


class ZDistributionLoss(nn.Module):
    """Match projected z statistics to the pretrained decoder distribution.

    Penalizes when batch z vectors deviate from the pretraining z
    distribution's per-dimension mean and standard deviation.  This
    prevents the projection from collapsing to a tiny subspace of the
    latent space where the decoder produces degenerate output.

    The decoder learned to produce diverse, well-formed IPA across its
    full training distribution.  Keeping the projected z vectors in
    that distribution ensures the decoder's full output diversity is
    available to the expression game.

    Args:
        z_mean: Per-dimension latent mean from pretraining, ``(latent_dim,)``.
        z_std: Per-dimension latent std from pretraining, ``(latent_dim,)``.
    """

    def __init__(self, z_mean: Tensor, z_std: Tensor) -> None:
        super().__init__()
        self.register_buffer("target_mean", z_mean.detach().clone())
        self.register_buffer("target_std", z_std.detach().clone())

    def forward(self, z: Tensor) -> tuple[Tensor, Tensor]:
        """Compute distribution matching loss.

        Args:
            z: ``(batch, latent_dim)`` projected z vectors.

        Returns:
            Tuple of ``(loss, coverage_ratio)`` — scalar MSE loss and
            mean per-dimension std ratio (detached diagnostic).
        """
        batch_mean = z.mean(dim=0)
        batch_std = z.std(dim=0)

        mean_loss = (batch_mean - self.target_mean).pow(2).mean()
        std_loss = (batch_std - self.target_std).pow(2).mean()

        ratio = (batch_std / self.target_std.clamp(min=1e-6)).mean().detach()
        return mean_loss + std_loss, ratio


class IPAEncoder(nn.Module):
    """Encode IPA token sequences into fixed-size vectors.

    Wraps a token embedding lookup + ``MessageEncoder`` (self-attention
    + learned query readout).  Both the sender's live expression and
    the cached candidate expressions go through the same encoder,
    enabling IPA-to-IPA comparison in the receiver.

    Args:
        vocab_size: Full vocabulary size (including BOS/EOS).
        hidden_dim: Token embedding and encoder hidden dimension.
        output_dim: Output message vector dimension.
        num_heads: Attention heads.
        num_layers: Self-attention layers.
    """

    def __init__(
        self,
        vocab_size: int,
        hidden_dim: int,
        output_dim: int,
        num_heads: int = 8,
        num_layers: int = 2,
    ) -> None:
        super().__init__()
        self.token_embed = nn.Embedding(vocab_size, hidden_dim)
        self.encoder = MessageEncoder(hidden_dim, output_dim, num_heads, num_layers)

    def init_from_decoder(self, decoder_token_embedding: nn.Embedding) -> None:
        """Warm-start token embeddings from the frozen decoder."""
        with torch.no_grad():
            self.token_embed.weight.copy_(decoder_token_embedding.weight)

    def forward(self, token_ids: Tensor, mask: Tensor) -> Tensor:
        """Encode IPA tokens to a fixed-size vector.

        Args:
            token_ids: ``(batch, seq_len)`` token IDs.
            mask: ``(batch, seq_len)`` boolean mask (True = valid).

        Returns:
            ``(batch, output_dim)`` message vector.
        """
        embedded = self.token_embed(token_ids)
        return self.encoder(embedded, mask)


class Receiver(nn.Module):
    """Score candidates against the message via learned dot-product.

    Args:
        dim: Dimensionality of message and candidate vectors.
    """

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.proj = nn.Linear(dim, dim)

    def forward(self, message: Tensor, candidates: Tensor) -> Tensor:
        """Score candidates.

        Args:
            message: ``(batch, dim)`` message vector.
            candidates: ``(batch, K, dim)`` candidate embeddings.

        Returns:
            ``(batch, K)`` logits.
        """
        projected = self.proj(message)
        return torch.bmm(
            projected.unsqueeze(1), candidates.transpose(1, 2),
        ).squeeze(1)

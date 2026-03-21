"""Implicit surface phonology via smoothness pressures.

Replaces explicit phonological machinery (vowel/consonant classifiers, sonority
heads, phoneme inventories) with a single module that maps token embeddings to
continuous surface-form vectors.  Phonotactic constraint emerges from three
implicit pressures:

- **Smoothness**: A GRU predicts each surface vector from preceding ones.
  Prediction error = pronounceability penalty.  Smooth sequences are more
  pronounceable.
- **Energy contour**: Learned scalar energy per surface vector with a soft
  penalty encouraging rise-peak-fall patterns (SSP analog without encoding
  the concept).
- **Diversity**: Batch variance of surface vectors must exceed a floor,
  preventing collapse to a single surface form.
"""

from __future__ import annotations

import logging

import torch
import torch.nn.functional as F  # noqa: N812
from torch import Tensor, nn

from lfm._registry import register
from lfm._types import TokenEmbeddings, TokenIds
from lfm.phonology.base import PhonologyModule
from lfm.phonology.config import PhonologyConfig

logger = logging.getLogger(__name__)


@register("phonology", "surface")
class SurfacePhonology(PhonologyModule):
    """Implicit surface phonology via smoothness pressures.

    Maps token embeddings to continuous surface-form vectors and applies
    three intrinsic regularizers — smoothness, energy contour, and
    diversity — as ``extra_losses()``.  No explicit phonological categories
    are encoded; the "phonemes" are whatever distinct regions of
    surface-form space the model converges on under communication pressure.

    Args:
        config: Phonology configuration specifying surface dimensions,
            smoothness parameters, and loss weights.
    """

    def __init__(self, config: PhonologyConfig) -> None:
        super().__init__(config)

        self.surface_dim = config.surface_dim
        self.max_surface_len = config.max_surface_len
        self.smoothness_weight = config.smoothness_weight
        self.energy_weight = config.energy_weight
        self.diversity_weight = config.diversity_weight
        self.min_variance = config.min_variance
        self.do_enrich = config.enrich

        # Surface projection (lazy-init — input dim unknown until first call)
        self.surface_proj: nn.Linear | None = None

        # Smoothness predictor
        self.smoothness_gru = nn.GRU(
            config.surface_dim,
            config.smoothness_hidden_dim,
            batch_first=True,
        )
        self.smoothness_head = nn.Linear(config.smoothness_hidden_dim, config.surface_dim)

        # Energy contour (SSP analog)
        self.energy_head = nn.Linear(config.surface_dim, 1)

        # Enrichment projection (lazy-init)
        self.enrich_proj: nn.Linear | None = None

        # Forward cache for extra_losses (no detach — gradients flow)
        self._cached_smoothness_loss: Tensor | None = None
        self._cached_energy: Tensor | None = None
        self._cached_surface_forms: Tensor | None = None

        # Load pre-trained smoothness weights if configured
        if config.pretrained_smoothness_path is not None:
            self._load_pretrained_smoothness(config.pretrained_smoothness_path)

    # ------------------------------------------------------------------
    # Pre-trained weight loading
    # ------------------------------------------------------------------

    def _load_pretrained_smoothness(self, path: str) -> None:
        """Load pre-trained smoothness GRU and head from a checkpoint.

        Args:
            path: Path to the ``.pt`` checkpoint file.

        Raises:
            FileNotFoundError: If the checkpoint file does not exist.
            ValueError: If checkpoint dimensions don't match config.
        """
        from pathlib import Path as _Path

        if not _Path(path).exists():
            raise FileNotFoundError(f"Pretrained smoothness checkpoint not found: {path}")

        checkpoint = torch.load(path, map_location="cpu", weights_only=True)

        if checkpoint["surface_dim"] != self.surface_dim:
            raise ValueError(
                f"Checkpoint surface_dim={checkpoint['surface_dim']} does not match "
                f"config surface_dim={self.surface_dim}"
            )
        if checkpoint["smoothness_hidden_dim"] != self.config.smoothness_hidden_dim:
            raise ValueError(
                f"Checkpoint smoothness_hidden_dim={checkpoint['smoothness_hidden_dim']} "
                f"does not match config smoothness_hidden_dim="
                f"{self.config.smoothness_hidden_dim}"
            )

        self.smoothness_gru.load_state_dict(checkpoint["smoothness_gru"])
        self.smoothness_head.load_state_dict(checkpoint["smoothness_head"])
        logger.info("Loaded pre-trained phonotactic priors from %s", path)

    # ------------------------------------------------------------------
    # Lazy initialization
    # ------------------------------------------------------------------

    def _ensure_projections(self, embed_dim: int) -> None:
        """Initialize projection layers on first forward call."""
        if self.surface_proj is None:
            device = next(self.smoothness_gru.parameters()).device
            self.surface_proj = nn.Linear(
                embed_dim, self.max_surface_len * self.surface_dim
            ).to(device)
        if self.do_enrich and self.enrich_proj is None:
            device = next(self.smoothness_gru.parameters()).device
            self.enrich_proj = nn.Linear(
                self.max_surface_len * self.surface_dim, embed_dim
            ).to(device)

    # ------------------------------------------------------------------
    # Core interface
    # ------------------------------------------------------------------

    def to_surface_forms(self, tokens: TokenIds, embeddings: TokenEmbeddings) -> Tensor:
        """Map token embeddings to continuous surface-form vectors.

        Args:
            tokens: Integer token indices ``(batch, seq_len)`` (unused but
                part of the interface for future token-conditioned variants).
            embeddings: Dense token embeddings ``(batch, seq_len, dim)``.

        Returns:
            Surface-form tensor ``(batch, seq_len, max_surface_len, surface_dim)``.
        """
        b, s, d = embeddings.shape
        self._ensure_projections(d)
        assert self.surface_proj is not None

        flat = self.surface_proj(embeddings)  # (b, s, L*C)
        return flat.view(b, s, self.max_surface_len, self.surface_dim)

    def forward(self, tokens: TokenIds, embeddings: TokenEmbeddings) -> dict[str, Tensor]:
        """Run implicit surface phonology.

        1. Project embeddings → surface forms ``(B, S, L, C)``
        2. GRU smoothness: predict next surface vec, compute prediction error
        3. Energy contour: ``energy_head(surface)`` → ``(B, S, L)``
        4. Enrichment: ``embeddings + enrich_proj(flat_surface)``
        5. Cache tensors for ``extra_losses()``

        Args:
            tokens: Integer token indices ``(batch, seq_len)``.
            embeddings: Dense token embeddings ``(batch, seq_len, dim)``.

        Returns:
            Dictionary with ``surface_forms``, ``energy_contour``,
            ``pronounceability_score``, and ``embeddings``.
        """
        b, s, _d = embeddings.shape
        self._ensure_projections(embeddings.size(-1))

        # 1. Surface forms
        surface = self.to_surface_forms(tokens, embeddings)  # (b, s, L, C)

        # 2. Smoothness: GRU prediction error across surface positions
        # Reshape to (b*s, L, C) for per-token surface sequences
        surface_flat = surface.view(b * s, self.max_surface_len, self.surface_dim)

        # Predict position t from positions 0..t-1
        gru_input = surface_flat[:, :-1, :]  # (b*s, L-1, C)
        gru_out, _ = self.smoothness_gru(gru_input)  # (b*s, L-1, H)
        predicted = self.smoothness_head(gru_out)  # (b*s, L-1, C)
        target = surface_flat[:, 1:, :]  # (b*s, L-1, C)

        # MSE prediction error per token
        smoothness_error = F.mse_loss(predicted, target, reduction="none")  # (b*s, L-1, C)
        smoothness_per_token = smoothness_error.mean(dim=(-2, -1))  # (b*s,)
        smoothness_loss = smoothness_per_token.mean()

        # Pronounceability = inverse of smoothness error (higher = more pronounceable)
        pronounceability = 1.0 / (1.0 + smoothness_per_token.view(b, s))  # (b, s)

        # 3. Energy contour
        energy = self.energy_head(surface).squeeze(-1)  # (b, s, L)

        # 4. Enrichment
        if self.do_enrich and self.enrich_proj is not None:
            flat_surface = surface.view(b, s, self.max_surface_len * self.surface_dim)
            embeddings = embeddings + self.enrich_proj(flat_surface)

        # 5. Cache for extra_losses (no detach — gradients flow through)
        self._cached_smoothness_loss = smoothness_loss
        self._cached_energy = energy
        self._cached_surface_forms = surface

        return {
            "surface_forms": surface,
            "energy_contour": energy,
            "pronounceability_score": pronounceability,
            "embeddings": embeddings,
        }

    # ------------------------------------------------------------------
    # Intrinsic losses
    # ------------------------------------------------------------------

    def extra_losses(self) -> dict[str, Tensor]:
        """Return smoothness, energy contour, and diversity losses.

        These are always-active intrinsic regularizers (like VQ-VAE
        commitment loss), not phase-dependent registered losses.
        Computed from tensors cached during ``forward()`` — gradients
        flow through.
        """
        losses: dict[str, Tensor] = {}

        if self._cached_smoothness_loss is None:
            return losses

        # Smoothness loss
        losses["smoothness"] = self._cached_smoothness_loss * self.smoothness_weight

        # Energy contour: soft penalty for non-rise-peak-fall patterns
        assert self._cached_energy is not None
        energy = self._cached_energy  # (B, S, L)
        # Consecutive differences: positive = rising, negative = falling
        diffs = energy[:, :, 1:] - energy[:, :, :-1]  # (B, S, L-1)
        # Find the peak position per token
        peak_idx = energy.argmax(dim=-1, keepdim=True)  # (B, S, 1)
        # Build position indices
        n_pos = energy.size(-1)
        positions = torch.arange(n_pos - 1, device=energy.device)  # (L-1,)
        # Before peak: energy should rise (penalize negative diffs)
        # After peak: energy should fall (penalize positive diffs)
        before_peak = positions.unsqueeze(0).unsqueeze(0) < peak_idx  # (B, S, L-1)
        after_peak = positions.unsqueeze(0).unsqueeze(0) >= peak_idx  # (B, S, L-1)

        rise_penalty = F.relu(-diffs) * before_peak.float()
        fall_penalty = F.relu(diffs) * after_peak.float()
        energy_loss = (rise_penalty + fall_penalty).mean()
        losses["energy_contour"] = energy_loss * self.energy_weight

        # Diversity: batch variance must exceed floor
        assert self._cached_surface_forms is not None
        surface = self._cached_surface_forms  # (B, S, L, C)
        # Variance across batch dimension
        batch_var = surface.var(dim=0).mean()
        diversity_loss = F.relu(self.min_variance - batch_var)
        losses["diversity"] = diversity_loss * self.diversity_weight

        return losses

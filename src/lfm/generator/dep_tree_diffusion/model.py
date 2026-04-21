"""DepTree Diffusion VAE — composes existing encoder/latent/skeleton with diffusion decoder.

Reuses the DepTreeVAE's encoder, latent space, skeleton decoder, and
phrase projector. Replaces only the autoregressive PhraseDecoder with
a TreeDiffusionDecoder that generates all content tokens simultaneously.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from lfm.generator.dep_tree_diffusion.config import DepTreeDiffusionConfig
from lfm.generator.dep_tree_diffusion.decoder import TreeDiffusionDecoder
from lfm.generator.dep_tree_vae.config import NUM_DEP_RELATIONS
from lfm.generator.dep_tree_vae.disentangle import DisentanglementModule
from lfm.generator.dep_tree_vae.encoder import SentenceEncoder
from lfm.generator.dep_tree_vae.latent import LatentSpace
from lfm.generator.dep_tree_vae.projector import PhraseZProjector
from lfm.generator.dep_tree_vae.skeleton import build_skeleton_decoder

logger = logging.getLogger(__name__)


@dataclass
class DiffusionVAEOutput:
    """Container for forward pass outputs."""

    recon_loss: Tensor
    skeleton_loss: Tensor
    kl_loss: Tensor
    dip_loss: Tensor
    disentangle: dict[str, Tensor]
    z_struct: Tensor
    z_content: Tensor
    mu: Tensor
    logvar: Tensor

    @property
    def total_loss(self) -> Tensor:
        return (
            self.recon_loss
            + self.skeleton_loss
            + self.kl_loss
            + self.dip_loss
            + self.disentangle["total"]
        )


class DepTreeDiffusionVAE(nn.Module):
    """VAE with dependency-tree-structured diffusion decoder.

    Shared components (from DepTreeVAE):
      - SentenceEncoder → (mu, logvar)
      - LatentSpace → z_struct, z_content
      - SkeletonDecoder → role sequence from z_struct
      - PhraseZProjector → per-role memory from z_content
      - DisentanglementModule → HSIC + aux losses

    New: TreeDiffusionDecoder replaces autoregressive PhraseDecoder.
    """

    def __init__(self, cfg: DepTreeDiffusionConfig, vocab_size: int) -> None:
        super().__init__()
        self.cfg = cfg
        self.vocab_size = vocab_size

        full_vocab = vocab_size + 3
        decoder_vocab = cfg.spm_vocab_size + 2

        # Shared components (same as DepTreeVAE)
        self.encoder = SentenceEncoder(full_vocab, cfg)
        self.latent = LatentSpace(cfg.latent)
        self.skeleton_decoder = build_skeleton_decoder(
            cfg.skeleton, cfg.latent.struct_dim,
        )
        self.phrase_projector = PhraseZProjector(
            content_dim=cfg.latent.content_dim,
            decoder_hidden_dim=cfg.diffusion.d_model,
            max_roles=cfg.skeleton.max_roles,
        )
        self.disentanglement = DisentanglementModule(
            cfg.disentanglement,
            struct_dim=cfg.latent.struct_dim,
            content_dim=cfg.latent.content_dim,
            content_vocab_size=full_vocab,
        )

        # Tree diffusion decoder (replaces autoregressive PhraseDecoder)
        dcfg = cfg.diffusion
        self.diffusion_decoder = TreeDiffusionDecoder(
            vocab_size=decoder_vocab,
            d_model=dcfg.d_model,
            num_layers=dcfg.num_layers,
            num_heads=dcfg.num_heads,
            dropout=dcfg.dropout,
        )

        # IDs
        self._pad_id = 0
        self._bos_id = cfg.spm_vocab_size
        self._eos_id = cfg.spm_vocab_size + 1
        self._decoder_vocab = decoder_vocab

        # Z stats tracking (for downstream game calibration)
        self.register_buffer("_z_struct_mean", torch.zeros(cfg.latent.struct_dim))
        self.register_buffer("_z_struct_std", torch.ones(cfg.latent.struct_dim))
        self.register_buffer("_z_content_mean", torch.zeros(cfg.latent.content_dim))
        self.register_buffer("_z_content_std", torch.ones(cfg.latent.content_dim))

        # Word dropout rate (set by trainer)
        self._word_dropout_p: float = 0.0

    def forward(
        self,
        tokens: Tensor,
        lengths: Tensor,
        role_ids: Tensor,
        role_lengths: Tensor,
        dep_depths: Tensor,
        kl_weight: float = 0.0,
    ) -> DiffusionVAEOutput:
        """Full forward pass with diffusion decoder.

        Args:
            tokens: (B, S) interleaved role+IPA token ids.
            lengths: (B,) sequence lengths.
            role_ids: (B, R) skeleton role ids (with BOS prefix).
            role_lengths: (B,) role sequence lengths.
            dep_depths: (B, S_content) tree depth per content token position.
            kl_weight: KL annealing weight.
        """
        device = tokens.device
        cfg = self.cfg

        # 1. Encode
        mu, logvar = self.encoder(tokens, lengths)

        # 2. Reparameterize + split
        z_struct, z_content, z = self.latent(mu, logvar)

        # Track z stats
        if self.training:
            with torch.no_grad():
                m = 0.01
                self._z_struct_mean.lerp_(z_struct.mean(dim=0), m)
                self._z_struct_std.lerp_(z_struct.std(dim=0).clamp(min=1e-6), m)
                self._z_content_mean.lerp_(z_content.mean(dim=0), m)
                self._z_content_std.lerp_(z_content.std(dim=0).clamp(min=1e-6), m)

        # 3. Skeleton decoder (teacher-forced)
        skeleton_loss_out = self.skeleton_decoder(
            z_struct, role_ids, role_lengths,
        )
        skeleton_loss = skeleton_loss_out[1] if skeleton_loss_out[1] is not None else torch.tensor(0.0, device=device)

        # 4. Phrase projector → per-role memory
        content_tokens, content_lengths, per_token_roles, per_token_depths = (
            self._extract_content(tokens, lengths, dep_depths)
        )
        skel_roles, skel_mask = self._skeleton_to_padded_roles(role_ids, role_lengths)
        memory = self.phrase_projector(z_content, skel_roles, skel_mask)

        # 5. Diffusion decoder loss
        recon_loss = self._diffusion_loss(
            content_tokens, content_lengths, per_token_roles,
            per_token_depths, memory,
        )

        # 6. KL
        kl_per_dim = 0.5 * (mu.pow(2) + logvar.exp() - 1 - logvar)
        if cfg.kl_free_bits > 0:
            kl_per_dim = kl_per_dim.clamp(min=cfg.kl_free_bits)
        kl_loss = kl_weight * kl_per_dim.sum(dim=-1).mean()

        # 6b. DIP-VAE
        dip_loss = torch.tensor(0.0, device=device)
        if cfg.dip_weight > 0 and z.size(0) >= 4:
            z_c = (z - z.mean(dim=0)).float()
            cov = (z_c.T @ z_c) / max(z.size(0) - 1, 1)
            off = cov - torch.diag(cov.diag())
            dip_loss = cfg.dip_weight * off.pow(2).sum() / z.size(1) ** 2

        # 7. Disentanglement
        role_bow = self._make_role_bow(role_ids, role_lengths, device)
        content_bow = self._make_content_bow(tokens, lengths, device)
        disent = self.disentanglement(z_struct, z_content, role_bow, content_bow)

        return DiffusionVAEOutput(
            recon_loss=recon_loss,
            skeleton_loss=skeleton_loss,
            kl_loss=kl_loss,
            dip_loss=dip_loss,
            disentangle=disent,
            z_struct=z_struct,
            z_content=z_content,
            mu=mu,
            logvar=logvar,
        )

    def _diffusion_loss(
        self,
        content_tokens: Tensor,
        content_lengths: Tensor,
        per_token_roles: Tensor,
        per_token_depths: Tensor,
        memory: Tensor,
    ) -> Tensor:
        """Compute diffusion denoising loss on content tokens.

        Samples a random global timestep, computes per-position noise
        from tree depth, corrupts token embeddings, and trains the
        decoder to predict clean embeddings.
        """
        b = content_tokens.size(0)
        s = content_tokens.size(1)
        device = content_tokens.device

        # Get clean token embeddings
        clamped = content_tokens.clamp(max=self._decoder_vocab - 1)
        x0 = self.diffusion_decoder.token_embedding(clamped)

        # Word dropout on x0 (same principle as autoregressive)
        if self.training and self._word_dropout_p > 0:
            drop = torch.rand(b, s, 1, device=device) < self._word_dropout_p
            x0 = x0.masked_fill(drop, 0.0)

        # Random global timestep
        t_global = torch.rand(b, device=device)

        # Per-position noise from tree depth
        t_per_pos = self.diffusion_decoder.tree_noise_schedule(
            t_global, per_token_depths, self.cfg.diffusion.depth_scale,
        )

        # Corrupt
        x_t, noise = self.diffusion_decoder.add_noise(x0, t_per_pos)

        # Padding mask
        padding_mask = torch.arange(s, device=device).unsqueeze(0) >= content_lengths.unsqueeze(1)

        # Predict clean embeddings
        x0_pred = self.diffusion_decoder(
            x_t, t_per_pos, per_token_roles, per_token_depths,
            memory, padding_mask,
        )

        # Project to logits and compute CE loss (more stable than MSE on embeddings)
        logits = self.diffusion_decoder.output_head(x0_pred)
        valid_mask = ~padding_mask
        flat_logits = logits.reshape(-1, logits.size(-1))
        flat_targets = clamped.reshape(-1)
        flat_mask = valid_mask.reshape(-1)

        if flat_mask.any():
            loss = F.cross_entropy(flat_logits[flat_mask], flat_targets[flat_mask])
        else:
            loss = torch.tensor(0.0, device=device)

        return loss

    def _extract_content(
        self, tokens: Tensor, lengths: Tensor, dep_depths: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """Extract content tokens, their roles, and depths from interleaved sequence."""
        b, s = tokens.shape
        device = tokens.device
        role_offset = self.cfg.spm_vocab_size + 2

        valid = torch.arange(s, device=device).unsqueeze(0) < lengths.unsqueeze(1)
        is_role = (tokens >= role_offset) & valid
        is_content = (~is_role) & (tokens > 0) & valid

        # Forward-fill role IDs
        role_vals = torch.where(is_role, tokens - role_offset, torch.zeros_like(tokens))
        role_group = is_role.long().cumsum(dim=1)
        max_groups = role_group.max().item() + 1
        group_role = torch.zeros(b, max_groups, dtype=torch.long, device=device)
        group_role.scatter_(1, role_group, role_vals)
        per_pos_role = group_role.gather(1, role_group).clamp(max=NUM_DEP_RELATIONS - 1)

        # Compact content tokens
        content_counts = is_content.sum(dim=1)
        max_content = content_counts.max().item()

        content_tokens = torch.zeros(b, max_content, dtype=torch.long, device=device)
        content_roles = torch.zeros(b, max_content, dtype=torch.long, device=device)
        content_depths = torch.zeros(b, max_content, dtype=torch.long, device=device)

        for i in range(b):
            idx = is_content[i].nonzero(as_tuple=True)[0]
            n = idx.numel()
            content_tokens[i, :n] = tokens[i, idx]
            content_roles[i, :n] = per_pos_role[i, idx]
            if dep_depths.size(1) >= n:
                content_depths[i, :n] = dep_depths[i, :n]

        return content_tokens, content_counts, content_roles, content_depths

    def _skeleton_to_padded_roles(
        self, role_ids: Tensor, role_lengths: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """Convert teacher-forced skeleton to padded roles + mask for projector."""
        from lfm.generator.dep_tree_vae.skeleton import SKEL_BOS, SKEL_EOS

        b = role_ids.size(0)
        device = role_ids.device
        roles_no_bos = role_ids[:, 1:]
        max_r = roles_no_bos.size(1)
        actual_len = (role_lengths - 1).clamp(min=1)

        mask = torch.arange(max_r, device=device).unsqueeze(0) < actual_len.unsqueeze(1)
        clamped = roles_no_bos.clamp(max=NUM_DEP_RELATIONS - 1)
        # Zero out past-length positions
        clamped = clamped * mask.long()

        return clamped, mask

    def _make_role_bow(self, role_ids: Tensor, role_lengths: Tensor, device: torch.device) -> Tensor:
        b = role_ids.size(0)
        bow = torch.zeros(b, NUM_DEP_RELATIONS, device=device)
        mask = torch.arange(role_ids.size(1), device=device).unsqueeze(0) < role_lengths.unsqueeze(1)
        for i in range(b):
            valid = role_ids[i][mask[i]]
            valid = valid[valid < NUM_DEP_RELATIONS]
            if valid.numel() > 0:
                bow[i].scatter_(0, valid, 1.0)
        return bow

    def _make_content_bow(self, tokens: Tensor, lengths: Tensor, device: torch.device) -> Tensor:
        b = tokens.size(0)
        role_offset = self.cfg.spm_vocab_size + 2
        bow = torch.zeros(b, self.vocab_size + 3, device=device)
        mask = torch.arange(tokens.size(1), device=device).unsqueeze(0) < lengths.unsqueeze(1)
        for i in range(b):
            valid = tokens[i][mask[i]]
            content = valid[valid < role_offset]
            if content.numel() > 0:
                bow[i].scatter_add_(0, content, torch.ones_like(content, dtype=torch.float))
        return bow

    def trainable_parameters(self) -> list[dict]:
        """Parameter groups for optimizer."""
        return [
            {"params": list(self.encoder.parameters())},
            {"params": list(self.latent.parameters())},
            {"params": list(self.skeleton_decoder.parameters())},
            {"params": list(self.phrase_projector.parameters())},
            {"params": list(self.diffusion_decoder.parameters())},
            {"params": list(self.disentanglement.parameters())},
        ]

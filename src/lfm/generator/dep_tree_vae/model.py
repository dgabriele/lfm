"""DepTreeVAE — top-level model combining all components."""

from __future__ import annotations

import logging
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from lfm.generator.dep_tree_vae.config import (
    DEP_REL_TO_ID,
    NUM_DEP_RELATIONS,
    DepTreeVAEConfig,
)
from lfm.generator.dep_tree_vae.disentangle import DisentanglementModule
from lfm.generator.dep_tree_vae.encoder import SentenceEncoder
from lfm.generator.dep_tree_vae.latent import LatentSpace
from lfm.generator.dep_tree_vae.projector import PhraseZProjector
from lfm.generator.dep_tree_vae.skeleton import (
    SKEL_BOS,
    SKEL_EOS,
    build_skeleton_decoder,
)
from lfm.generator.layers import PhraseDecoder, precompute_rope_freqs

logger = logging.getLogger(__name__)


@dataclass
class DepTreeVAEOutput:
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


class DepTreeVAE(nn.Module):
    """VAE with dependency-tree-disentangled latent space.

    The latent vector is split into:
        - **z_struct** — encodes syntactic skeleton (dependency role sequence)
        - **z_content** — encodes lexical content (which words fill each role)

    Generation is single-pass autoregressive: the decoder receives
    per-position memory that combines z_content with role and position
    conditioning.  The skeleton decoder provides the role sequence
    from z_struct, and the phrase decoder generates IPA tokens
    conditioned on the per-position memory.

    Architecture:
        Encoder → (mu, logvar) → split(z_struct, z_content)
        z_struct → SkeletonDecoder → role sequence
        z_content + roles + positions → PhraseZProjector → per-position memory
        memory → PhraseDecoder (frozen) → IPA tokens
    """

    def __init__(self, cfg: DepTreeVAEConfig, vocab_size: int) -> None:
        super().__init__()
        self.cfg = cfg
        self.vocab_size = vocab_size

        # Encoder sees the full interleaved vocab (SPM + specials + role tokens)
        full_vocab = vocab_size + 3
        # Decoder only generates IPA tokens (SPM + BOS + EOS), not role tokens
        decoder_vocab = cfg.spm_vocab_size + 2

        # Components
        self.encoder = SentenceEncoder(full_vocab, cfg)
        self.latent = LatentSpace(cfg.latent)
        self.skeleton_decoder = build_skeleton_decoder(
            cfg.skeleton, cfg.latent.struct_dim,
        )
        self.phrase_projector = PhraseZProjector(
            content_dim=cfg.latent.content_dim,
            decoder_hidden_dim=cfg.decoder_hidden_dim,
            max_roles=cfg.skeleton.max_roles,
        )
        self.disentanglement = DisentanglementModule(
            cfg.disentanglement,
            struct_dim=cfg.latent.struct_dim,
            content_dim=cfg.latent.content_dim,
            content_vocab_size=full_vocab,
        )

        # Phrase decoder (can be loaded from pretrained checkpoint)
        h = cfg.decoder_hidden_dim
        self.phrase_decoder = PhraseDecoder(
            d_model=h,
            nhead=cfg.decoder_num_heads,
            num_layers=cfg.decoder_num_layers,
            dim_feedforward=h * 4,
            dropout=cfg.decoder_dropout,
            share_layers=cfg.share_decoder_layers,
        )
        self.dec_token_embedding = nn.Embedding(decoder_vocab, h)
        self.dec_pos_embedding: nn.Module
        if cfg.use_rope:
            self.dec_pos_embedding = nn.Identity()
            self.register_buffer(
                "_rope_freqs",
                precompute_rope_freqs(h // cfg.decoder_num_heads, cfg.max_seq_len),
            )
        else:
            self.dec_pos_embedding = nn.Embedding(cfg.max_seq_len, h)
            self.register_buffer("_rope_freqs", None)

        self.output_head = nn.Linear(h, decoder_vocab)
        self._decoder_vocab = decoder_vocab

        # IDs (match pretrained decoder convention)
        self._pad_id = 0
        self._bos_id = cfg.spm_vocab_size      # 8000
        self._eos_id = cfg.spm_vocab_size + 1   # 8001

        # Word dropout rate (set by trainer per step via annealing schedule)
        self._word_dropout_p: float = 0.0

        # Z distribution stats (updated during training for downstream calibration)
        self.register_buffer("_z_struct_mean", torch.zeros(cfg.latent.struct_dim))
        self.register_buffer("_z_struct_std", torch.ones(cfg.latent.struct_dim))
        self.register_buffer("_z_content_mean", torch.zeros(cfg.latent.content_dim))
        self.register_buffer("_z_content_std", torch.ones(cfg.latent.content_dim))

        # Load pretrained decoder if configured
        if cfg.pretrained_decoder_path:
            self._load_pretrained_decoder(cfg.pretrained_decoder_path)
        if cfg.freeze_decoder:
            self._freeze_decoder()

    def _load_pretrained_decoder(self, path: str) -> None:
        """Load decoder weights from a pretrained checkpoint."""
        ckpt = torch.load(path, map_location="cpu", weights_only=True)
        self.phrase_decoder.load_state_dict(ckpt["decoder"])
        self.output_head.load_state_dict(ckpt["output_head"])
        self.dec_token_embedding.load_state_dict(ckpt["token_embedding"])
        if "pos_embedding" in ckpt and not isinstance(
            self.dec_pos_embedding, nn.Identity
        ):
            self.dec_pos_embedding.load_state_dict(ckpt["pos_embedding"])
        logger.info("Loaded pretrained decoder from %s", path)

    def _freeze_decoder(self) -> None:
        """Freeze all decoder parameters."""
        for module in (
            self.phrase_decoder, self.output_head,
            self.dec_token_embedding, self.dec_pos_embedding,
        ):
            for p in module.parameters():
                p.requires_grad = False
        logger.info("Froze phrase decoder parameters")

    def forward(
        self,
        tokens: Tensor,
        lengths: Tensor,
        role_ids: Tensor,
        role_lengths: Tensor,
        kl_weight: float = 0.0,
    ) -> DepTreeVAEOutput:
        """Full forward pass: encode → split → skeleton + reconstruct.

        Args:
            tokens: ``(B, S)`` interleaved role+IPA token ids.
            lengths: ``(B,)`` sequence lengths.
            role_ids: ``(B, R)`` dependency role ids for the skeleton.
                Includes BOS prefix for teacher forcing.
            role_lengths: ``(B,)`` role sequence lengths (including BOS).
            kl_weight: KL annealing weight.

        Returns:
            DepTreeVAEOutput with all losses and latent vectors.
        """
        device = tokens.device

        # 1. Encode
        mu, logvar = self.encoder(tokens, lengths)

        # 2. Reparameterize + split
        z_struct, z_content, z = self.latent(mu, logvar)

        # Track running z stats for downstream calibration
        if self.training:
            with torch.no_grad():
                momentum = 0.01
                self._z_struct_mean.lerp_(z_struct.mean(dim=0), momentum)
                self._z_struct_std.lerp_(z_struct.std(dim=0).clamp(min=1e-6), momentum)
                self._z_content_mean.lerp_(z_content.mean(dim=0), momentum)
                self._z_content_std.lerp_(z_content.std(dim=0).clamp(min=1e-6), momentum)

        # 3. Skeleton decoder (teacher-forced on role sequence)
        _, skeleton_loss = self.skeleton_decoder(
            z_struct, role_ids, role_lengths,
        )

        # 4. Build per-role memory from z_content + skeleton roles
        content_tokens, content_lengths, content_roles, _ = (
            self._split_roles_and_content(tokens, lengths)
        )
        content_lengths = content_lengths.clamp(min=1)

        # Get the skeleton role sequence for memory construction.
        # During training we use the ground-truth roles (teacher forcing).
        # role_ids has BOS prefix — strip it and extract actual roles.
        skel_roles = role_ids[:, 1:]  # strip BOS
        skel_mask = torch.arange(skel_roles.size(1), device=device).unsqueeze(0) < (role_lengths - 1).unsqueeze(1)
        # Clamp to valid role range
        skel_roles = skel_roles.clamp(max=NUM_DEP_RELATIONS - 1)

        memory = self.phrase_projector(z_content, skel_roles, skel_mask)

        # 5. Reconstruct through the phrase decoder
        recon_loss = self._decode_and_loss(tokens, lengths, memory, content_tokens, content_lengths)

        # 6. KL divergence
        kl_per_dim = 0.5 * (mu.pow(2) + logvar.exp() - 1 - logvar)
        if self.cfg.kl_free_bits > 0:
            kl_per_dim = kl_per_dim.clamp(min=self.cfg.kl_free_bits)
        kl_loss = kl_weight * kl_per_dim.sum(dim=-1).mean()

        # 6b. DIP-VAE: off-diagonal covariance penalty on aggregate posterior
        dip_loss = torch.tensor(0.0, device=device)
        if self.cfg.dip_weight > 0 and z.size(0) >= 4:
            z_centered = (z - z.mean(dim=0)).float()
            cov = (z_centered.T @ z_centered) / max(z.size(0) - 1, 1)
            off_diag = cov - torch.diag(cov.diag())
            dip_loss = self.cfg.dip_weight * off_diag.pow(2).sum() / z.size(1) ** 2

        # 7. Disentanglement losses
        role_bow = self._make_role_bow(role_ids, role_lengths, device)
        content_bow = self._make_content_bow(tokens, lengths, device)
        disent = self.disentanglement(
            z_struct, z_content, role_bow, content_bow,
        )

        return DepTreeVAEOutput(
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

    def _split_roles_and_content(
        self, tokens: Tensor, lengths: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """Split interleaved sequence into role ids and content token ids.

        Role tokens have ids >= role_offset.  Each role token applies
        to all subsequent content tokens until the next role token.
        Fully vectorized — no Python loops over batch/positions.

        Returns:
            content_tokens: ``(B, max_content)`` IPA token ids.
            content_lengths: ``(B,)`` number of content tokens per sample.
            content_roles: ``(B, max_content)`` dep role id per content token.
            content_positions: ``(B, max_content)`` position indices.
        """
        b, s = tokens.shape
        device = tokens.device
        role_offset = self.cfg.spm_vocab_size + 2

        # Masks
        valid = torch.arange(s, device=device).unsqueeze(0) < lengths.unsqueeze(1)
        is_role = (tokens >= role_offset) & valid
        is_content = (~is_role) & (tokens > 0) & valid

        # Propagate role IDs forward: each content token inherits
        # the most recent role token's ID.
        # Extract raw role values (0 where not a role token)
        role_vals = torch.where(is_role, tokens - role_offset, torch.zeros_like(tokens))
        # cummax trick: at each position, the role ID is the last
        # non-zero role_vals seen. Use cumsum of is_role as a group
        # index, then scatter the role values.
        role_group = is_role.long().cumsum(dim=1)  # (B, S) — increments at each role token
        # For each group, the role value is at the first position of that group
        # Broadcast: role_at_pos[i,j] = role_vals[i, position of group start]
        # Simpler: just use the role_vals * is_role and forward-fill
        # Forward fill via cummax on (role_group * large_val + role_vals)
        # Actually simplest: scattered gather
        role_ids_at_role_pos = role_vals  # only non-zero where is_role
        # Forward-fill: for each position, take the role_id from the
        # last role position. Use scatter with role_group as index.
        max_groups = role_group.max().item() + 1
        group_role = torch.zeros(b, max_groups, dtype=torch.long, device=device)
        group_role.scatter_(1, role_group, role_ids_at_role_pos)
        # Now each content position's role = group_role[sample, role_group[sample, pos]]
        per_pos_role = group_role.gather(1, role_group).clamp(max=NUM_DEP_RELATIONS - 1)

        # Extract content tokens into compact form
        content_counts = is_content.sum(dim=1)  # (B,)
        max_content = max(content_counts.max().item(), 1)

        # Sort content positions to front per sample using argsort trick
        # is_content as float, negate so True sorts first
        sort_keys = (~is_content).long()  # 0 for content, 1 for non-content
        sorted_indices = sort_keys.argsort(dim=1, stable=True)  # content positions first

        content_tokens = tokens.gather(1, sorted_indices)[:, :max_content]
        content_roles = per_pos_role.gather(1, sorted_indices)[:, :max_content]
        content_lengths = content_counts.clamp(min=1)
        positions = torch.arange(max_content, device=device).unsqueeze(0).expand(b, -1)

        return content_tokens, content_lengths, content_roles, positions

    def _decode_and_loss(
        self,
        tokens: Tensor,
        lengths: Tensor,
        memory: Tensor,
        content_tokens: Tensor,
        content_lengths: Tensor,
    ) -> Tensor:
        """Run phrase decoder on content tokens with role-level memory.

        The decoder cross-attends to ``memory`` (one vector per skeleton
        role) while autoregressively generating content tokens.  The
        cross-attention learns to shift focus across roles as it fills
        successive syntactic slots — no forced positional alignment.
        """
        from lfm.generator.layers import multiscale_causal_mask

        b = content_tokens.size(0)
        device = content_tokens.device

        content_tokens = content_tokens.clamp(max=self._decoder_vocab - 1)

        # Teacher forcing: shift right with BOS
        bos = torch.full(
            (b, 1), self._bos_id, dtype=torch.long, device=device,
        )
        decoder_input_ids = torch.cat(
            [bos, content_tokens[:, :-1]], dim=1,
        )

        cs = decoder_input_ids.size(1)
        dec_input = self.dec_token_embedding(decoder_input_ids)

        # Word dropout: zero out random token embeddings to force
        # the decoder to rely on z (cross-attention memory)
        if self.training and self._word_dropout_p > 0:
            drop_mask = torch.rand(b, cs, 1, device=device) < self._word_dropout_p
            drop_mask[:, 0] = False  # never drop BOS
            dec_input = dec_input.masked_fill(drop_mask, 0.0)
        if not isinstance(self.dec_pos_embedding, nn.Identity):
            pos = torch.arange(cs, device=device).unsqueeze(0)
            dec_input = dec_input + self.dec_pos_embedding(pos)

        tgt_mask = multiscale_causal_mask(
            cs,
            self.cfg.decoder_num_heads,
            tuple(self.cfg.attention_head_windows),
            self.cfg.attention_global_every,
            device=device,
        )
        rope = self._rope_freqs[:cs] if self._rope_freqs is not None else None

        # memory is (B, num_roles, h) — decoder cross-attends to all roles
        hidden = self.phrase_decoder(
            dec_input, memory, tgt_mask=tgt_mask, rope_freqs=rope,
        )
        logits = self.output_head(hidden)

        # Masked CE loss over content tokens
        mask = torch.arange(cs, device=device).unsqueeze(0) < content_lengths.unsqueeze(1)
        flat_logits = logits.reshape(-1, logits.size(-1))
        flat_targets = content_tokens[:, :cs].reshape(-1)
        flat_mask = mask.reshape(-1)

        loss = F.cross_entropy(
            flat_logits[flat_mask], flat_targets[flat_mask],
        )
        return loss

    def _make_role_bow(
        self, role_ids: Tensor, role_lengths: Tensor, device: torch.device,
    ) -> Tensor:
        """Create multi-hot role presence vector."""
        b = role_ids.size(0)
        bow = torch.zeros(b, NUM_DEP_RELATIONS, device=device)
        mask = torch.arange(role_ids.size(1), device=device).unsqueeze(0) < role_lengths.unsqueeze(1)
        valid = role_ids.clone()
        valid[~mask] = 0
        valid = valid.clamp(max=NUM_DEP_RELATIONS - 1)
        bow.scatter_(1, valid, 1.0)
        bow[:, 0] = 0  # Don't count padding as a role
        return bow

    def _make_content_bow(
        self, tokens: Tensor, lengths: Tensor, device: torch.device,
    ) -> Tensor:
        """Create multi-hot content word presence vector."""
        b = tokens.size(0)
        cv = self.disentanglement.content_predictor[-1].out_features
        bow = torch.zeros(b, cv, device=device)
        mask = torch.arange(tokens.size(1), device=device).unsqueeze(0) < lengths.unsqueeze(1)
        # Only count content tokens (odd positions)
        content_mask = mask.clone()
        content_mask[:, ::2] = False  # zero out role positions
        valid = tokens.clone()
        valid[~content_mask] = 0
        bow.scatter_(1, valid, 1.0)
        bow[:, 0] = 0  # Don't count padding
        return bow

    def trainable_parameters(self) -> list[dict]:
        """Parameter groups for optimizer setup."""
        frozen = set()
        if self.cfg.freeze_decoder:
            for m in (
                self.phrase_decoder, self.output_head,
                self.dec_token_embedding, self.dec_pos_embedding,
            ):
                frozen.update(id(p) for p in m.parameters())

        trainable = [
            p for p in self.parameters()
            if p.requires_grad and id(p) not in frozen
        ]
        return [{"params": trainable}]

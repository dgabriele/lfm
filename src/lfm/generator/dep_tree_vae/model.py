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

        # Full vocab includes specials: PAD=0, BOS=1, EOS=2
        full_vocab = vocab_size + 3

        # Components
        self.encoder = SentenceEncoder(full_vocab, cfg)
        self.latent = LatentSpace(cfg.latent)
        self.skeleton_decoder = build_skeleton_decoder(
            cfg.skeleton, cfg.latent.struct_dim,
        )
        self.phrase_projector = PhraseZProjector(
            content_dim=cfg.latent.content_dim,
            decoder_hidden_dim=cfg.decoder_hidden_dim,
            max_seq_len=cfg.max_seq_len,
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
        self.dec_token_embedding = nn.Embedding(full_vocab, h)
        self.dec_pos_embedding: nn.Module
        if cfg.use_rope:
            self.dec_pos_embedding = nn.Identity()
            self._rope_freqs = precompute_rope_freqs(
                h // cfg.decoder_num_heads, cfg.max_seq_len,
            )
        else:
            self.dec_pos_embedding = nn.Embedding(cfg.max_seq_len, h)
            self._rope_freqs = None

        self.output_head = nn.Linear(h, full_vocab)

        # IDs
        self._pad_id = 0
        self._bos_id = 1
        self._eos_id = 2

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

        # 3. Skeleton decoder (teacher-forced on role sequence)
        _, skeleton_loss = self.skeleton_decoder(
            z_struct, role_ids, role_lengths,
        )

        # 4. Build per-position memory from z_content + roles
        # Extract role ids per IPA token position (every other token
        # in the interleaved sequence is a role token).
        content_roles, content_positions = self._extract_content_roles(
            tokens, lengths,
        )
        memory = self.phrase_projector(
            z_content, content_roles, content_positions,
        )

        # 5. Reconstruct through the phrase decoder
        recon_loss = self._decode_and_loss(tokens, lengths, memory)

        # 6. KL divergence
        kl_per_dim = 0.5 * (mu.pow(2) + logvar.exp() - 1 - logvar)
        if self.cfg.kl_free_bits > 0:
            kl_per_dim = kl_per_dim.clamp(min=self.cfg.kl_free_bits)
        kl_loss = kl_weight * kl_per_dim.sum(dim=-1).mean()

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
            disentangle=disent,
            z_struct=z_struct,
            z_content=z_content,
            mu=mu,
            logvar=logvar,
        )

    def _extract_content_roles(
        self, tokens: Tensor, lengths: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """Extract per-content-token role ids and positions.

        In the interleaved sequence ``[role] word [role] word ...``,
        even positions (0, 2, 4...) are roles, odd positions are content.
        Returns role_id and position for each content token.
        """
        b, s = tokens.shape
        device = tokens.device

        # Content tokens are at odd positions; their role is the
        # preceding even position.  For positions beyond the sequence
        # length, use pad (0).
        max_content = s // 2
        content_roles = torch.zeros(b, max_content, dtype=torch.long, device=device)
        positions = torch.arange(max_content, device=device).unsqueeze(0).expand(b, -1)

        for i in range(max_content):
            role_pos = 2 * i
            content_roles[:, i] = tokens[:, role_pos].clamp(max=NUM_DEP_RELATIONS - 1)

        return content_roles, positions

    def _decode_and_loss(
        self,
        tokens: Tensor,
        lengths: Tensor,
        memory: Tensor,
    ) -> Tensor:
        """Run phrase decoder on content tokens with per-position memory.

        The decoder sees content tokens (IPA words) and cross-attends
        to the per-position memory from the projector.
        """
        from lfm.generator.layers import multiscale_causal_mask

        b, s = tokens.shape
        device = tokens.device

        # Extract content tokens (odd positions in interleaved sequence)
        max_content = s // 2
        content_tokens = tokens[:, 1::2][:, :max_content]
        content_lengths = (lengths // 2).clamp(min=1)

        # Teacher forcing: shift right with BOS
        bos = torch.full(
            (b, 1), self._bos_id, dtype=torch.long, device=device,
        )
        decoder_input_ids = torch.cat(
            [bos, content_tokens[:, :-1]], dim=1,
        )

        cs = decoder_input_ids.size(1)
        dec_input = self.dec_token_embedding(decoder_input_ids)
        if not isinstance(self.dec_pos_embedding, nn.Identity):
            pos = torch.arange(cs, device=device).unsqueeze(0)
            dec_input = dec_input + self.dec_pos_embedding(pos)

        tgt_mask = multiscale_causal_mask(
            cs,
            self.cfg.decoder_num_heads,
            tuple(self.cfg.attention_head_windows),
            self.cfg.attention_global_every,
        )
        rope = self._rope_freqs[:cs].to(device) if self._rope_freqs is not None else None

        # Truncate memory to match decoder sequence length
        mem = memory[:, :cs, :]

        hidden = self.phrase_decoder(
            dec_input, mem, tgt_mask=tgt_mask, rope_freqs=rope,
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
        full_vocab = self.output_head.out_features
        bow = torch.zeros(b, full_vocab, device=device)
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

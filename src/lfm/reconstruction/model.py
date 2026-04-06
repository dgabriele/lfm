"""Reconstruction model: z-gen → frozen decoder → inverse decoder.

Jointly trains the z-generator and inverse decoder to preserve
input embedding information through the frozen linguistic bottleneck.
The language IS the encoding — the inverse decoder recovers what
was encoded from the surface-level IPA tokens.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F  # noqa: N812
from torch import Tensor, nn

from lfm.agents.components import ZDiversityLoss, embed_tokens_straight_through
from lfm.agents.decode import (
    ExpressionDecoder,
    _compute_phrase_assignment,
    rerun_decoder_multiphrase_with_grad,
)
from lfm.agents.diffusion import DiffusionZGenerator
from lfm.faculty.model import LanguageFaculty
from lfm.reconstruction.config import ReconstructionConfig
from lfm.reconstruction.inverse_decoder import InverseDecoder


class ReconstructionModel(nn.Module):
    """Reconstruction through the linguistic bottleneck.

    Pipeline::

        embedding
          → DiffusionZGenerator → K z-vectors
          → ExpressionDecoder (Phase 1: AR decode, no grad)
          → Phase 2: decoder rerun with grad → hidden states
          → embed_tokens_straight_through → differentiable token repr
          → InverseDecoder → reconstructed embedding
          → cosine similarity loss to original

    The z-generator learns what to encode.  The inverse decoder learns
    how to decode.  The frozen PhraseDecoder constrains everything to
    be linguistically structured.

    Args:
        config: Reconstruction configuration.
        faculty: Pre-built ``LanguageFaculty`` with frozen decoder.
    """

    def __init__(
        self, config: ReconstructionConfig, faculty: LanguageFaculty,
    ) -> None:
        super().__init__()
        self.config = config
        self.faculty = faculty

        gen = faculty.generator
        gen.eval()
        device = next(gen.parameters()).device
        with torch.no_grad():
            faculty(torch.randn(1, config.embedding_dim, device=device))

        hidden_dim = gen.config.decoder_hidden_dim

        # Z-generator: embedding → K z-vectors
        self.z_gen = DiffusionZGenerator(
            input_dim=config.embedding_dim,
            latent_dim=gen._latent_dim,
            d_model=config.z_hidden_dim,
            max_phrases=config.max_phrases,
            num_steps=config.diffusion_steps,
            num_layers=config.diffusion_layers,
            num_heads=config.diffusion_heads,
            variable_phrases=False,
            z_mean=gen._z_mean if gen._z_stats_initialized else None,
            z_std=gen._z_std if gen._z_stats_initialized else None,
        )

        # Phase 1 decoder (shared, frozen)
        self.decoder = ExpressionDecoder(gen)

        # Inverse decoder: IPA tokens → embedding
        self.inverse = InverseDecoder(
            token_dim=hidden_dim,
            output_dim=config.embedding_dim,
            num_heads=config.inverse_num_heads,
            num_layers=config.inverse_num_layers,
        )

        # Z diversity regularization
        if config.z_diversity_weight > 0 and gen._z_stats_initialized:
            self.z_diversity = ZDiversityLoss(gen._z_mean, gen._z_std)
        else:
            self.z_diversity = None

    @property
    def gen(self):
        return self.faculty.generator

    def param_groups(self) -> list[dict]:
        """Return optimizer parameter groups."""
        cfg = self.config
        return [
            {"params": list(self.z_gen.parameters()), "lr": cfg.z_gen_lr},
            {"params": list(self.inverse.parameters()), "lr": cfg.inverse_lr},
        ]

    def forward(self, embeddings: Tensor) -> dict[str, Tensor]:
        """Forward pass: encode → decode → reconstruct.

        Args:
            embeddings: ``(B, embedding_dim)`` input embeddings.

        Returns:
            Dict with ``loss``, ``cosine_sim``, ``num_phrases``,
            ``total_tokens``.
        """
        cfg = self.config
        device = embeddings.device

        # Z-generation
        z_seq, z_weights, num_phrases = self.z_gen(embeddings)

        # Phase 1: AR decode (no grad)
        tokens, gen_mask, bounds = self.decoder.decode(z_seq, z_weights)

        # Phase 2: decoder rerun with gradients
        hidden = rerun_decoder_multiphrase_with_grad(
            self.gen, z_seq, z_weights, tokens, gen_mask, bounds,
        )
        trimmed_mask = gen_mask[:, :hidden.size(1)]

        # Straight-through token representations (what the LLM sees)
        token_repr = embed_tokens_straight_through(
            hidden, self.gen.output_head, self.gen.token_embedding,
        )

        # Reconstruct embedding from tokens
        reconstructed = self.inverse(token_repr, trimmed_mask)

        # Reconstruction loss: cosine similarity
        cos_sim = F.cosine_similarity(embeddings, reconstructed, dim=-1)
        recon_loss = (1 - cos_sim).mean()

        loss = recon_loss

        # Z diversity regularization
        if self.z_diversity is not None:
            div_loss, _ = self.z_diversity(z_seq, z_weights)
            loss = loss + cfg.z_diversity_weight * div_loss

        with torch.no_grad():
            total_tokens = trimmed_mask.float().sum(dim=1).mean()

        return {
            "loss": loss,
            "cosine_sim": cos_sim.mean().detach(),
            "recon_loss": recon_loss.detach(),
            "num_phrases": num_phrases.mean().detach(),
            "total_tokens": total_tokens,
        }

    def checkpoint_state(self) -> dict:
        return {
            "z_gen": self.z_gen.state_dict(),
            "inverse": self.inverse.state_dict(),
            "config": self.config.model_dump(),
        }

    def load_checkpoint_state(self, ckpt: dict) -> None:
        self.z_gen.load_state_dict(ckpt["z_gen"])
        self.inverse.load_state_dict(ckpt["inverse"])

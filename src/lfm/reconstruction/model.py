"""Reconstruction model with dual-path z-gen training.

Two reconstruction paths, one z-generator:

1. **Direct path** (strong z-gen gradient): pool z-vectors → MLP →
   predicted embedding.  Gives the z-gen immediate gradient to encode
   maximal information about the input.

2. **Surface path** (linguistic constraint): z-vectors → frozen decoder
   → surface tokens → HiddenStatePredictor → InverseDecoder →
   reconstructed embedding.  Forces the z-gen to produce z-vectors
   that decode into linguistically structured, recoverable IPA.

Both paths use contrastive loss: the reconstructed embedding must be
closer to its source than to other embeddings in the batch.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F  # noqa: N812
from torch import Tensor, nn

from lfm.agents.components import ZDiversityLoss, embed_tokens_straight_through
from lfm.agents.decode import (
    ExpressionDecoder,
    rerun_decoder_multiphrase_with_grad,
)
from lfm.agents.diffusion import DiffusionZGenerator
from lfm.faculty.model import LanguageFaculty
from lfm.reconstruction.config import ReconstructionConfig
from lfm.reconstruction.inverse_decoder import HiddenStatePredictor, InverseDecoder


class DirectZHead(nn.Module):
    """Pool z-vectors and project directly to embedding space.

    Bypasses the decoder entirely — gives the z-gen a direct gradient
    signal to be maximally informative about the input embedding.

    Args:
        latent_dim: Dimension of each z-vector.
        output_dim: Dimension of the target embedding.
        hidden_dim: MLP hidden dimension.
    """

    def __init__(
        self, latent_dim: int, output_dim: int, hidden_dim: int = 512,
    ) -> None:
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, z_seq: Tensor, z_weights: Tensor) -> Tensor:
        """Reconstruct embedding from weighted z-vectors.

        Args:
            z_seq: ``(B, K, latent_dim)`` z-vector sequence.
            z_weights: ``(B, K)`` activity weights.

        Returns:
            ``(B, output_dim)`` predicted embedding.
        """
        # Weighted mean of z-vectors
        w = z_weights.unsqueeze(-1)
        pooled = (z_seq * w).sum(dim=1) / w.sum(dim=1).clamp(min=1)
        return self.proj(pooled)


class ReconstructionModel(nn.Module):
    """Dual-path reconstruction through the linguistic bottleneck.

    Pipeline::

        embedding → z_gen → z-vectors
            ├─ Direct path:  pool(z) → MLP → z_recon (strong z-gen gradient)
            └─ Surface path: frozen decoder → tokens → predictor → inverse → surface_recon
                             (hidden states supervise predictor via MSE)

    All reconstruction losses are contrastive: reconstructed embedding
    must be closer to its source than to batch negatives.

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

        # Direct path: z-vectors → embedding (bypasses decoder)
        self.direct_head = DirectZHead(
            latent_dim=gen._latent_dim,
            output_dim=config.embedding_dim,
            hidden_dim=config.z_hidden_dim,
        )

        # Phase 1 decoder (shared, frozen)
        self.decoder = ExpressionDecoder(gen)

        # Surface path stage 1: surface tokens → predicted hidden states
        self.predictor = HiddenStatePredictor(
            token_dim=hidden_dim,
            num_heads=config.predictor_num_heads,
            num_layers=config.predictor_num_layers,
        )

        # Surface path stage 2: predicted hidden states → embedding
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
            {"params": list(self.direct_head.parameters()), "lr": cfg.inverse_lr},
            {"params": list(self.predictor.parameters()), "lr": cfg.inverse_lr},
            {"params": list(self.inverse.parameters()), "lr": cfg.inverse_lr},
        ]

    @staticmethod
    def _contrastive_loss(
        reconstructed: Tensor, targets: Tensor, temperature: float = 0.1,
    ) -> tuple[Tensor, Tensor]:
        """Batch-contrastive loss: reconstructed must match its own target.

        Args:
            reconstructed: ``(B, D)`` predicted embeddings.
            targets: ``(B, D)`` ground truth embeddings.
            temperature: Softmax temperature.

        Returns:
            loss: Scalar contrastive loss.
            accuracy: Fraction of correct top-1 matches.
        """
        recon_norm = F.normalize(reconstructed, dim=-1)
        target_norm = F.normalize(targets, dim=-1)
        # (B, B) similarity matrix — diagonal is the correct match
        logits = recon_norm @ target_norm.T / temperature
        labels = torch.arange(logits.size(0), device=logits.device)
        loss = F.cross_entropy(logits, labels)
        acc = (logits.argmax(dim=-1) == labels).float().mean()
        return loss, acc

    def forward(self, embeddings: Tensor) -> dict[str, Tensor]:
        """Dual-path forward pass.

        Args:
            embeddings: ``(B, embedding_dim)`` input embeddings.

        Returns:
            Dict with ``loss``, ``cosine_sim``, ``direct_acc``,
            ``surface_acc``, ``hs_loss``, ``num_phrases``, ``total_tokens``.
        """
        cfg = self.config

        # Z-generation
        z_seq, z_weights, num_phrases = self.z_gen(embeddings)

        # === Direct path (strong z-gen gradient) ===
        z_recon = self.direct_head(z_seq, z_weights)
        direct_loss, direct_acc = self._contrastive_loss(z_recon, embeddings)

        # === Surface path (linguistic constraint) ===
        # Phase 1: AR decode (no grad)
        tokens, gen_mask, bounds = self.decoder.decode(z_seq, z_weights)

        # Phase 2: decoder rerun with gradients → actual hidden states
        hidden = rerun_decoder_multiphrase_with_grad(
            self.gen, z_seq, z_weights, tokens, gen_mask, bounds,
        )
        trimmed_mask = gen_mask[:, :hidden.size(1)]

        # Straight-through token representations
        token_repr = embed_tokens_straight_through(
            hidden, self.gen.output_head, self.gen.token_embedding,
        )

        # Predict hidden states from surface tokens
        predicted_hidden = self.predictor(token_repr, trimmed_mask)

        # Hidden state prediction loss (MSE against actual)
        mask_f = trimmed_mask.unsqueeze(-1).float()
        hs_loss = (
            ((predicted_hidden - hidden.detach()) ** 2 * mask_f).sum()
            / mask_f.sum().clamp(min=1)
            / hidden.size(-1)
        )

        # Reconstruct embedding from predicted hidden states
        surface_recon = self.inverse(predicted_hidden, trimmed_mask)
        surface_loss, surface_acc = self._contrastive_loss(
            surface_recon, embeddings,
        )

        # Combined loss
        loss = (
            direct_loss
            + surface_loss
            + cfg.hidden_state_loss_weight * hs_loss
        )

        # Z diversity regularization
        if self.z_diversity is not None:
            div_loss, _ = self.z_diversity(z_seq, z_weights)
            loss = loss + cfg.z_diversity_weight * div_loss

        with torch.no_grad():
            total_tokens = trimmed_mask.float().sum(dim=1).mean()
            cos_sim = F.cosine_similarity(
                embeddings, surface_recon.detach(), dim=-1,
            ).mean()
            z_cos_sim = F.cosine_similarity(
                embeddings, z_recon.detach(), dim=-1,
            ).mean()

        return {
            "loss": loss,
            "cosine_sim": cos_sim,
            "z_cos_sim": z_cos_sim,
            "direct_acc": direct_acc.detach(),
            "surface_acc": surface_acc.detach(),
            "hs_loss": hs_loss.detach(),
            "num_phrases": num_phrases.mean().detach(),
            "total_tokens": total_tokens,
        }

    def checkpoint_state(self) -> dict:
        return {
            "z_gen": self.z_gen.state_dict(),
            "direct_head": self.direct_head.state_dict(),
            "predictor": self.predictor.state_dict(),
            "inverse": self.inverse.state_dict(),
            "config": self.config.model_dump(),
        }

    def load_checkpoint_state(self, ckpt: dict) -> None:
        self.z_gen.load_state_dict(ckpt["z_gen"])
        self.direct_head.load_state_dict(ckpt["direct_head"])
        self.predictor.load_state_dict(ckpt["predictor"])
        self.inverse.load_state_dict(ckpt["inverse"])

"""Configuration for the tree-structured diffusion decoder."""

from __future__ import annotations

from pydantic import Field

from lfm.config import LFMBaseConfig
from lfm.generator.dep_tree_vae.config import DepTreeVAEConfig


class DiffusionDecoderConfig(LFMBaseConfig):
    """Diffusion decoder hyperparameters."""

    d_model: int = 512
    num_layers: int = 6
    num_heads: int = 8
    dropout: float = 0.1
    num_diffusion_steps: int = 8
    max_tokens_per_role: int = 6
    depth_scale: float = 1.0
    min_noise: float = 0.3


class DepTreeDiffusionConfig(DepTreeVAEConfig):
    """Extends DepTreeVAEConfig with diffusion decoder parameters.

    The encoder, latent space, and skeleton decoder configs are inherited.
    The autoregressive PhraseDecoder is replaced by a tree-structured
    diffusion decoder that generates all content tokens simultaneously.
    """

    diffusion: DiffusionDecoderConfig = Field(
        default_factory=DiffusionDecoderConfig,
    )

    # Completeness scorer — frozen discriminator for structural coherence
    completeness_scorer_path: str = ""
    completeness_weight: float = 0.1

    output_dir: str = "data/models/dep_tree_diffusion_v1"

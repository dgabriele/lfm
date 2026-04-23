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
    max_word_position: int = 8
    num_memory_tokens: int = 4


class DepTreeDiffusionConfig(DepTreeVAEConfig):
    """Extends DepTreeVAEConfig with diffusion decoder parameters.

    The encoder, latent space, and skeleton decoder configs are inherited.
    The autoregressive PhraseDecoder is replaced by a tree-structured
    diffusion decoder that generates all content tokens simultaneously.
    """

    diffusion: DiffusionDecoderConfig = Field(
        default_factory=DiffusionDecoderConfig,
    )

    # Topology — z distances preserve decoded-output distances
    topo_weight: float = 1.0

    # Interpolation smoothness — midpoint decoded output should be
    # between the endpoints' decoded outputs, not in a random direction
    interp_weight: float = 0.5

    # Entropy floor — prevents vocabulary collapse at tail positions
    # where the model cycles through a small pool of affixes.
    # Only penalizes positions with entropy below the threshold.
    # Normal function words (and, of, the) have moderate entropy;
    # degenerate suffix cycling has very low entropy.
    entropy_floor: float = 1.5  # nats (~4-5 plausible tokens minimum)
    entropy_weight: float = 0.05

    # Completeness scorer — frozen discriminator for structural coherence
    completeness_scorer_path: str = ""
    completeness_weight: float = 0.1

    output_dir: str = "data/models/dep_tree_diffusion_v1"

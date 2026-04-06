"""Configuration for reconstruction-based expression training."""

from __future__ import annotations

from lfm.config.base import LFMBaseConfig
from lfm.faculty.config import FacultyConfig
from lfm.generator.config import GeneratorConfig


class ReconstructionConfig(LFMBaseConfig):
    """Configuration for reconstruction training.

    The z-generator learns to produce z-vectors whose decoded IPA
    tokens carry enough information for the inverse decoder to
    reconstruct the original input embedding.
    """

    # Faculty (frozen decoder)
    embedding_dim: int = 384
    decoder_path: str = "data/vae_decoder.pt"
    spm_path: str = "data/spm.model"
    num_memory_tokens: int = 8
    max_output_len: int = 109

    # Z-generator
    z_hidden_dim: int = 512
    max_phrases: int = 3
    diffusion_steps: int = 4
    diffusion_layers: int = 4
    diffusion_heads: int = 8

    # Inverse decoder
    inverse_num_layers: int = 4
    inverse_num_heads: int = 8

    # Regularization
    z_diversity_weight: float = 0.5

    # Data
    embedding_store_dir: str = "data/embeddings"

    # Training
    batch_size: int = 128
    steps: int = 5000
    z_gen_lr: float = 1e-4
    inverse_lr: float = 3e-4
    max_grad_norm: float = 1.0
    warmup_steps: int = 500

    # Output
    checkpoint_every: int = 200
    log_every: int = 20
    output_dir: str = "data/reconstruction"
    device: str = "cuda"
    seed: int = 42

    def build_faculty_config(self) -> FacultyConfig:
        return FacultyConfig(
            dim=self.embedding_dim,
            generator=GeneratorConfig(
                pretrained_decoder_path=self.decoder_path,
                spm_model_path=self.spm_path,
                freeze_decoder=True,
                max_output_len=self.max_output_len,
                num_statements=1,
                num_memory_tokens=self.num_memory_tokens,
            ),
        )

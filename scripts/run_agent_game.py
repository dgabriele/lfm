"""Run the LFM embedding agent game with a frozen generator decoder.

This script:
1. Loads (or generates) an EmbeddingStore
2. Configures a LanguageFaculty with the pretrained VAE generator
3. Runs the reconstruction game — can the original embedding be
   recovered from the linguistic bottleneck?

Usage::

    # Generate synthetic embeddings first (if no real ones):
    python scripts/generate_synthetic_embeddings.py

    # Run the game (after VAE pretraining produces data/vae_decoder.pt):
    python scripts/run_agent_game.py
"""

from __future__ import annotations

import logging

from lfm.embeddings import EmbeddingGameConfig, run_embedding_reconstruction
from lfm.embeddings.config import EmbeddingStoreConfig, SamplerConfig
from lfm.faculty.config import FacultyConfig
from lfm.generator.config import GeneratorConfig

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(message)s",
)


def main(
    store_dir: str = "data/embeddings",
    decoder_path: str = "data/vae_decoder.pt",
    spm_path: str = "data/spm.model",
    steps: int = 10000,
    lr: float = 1e-3,
    batch_size: int = 64,
    embedding_dim: int = 1024,
    device: str = "cuda",
) -> dict[str, float]:
    """Run the embedding reconstruction game.

    Args:
        store_dir: Path to precomputed embedding store.
        decoder_path: Path to pretrained VAE decoder checkpoint.
        spm_path: Path to sentencepiece model.
        steps: Training steps.
        lr: Learning rate (for input projection + game decoder only).
        batch_size: Batch size.
        embedding_dim: Dimensionality of the precomputed embeddings.
        device: GPU or CPU.

    Returns:
        Final metrics dict.
    """
    # Faculty with frozen generator — only input_proj trains
    faculty_config = FacultyConfig(
        dim=embedding_dim,
        generator=GeneratorConfig(
            pretrained_decoder_path=decoder_path,
            spm_model_path=spm_path,
            freeze_decoder=True,
        ),
        # No modular stages — generator handles linguistic structure
        quantizer=None,
        phonology=None,
        morphology=None,
        syntax=None,
        sentence=None,
        channel=None,
    )

    game_config = EmbeddingGameConfig(
        store=EmbeddingStoreConfig(store_dir=store_dir),
        sampler=SamplerConfig(
            batch_size=batch_size,
            num_negatives=7,
        ),
        embedding_dim=embedding_dim,
    )

    results = run_embedding_reconstruction(
        store_dir=store_dir,
        faculty_config=faculty_config,
        game_config=game_config,
        steps=steps,
        lr=lr,
        device=device,
    )

    return results


if __name__ == "__main__":
    main()

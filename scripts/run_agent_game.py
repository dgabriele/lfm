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

from lfm.embeddings import EmbeddingGameConfig
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
    lr: float = 1e-2,
    batch_size: int = 16,
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
            max_output_len=8,  # short sequence for gradient flow
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

    # Direct training loop with cosine-only loss for clearer signal
    import torch

    from lfm.embeddings.games import EmbeddingReconstructionGame
    from lfm.embeddings.store import EmbeddingStore
    from lfm.faculty.model import LanguageFaculty

    torch_device = torch.device(device)

    store = EmbeddingStore(store_dir)
    store.load()
    print(f"Store: {store.num_passages} passages, dim={store.embedding_dim}")

    faculty = LanguageFaculty(faculty_config).to(torch_device)
    game = EmbeddingReconstructionGame(game_config).to(torch_device)

    # Only train unfrozen params
    trainable = [p for p in faculty.parameters() if p.requires_grad]
    trainable += list(game.parameters())
    print(f"Trainable params: {sum(p.numel() for p in trainable):,}")

    optimizer = torch.optim.AdamW(trainable, lr=lr)

    # Simple batch sampling from the store
    embeddings_np = store._embeddings
    n = embeddings_np.shape[0]

    for step in range(steps):
        # Sample batch
        idx = torch.randint(0, n, (batch_size,))
        agent_state = torch.tensor(
            embeddings_np[idx], dtype=torch.float32
        ).to(torch_device)

        # Forward through faculty + game
        lfm_outputs = faculty(agent_state)
        reconstructed = game.decode_message(lfm_outputs)

        # Cosine similarity loss (pure — no MSE noise)
        cosine_sim = torch.nn.functional.cosine_similarity(
            reconstructed, agent_state, dim=-1
        )
        loss = (1.0 - cosine_sim).mean()

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable, 1.0)
        optimizer.step()

        if step % 100 == 0:
            print(
                f"step={step}  loss={loss.item():.4f}  "
                f"cosine_sim={cosine_sim.mean().item():.4f}"
            )

    results = {
        "final_cosine_sim": cosine_sim.mean().item(),
        "final_loss": loss.item(),
    }
    print(f"Final: {results}")
    return results


if __name__ == "__main__":
    main()

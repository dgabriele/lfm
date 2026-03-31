"""Binned similarity pipeline for agent game checkpoints.

Encodes embedding store samples through a trained agent checkpoint
and generates similarity dashboards.
"""

from __future__ import annotations

import logging

import numpy as np
import torch

from lfm.visualize.binned_similarity import compute_and_render_all

logger = logging.getLogger(__name__)


def encode_embeddings(
    faculty,
    embeddings: np.ndarray,
    device: torch.device,
    batch_size: int = 128,
) -> tuple[np.ndarray, np.ndarray]:
    """Encode embeddings through the faculty → tokens + hidden centroids."""
    n = embeddings.shape[0]
    all_tokens = []
    all_centroids = []

    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        batch = torch.tensor(
            embeddings[start:end], dtype=torch.float32,
        ).to(device)

        with torch.no_grad():
            out = faculty(batch)

        tok = out["generator.tokens"].cpu().numpy()
        mask = out["generator.mask"].cpu().float()
        states = out["generator.embeddings"]

        centroid = (states * mask.unsqueeze(-1).to(device)).sum(1)
        centroid = centroid / mask.sum(1, keepdim=True).to(device).clamp(min=1)
        all_centroids.append(centroid.cpu().numpy())
        all_tokens.append(tok)

        if start % (batch_size * 10) == 0 and start > 0:
            logger.info("  encoded %d / %d", start, n)

    return np.concatenate(all_tokens), np.concatenate(all_centroids)


def generate(
    checkpoint_path: str,
    embedding_store_dir: str = "data/embeddings",
    decoder_path: str = "data/vae_decoder.pt",
    spm_path: str = "data/spm.model",
    num_memory_tokens: int = 8,
    output_dir: str = "output/viz",
    n_bins: int = 10000,
    metrics: list[str] | None = None,
    batch_size: int = 128,
    device: str = "cuda",
) -> list:
    """Generate binned similarity dashboards from agent game checkpoint."""
    if metrics is None:
        metrics = ["jaccard", "cosine", "edit"]

    torch_device = torch.device(device)

    from lfm.embeddings.store import EmbeddingStore
    from lfm.faculty.config import FacultyConfig
    from lfm.faculty.model import LanguageFaculty
    from lfm.generator.config import GeneratorConfig

    faculty = LanguageFaculty(FacultyConfig(
        dim=384,
        generator=GeneratorConfig(
            pretrained_decoder_path=decoder_path,
            spm_model_path=spm_path,
            freeze_decoder=True,
            num_memory_tokens=num_memory_tokens,
        ),
    )).to(torch_device)

    gen = faculty.generator
    gen.eval()

    with torch.no_grad():
        faculty(torch.randn(1, 384, device=torch_device))

    ckpt = torch.load(checkpoint_path, map_location=torch_device, weights_only=False)
    if "input_proj" in ckpt:
        gen._input_proj.load_state_dict(ckpt["input_proj"])
        gen._input_refine.load_state_dict(ckpt["input_refine"])
    logger.info(
        "Loaded checkpoint (step=%s, acc=%s)",
        ckpt.get("step"), ckpt.get("accuracy"),
    )

    store = EmbeddingStore(embedding_store_dir)
    store.load()
    logger.info("Encoding %d embeddings...", store.num_passages)

    tokens, centroids = encode_embeddings(
        faculty, store._embeddings, torch_device, batch_size,
    )
    logger.info("Encoded: tokens %s, centroids %s", tokens.shape, centroids.shape)

    return compute_and_render_all(tokens, centroids, metrics, n_bins, output_dir)

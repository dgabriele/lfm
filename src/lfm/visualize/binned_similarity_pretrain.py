"""Binned similarity pipeline for pretrained decoder z distribution.

Encodes pretrain corpus z vectors through the frozen decoder and
generates similarity dashboards showing latent manifold structure.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import torch

from lfm.visualize.binned_similarity import compute_and_render_all

logger = logging.getLogger(__name__)


def decode_z_to_tokens(
    gen,
    z_vectors: np.ndarray,
    device: torch.device,
    batch_size: int = 256,
) -> tuple[np.ndarray, np.ndarray]:
    """Decode pretrained z vectors through the frozen decoder.

    Returns:
        tokens: ``(N, max_len)`` int32 token IDs.
        centroids: ``(N, hidden_dim)`` mean-pooled decoder hidden states.
    """
    n = z_vectors.shape[0]
    all_tokens = []
    all_centroids = []

    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        z = torch.tensor(z_vectors[start:end], dtype=torch.float32).to(device)

        with torch.no_grad():
            tok_ids, _, dec_states, lengths, mask = gen._decode(z)

        all_tokens.append(tok_ids.cpu().numpy())
        mask_f = mask.float().unsqueeze(-1).to(device)
        centroid = (dec_states * mask_f).sum(1) / mask_f.sum(1).clamp(min=1)
        all_centroids.append(centroid.cpu().numpy())

        if start % (batch_size * 20) == 0 and start > 0:
            logger.info("  decoded %d / %d", start, n)

    return np.concatenate(all_tokens), np.concatenate(all_centroids)


def generate(
    resume_path: str,
    cache_path: str = "data/models/v4-phase1/preprocessed_cache.pt",
    spm_path: str = "data/models/v4-phase1/spm.model",
    output_dir: str = "output/viz",
    n_samples: int = 200000,
    n_bins: int = 10000,
    metrics: list[str] | None = None,
    batch_size: int = 256,
    device: str = "cuda",
) -> list[Path]:
    """Generate binned similarity from pretrained z → decoder → IPA.

    Loads z vectors from the pretrain cache (via the encoder), decodes
    through the frozen decoder, and computes similarity dashboards.
    """
    if metrics is None:
        metrics = ["jaccard", "cosine"]

    torch_device = torch.device(device)

    # Use the existing viz loader to get the full model + encode z
    from lfm.visualize.config import VisualizeConfig
    from lfm.visualize.loader import _encode_token_ids, load_checkpoint

    viz_config = VisualizeConfig(
        checkpoint=resume_path, spm_model=spm_path, device=device,
        batch_size=2048,
    )
    model_data = load_checkpoint(viz_config)

    logger.info("Loading preprocessed cache: %s", cache_path)
    cache = torch.load(cache_path, map_location="cpu", weights_only=False)
    token_ids_list = cache["token_ids_list"]
    vocab_size = cache["vocab_size"]

    # Subsample
    total = len(token_ids_list)
    if total > n_samples:
        rng = np.random.default_rng(42)
        indices = rng.choice(total, size=n_samples, replace=False)
        indices.sort()
        token_ids_list = [token_ids_list[i] for i in indices]
        logger.info("Subsampled %d → %d", total, n_samples)

    logger.info("Encoding %d sequences through encoder...", len(token_ids_list))
    z_all = _encode_token_ids(token_ids_list, vocab_size, model_data, viz_config)
    z_np = z_all.cpu().numpy() if isinstance(z_all, torch.Tensor) else z_all
    logger.info("Encoded z: shape=%s", z_np.shape)

    # Build generator for decoding (reuse the loaded model data)
    from lfm.faculty.config import FacultyConfig
    from lfm.faculty.model import LanguageFaculty
    from lfm.generator.config import GeneratorConfig

    # Read config from checkpoint
    ckpt = torch.load(resume_path, map_location="cpu", weights_only=False)
    cfg = ckpt["config"]
    n_mem = cfg.get("num_memory_tokens", 1)

    faculty = LanguageFaculty(FacultyConfig(
        dim=cfg["latent_dim"],
        generator=GeneratorConfig(
            pretrained_decoder_path=cfg.get("output_path", ""),
            spm_model_path=spm_path,
            freeze_decoder=True,
            latent_dim=cfg["latent_dim"],
            decoder_hidden_dim=cfg["decoder_hidden_dim"],
            decoder_num_layers=cfg["decoder_num_layers"],
            decoder_num_heads=cfg["decoder_num_heads"],
            num_memory_tokens=n_mem,
            vocab_size=cfg["spm_vocab_size"],
            use_rope=cfg.get("use_rope", True),
            share_decoder_layers=cfg.get("share_decoder_layers", True),
            attention_head_windows=tuple(cfg.get("attention_head_windows", (3, 3, 7, 7, 15, 15, 0, 0))),
            attention_global_every=cfg.get("attention_global_every", 7),
        ),
    )).to(torch_device)
    gen = faculty.generator
    gen.eval()

    logger.info("Decoding %d z vectors...", len(z_np))
    tokens, centroids = decode_z_to_tokens(gen, z_np, torch_device, batch_size)
    logger.info("Decoded: tokens %s, centroids %s", tokens.shape, centroids.shape)

    return compute_and_render_all(
        tokens, centroids, metrics, n_bins, output_dir, prefix="pretrain_",
    )

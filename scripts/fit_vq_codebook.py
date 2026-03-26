#!/usr/bin/env python3
"""Fit a GroupedVQ codebook to v1's encoder z distribution.

Encodes the training corpus through the pretrained VAE encoder,
then fits a GroupedVQ codebook to the resulting z vectors. The
codebook provides discrete navigation anchors over the continuous
latent space — used at agent time for calibrated code selection
with continuous residual expressivity.

No decoder retraining needed. The v1 decoder was trained on the
continuous manifold; the codebook just tiles it with reference points.

Usage::

    poetry run python scripts/fit_vq_codebook.py
    poetry run python scripts/fit_vq_codebook.py \
        --checkpoint data/models/v1/vae_resume.pt \
        --cache data/models/v1/preprocessed_cache.pt \
        --output data/models/v1/vq_codebook.pt
"""

from __future__ import annotations

import argparse
import logging
import sys

import torch

# Force line-buffered output
sys.stderr = open(sys.stderr.fileno(), "w", buffering=1, closefd=False)
sys.stdout = open(sys.stdout.fileno(), "w", buffering=1, closefd=False)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
logger = logging.getLogger(__name__)


def main(
    checkpoint: str = "data/vae_resume.pt",
    spm_model: str = "data/spm.model",
    cache_path: str = "data/models/v1/preprocessed_cache.pt",
    output: str = "data/models/v1/vq_codebook.pt",
    num_groups: int = 32,
    codebook_size: int = 256,
    num_epochs: int = 20,
    batch_size: int = 4096,
    device: str = "cuda",
) -> None:
    """Fit VQ codebook to encoder z distribution."""
    import numpy as np
    from torch.utils.data import DataLoader, TensorDataset

    from lfm.generator.quantize import GroupedVQ
    from lfm.visualize.config import VisualizeConfig
    from lfm.visualize.loader import _encode_token_ids, load_checkpoint

    # 1. Load model and encode corpus
    logger.info("Loading checkpoint: %s", checkpoint)
    config = VisualizeConfig(
        checkpoint=checkpoint, spm_model=spm_model, device=device,
        batch_size=2048,
    )
    model_data = load_checkpoint(config)

    logger.info("Loading preprocessed cache: %s", cache_path)
    cache = torch.load(cache_path, map_location="cpu", weights_only=False)
    token_ids_list = cache["token_ids_list"]
    vocab_size = cache["vocab_size"]

    logger.info("Encoding %d sequences through v1 encoder...", len(token_ids_list))
    z_all = _encode_token_ids(token_ids_list, vocab_size, model_data, config)
    logger.info("Encoded: z shape=%s, mean_std=%.4f", z_all.shape, z_all.std(dim=0).mean())

    # 2. Fit GroupedVQ codebook
    latent_dim = z_all.shape[1]
    assert latent_dim % num_groups == 0, (
        f"latent_dim {latent_dim} not divisible by num_groups {num_groups}"
    )

    logger.info(
        "Fitting GroupedVQ: %d groups × %d codes × %d dims",
        num_groups, codebook_size, latent_dim // num_groups,
    )

    vq = GroupedVQ(
        num_groups=num_groups,
        codebook_size=codebook_size,
        embedding_dim=latent_dim,
        commitment_weight=0.25,
        entropy_weight=0.1,
        balance_weight=0.1,
        ema_update=True,
        ema_decay=0.99,
    ).to(device)
    vq.train()

    # DataLoader over z vectors
    dataset = TensorDataset(z_all)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    for epoch in range(num_epochs):
        total_loss = 0.0
        total_qe = 0.0
        count = 0

        for (z_batch,) in loader:
            z_batch = z_batch.to(device)
            quantized, loss, indices = vq(z_batch)
            total_loss += loss.item() * z_batch.size(0)
            total_qe += sum(vq.quant_errors) / len(vq.quant_errors) * z_batch.size(0)
            count += z_batch.size(0)

        avg_loss = total_loss / count
        avg_qe = total_qe / count
        util = vq.utilization
        mean_util = sum(util) / len(util)

        logger.info(
            "  Epoch %d/%d: loss=%.4f qe=%.6f util=%.0f%%",
            epoch + 1, num_epochs, avg_loss, avg_qe, mean_util * 100,
        )

        resets = vq.reset_dead_codes()
        if any(r > 0 for r in resets):
            logger.info("    Reset %d dead codes", sum(resets))
        vq.reset_usage()

    # 3. Save codebook
    logger.info("Saving codebook to %s", output)
    torch.save({
        "vq_state_dict": vq.state_dict(),
        "num_groups": num_groups,
        "codebook_size": codebook_size,
        "embedding_dim": latent_dim,
        "z_mean": z_all.mean(dim=0),
        "z_std": z_all.std(dim=0),
        "num_samples": len(z_all),
        "final_qe": avg_qe,
        "final_util": mean_util,
    }, output)

    logger.info(
        "Done. %d groups × %d codes, qe=%.6f, util=%.0f%%",
        num_groups, codebook_size, avg_qe, mean_util * 100,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fit VQ codebook to v1 encoder z")
    parser.add_argument("--checkpoint", default="data/vae_resume.pt")
    parser.add_argument("--spm-model", default="data/spm.model")
    parser.add_argument("--cache", default="data/models/v1/preprocessed_cache.pt")
    parser.add_argument("--output", default="data/models/v1/vq_codebook.pt")
    parser.add_argument("--num-groups", type=int, default=32)
    parser.add_argument("--codebook-size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    main(
        args.checkpoint, args.spm_model, args.cache, args.output,
        args.num_groups, args.codebook_size, args.epochs, args.batch_size,
        args.device,
    )

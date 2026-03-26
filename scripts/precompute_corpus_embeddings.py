#!/usr/bin/env python3
"""Precompute sentence-transformer embeddings for the training corpus.

Reads raw text from a pre-generated HDF5 dataset (created via
``lfm dataset generate``) and encodes with a sentence-transformer.
Embeddings are saved as a numpy array aligned by sample index.

The embeddings are used by contrastive pretraining (InfoNCE loss) to
give the VAE latent space semantic structure.

Usage::

    # First, generate the dataset
    lfm dataset generate --source leipzig --no-llm-gate

    # Then compute embeddings
    poetry run python scripts/precompute_corpus_embeddings.py
    poetry run python scripts/precompute_corpus_embeddings.py \
        --dataset data/datasets/leipzig \
        --output data/datasets/leipzig/embeddings.npy
"""

from __future__ import annotations

import argparse
import logging

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
logger = logging.getLogger(__name__)


def main(
    dataset_dir: str = "data/datasets/leipzig",
    output_path: str = "",
    model_name: str = "all-MiniLM-L6-v2",
    batch_size: int = 512,
    device: str = "cuda",
) -> None:
    from pathlib import Path

    from sentence_transformers import SentenceTransformer

    from lfm.data.dataset.reader import DatasetReader

    # Load dataset
    reader = DatasetReader(dataset_dir)
    logger.info("Dataset: %s (%d samples)", dataset_dir, len(reader))

    # Read raw text from HDF5
    logger.info("Reading raw text from samples.h5...")
    texts: list[str] = []
    for sample in reader.iter_samples():
        texts.append(sample["raw"])

    logger.info("Loaded %d raw text samples", len(texts))

    # Encode with sentence-transformer
    logger.info("Encoding with %s on %s...", model_name, device)
    model = SentenceTransformer(model_name, device=device)
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )

    # Save as float16 (600K x 384 x 2 bytes = ~440 MB)
    embeddings = embeddings.astype(np.float16)
    if not output_path:
        output_path = str(Path(dataset_dir) / "embeddings.npy")
    np.save(output_path, embeddings)
    logger.info(
        "Saved: shape=%s, dtype=%s, %.1f MB → %s",
        embeddings.shape, embeddings.dtype,
        embeddings.nbytes / 1e6, output_path,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Precompute corpus embeddings")
    parser.add_argument("--dataset", default="data/datasets/leipzig")
    parser.add_argument("--output", default="")
    parser.add_argument("--model", default="all-MiniLM-L6-v2")
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    main(args.dataset, args.output, args.model, args.batch_size, args.device)

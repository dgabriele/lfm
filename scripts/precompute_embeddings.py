"""Precompute real LLM embeddings from Leipzig text data.

Encodes Leipzig sentence data with a small sentence-transformer model
(all-MiniLM-L6-v2, 384-dim, 23M params) and saves as an EmbeddingStore
for use in the referential game.

Usage::

    python scripts/precompute_embeddings.py
"""

from __future__ import annotations

import logging

import numpy as np
from sklearn.cluster import MiniBatchKMeans

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(message)s",
)
logger = logging.getLogger(__name__)


def main(
    leipzig_dir: str = "data/leipzig",
    store_dir: str = "data/embeddings",
    model_name: str = "all-MiniLM-L6-v2",
    max_sentences: int = 10000,
    num_clusters: int = 64,
    batch_size: int = 256,
    device: str = "cuda",
) -> None:
    """Encode Leipzig sentences and save as EmbeddingStore.

    Args:
        leipzig_dir: Path to Leipzig data directory.
        store_dir: Where to save the embedding store.
        model_name: HuggingFace sentence-transformer model ID.
        max_sentences: Maximum sentences to encode.
        num_clusters: Number of clusters for stratified sampling.
        batch_size: Encoding batch size.
        device: Device for the encoder.
    """
    from sentence_transformers import SentenceTransformer

    from lfm.data.loaders.leipzig import LeipzigCorpusConfig, LeipzigCorpusLoader
    from lfm.embeddings.store import EmbeddingStore

    # 1. Load sentences from Leipzig (English only for semantic coherence)
    logger.info("Loading Leipzig sentences...")
    loader = LeipzigCorpusLoader(
        LeipzigCorpusConfig(
            data_dir=leipzig_dir,
            languages=["eng"],
            max_samples_per_language=max_sentences,
            min_line_length=30,
        )
    )
    samples = loader.load()
    texts = [text for _, text in samples]
    logger.info("Loaded %d sentences", len(texts))

    if not texts:
        # Fall back to all languages if no English
        logger.info("No English found, using all languages...")
        loader = LeipzigCorpusLoader(
            LeipzigCorpusConfig(
                data_dir=leipzig_dir,
                max_samples_per_language=max_sentences // 16,
                min_line_length=30,
            )
        )
        samples = loader.load()
        texts = [text for _, text in samples]
        logger.info("Loaded %d sentences from all languages", len(texts))

    # 2. Encode with sentence-transformer
    logger.info("Loading encoder: %s", model_name)
    encoder = SentenceTransformer(model_name, device=device)
    embedding_dim = encoder.get_sentence_embedding_dimension()
    logger.info("Encoding %d sentences (dim=%d)...", len(texts), embedding_dim)

    embeddings = encoder.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    embeddings = embeddings.astype(np.float32)
    logger.info("Encoded: shape=%s", embeddings.shape)

    # Free encoder VRAM
    del encoder
    import torch

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # 3. Cluster
    logger.info("Clustering into %d clusters...", num_clusters)
    kmeans = MiniBatchKMeans(
        n_clusters=num_clusters,
        random_state=42,
        batch_size=1024,
    )
    cluster_labels = kmeans.fit_predict(embeddings).astype(np.int32)

    # 4. Save
    metadata = {
        "num_passages": len(texts),
        "embedding_dim": int(embedding_dim),
        "num_clusters": num_clusters,
        "source": "leipzig_english",
        "encoder": model_name,
    }

    store = EmbeddingStore.create(
        store_dir=store_dir,
        embeddings=embeddings,
        cluster_labels=cluster_labels,
        metadata=metadata,
    )
    logger.info(
        "Saved: %d embeddings, dim=%d, %d clusters → %s",
        store.num_passages,
        store.embedding_dim,
        store.num_clusters,
        store_dir,
    )


if __name__ == "__main__":
    main()

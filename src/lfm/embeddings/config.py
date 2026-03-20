"""Configuration models for the LFM embeddings pipeline.

Defines all Pydantic configs used by the offline precomputation pipeline
(text encoding, chunking, clustering, storage) and the training-time
embedding game (sampling, loss weighting).
"""

from __future__ import annotations

from lfm.config.base import LFMBaseConfig


class TextEncoderConfig(LFMBaseConfig):
    """Configuration for a frozen text encoder used to produce passage embeddings.

    Attributes:
        name: Registry key identifying which encoder implementation to use.
        model_id: HuggingFace model identifier or local path.
        embedding_dim: Dimensionality of the output embedding vectors.
        max_tokens: Maximum number of tokens per input passage.
        batch_size: Micro-batch size for encoding.
        dtype: Floating-point precision for model weights (``"float16"`` or
            ``"float32"``).
        device: Device to load the model onto (``"cuda"`` or ``"cpu"``).
        prompt_name: Optional prompt template name for models that use
            instruction-prefixed encoding (e.g. Instructor, Qwen3).
        trust_remote_code: Whether to trust remote code when loading the model
            from HuggingFace Hub.
    """

    name: str = "sentence_transformers"
    model_id: str = "dunzhang/stella_en_400M_v5"
    embedding_dim: int = 1024
    max_tokens: int = 512
    batch_size: int = 128
    dtype: str = "float16"
    device: str = "cuda"
    prompt_name: str | None = None
    trust_remote_code: bool = False


class ChunkerConfig(LFMBaseConfig):
    """Configuration for the sliding-window text chunker.

    Attributes:
        min_tokens: Minimum window size in tokens. Windows shorter than this
            are discarded.
        max_tokens: Maximum window size in tokens (the nominal chunk length).
        overlap_tokens: Number of tokens shared between consecutive windows.
        tokenizer_name: Optional explicit tokenizer name. When ``None``, the
            chunker uses the encoder's own tokenizer.
    """

    min_tokens: int = 80
    max_tokens: int = 200
    overlap_tokens: int = 20
    tokenizer_name: str | None = None


class ClusterConfig(LFMBaseConfig):
    """Configuration for post-encoding embedding clustering.

    Attributes:
        method: Clustering algorithm -- ``"kmeans"`` for MiniBatchKMeans or
            ``"hdbscan"`` for density-based clustering.
        num_clusters: Number of clusters for k-means. Ignored by HDBSCAN.
        min_cluster_size: Minimum cluster size parameter for HDBSCAN. Ignored
            by k-means.
        random_state: Random seed for reproducibility.
    """

    method: str = "kmeans"
    num_clusters: int = 256
    min_cluster_size: int = 50
    random_state: int = 42


class EmbeddingStoreConfig(LFMBaseConfig):
    """Configuration for the on-disk embedding store.

    Attributes:
        store_dir: Path to the directory where embeddings and metadata are
            persisted.
        embedding_dtype: NumPy dtype string for saved embedding arrays.
        save_text: Whether to also persist the raw passage text alongside
            embeddings.
    """

    store_dir: str = "data/embeddings"
    embedding_dtype: str = "float16"
    save_text: bool = False


class SamplerConfig(LFMBaseConfig):
    """Configuration for the training-time embedding batch sampler.

    Attributes:
        batch_size: Number of anchor passages per training batch.
        num_negatives: Number of negative (distractor) passages per anchor.
        curriculum_start: Initial difficulty level (0.0 = easy) for curriculum
            sampling.
        curriculum_end: Final difficulty level (1.0 = hard).
        curriculum_warmup_steps: Number of steps over which difficulty linearly
            ramps from ``curriculum_start`` to ``curriculum_end``.
        within_cluster_ratio: Fraction of negatives drawn from the same cluster
            as the anchor (hard negatives). The remainder are drawn from
            different clusters.
        prefetch_batches: Number of batches to prefetch in the data loader.
        pin_memory: Whether to pin host memory for faster GPU transfers.
    """

    batch_size: int = 64
    num_negatives: int = 7
    curriculum_start: float = 0.0
    curriculum_end: float = 1.0
    curriculum_warmup_steps: int = 10000
    within_cluster_ratio: float = 0.5
    prefetch_batches: int = 4
    pin_memory: bool = True


class PrecomputePipelineConfig(LFMBaseConfig):
    """Configuration for the full offline precomputation pipeline.

    Orchestrates chunking, encoding, clustering, and storage in a single
    reproducible run.

    Attributes:
        encoder: Configuration for the text encoder.
        chunker: Configuration for the text chunker.
        cluster: Configuration for the clustering step.
        store: Configuration for the on-disk embedding store.
        corpus_paths: List of file or directory paths comprising the corpus.
        corpus_format: Format of the corpus files -- ``"text"`` for plain text
            or ``"jsonl"`` for newline-delimited JSON.
    """

    encoder: TextEncoderConfig = TextEncoderConfig()
    chunker: ChunkerConfig = ChunkerConfig()
    cluster: ClusterConfig = ClusterConfig()
    store: EmbeddingStoreConfig = EmbeddingStoreConfig()
    corpus_paths: list[str] = []
    corpus_format: str = "text"


class EmbeddingGameConfig(LFMBaseConfig):
    """Configuration for the embedding reconstruction/referential game.

    Used at training time to set up the game that drives the language faculty
    to produce messages from which embeddings can be approximately recovered.

    Attributes:
        store: Configuration for locating the precomputed embedding store.
        sampler: Configuration for the batch sampler.
        embedding_dim: Dimensionality of the precomputed embeddings.
        reconstruction_weight: Loss weight for the reconstruction objective.
        referential_weight: Loss weight for the referential (contrastive)
            objective.
        num_distractors: Number of distractor embeddings per referential trial.
        batch_size: Training batch size.
    """

    store: EmbeddingStoreConfig = EmbeddingStoreConfig()
    sampler: SamplerConfig = SamplerConfig()
    embedding_dim: int = 1024
    reconstruction_weight: float = 1.0
    referential_weight: float = 1.0
    num_distractors: int = 7
    batch_size: int = 64

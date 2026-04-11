"""Configuration for Qwen-latent target embedding construction."""

from __future__ import annotations

from typing import Literal

from lfm.config.base import LFMBaseConfig


class CorpusSourceConfig(LFMBaseConfig):
    """One entry in a mixed corpus.

    A source is either a **local file** (``path``) or a **HuggingFace
    streaming dataset** (``hf_dataset``).  Exactly one must be set.

    Attributes:
        path: Local path to a JSONL or plain-text file.
        hf_dataset: HuggingFace dataset identifier
            (e.g. ``cerebras/SlimPajama-627B``).  When set, the source
            is streamed via ``datasets.load_dataset(..., streaming=True)``.
        hf_config: Optional HF dataset config/subset name.
        hf_split: HF split name; defaults to ``"train"``.
        hf_trust_remote_code: Allow HF to execute custom loading scripts
            bundled with the dataset.  Required for some datasets.
        text_field: Record field holding the text content.  For plain-
            text local files, set to ``None`` to treat each non-empty
            line as a sample.
        weight: Sampling weight relative to other sources in a mix.
        max_samples: Optional per-source cap; ``None`` means no cap.
        min_length: Minimum text length (chars) to keep; shorter
            records are skipped.  Only applied to HF sources.
        max_length: Maximum text length (chars); longer records are
            truncated.  Only applied to HF sources.
        name: Human-readable label for logging.
    """

    path: str | None = None
    hf_dataset: str | None = None
    hf_config: str | None = None
    hf_split: str = "train"
    hf_trust_remote_code: bool = False
    text_field: str | None = "text"
    weight: float = 1.0
    max_samples: int | None = None
    min_length: int = 20
    max_length: int = 4000
    name: str = ""


class ExtractorConfig(LFMBaseConfig):
    """Hidden-state extraction settings.

    Attributes:
        model_name: HuggingFace causal LM to load.
        layer: Transformer hidden-state layer index.  ``-1`` = last.
        pooling: How to reduce the sequence dimension.
        dtype: Model dtype for the frozen LLM.
        batch_size: Batch size for the extraction forward pass.
        max_len: Maximum input length in tokens.
    """

    model_name: str = "Qwen/Qwen2.5-0.5B"
    layer: int = -1
    pooling: Literal["last_token", "mean", "bos"] = "last_token"
    dtype: Literal["bfloat16", "float16", "float32"] = "bfloat16"
    batch_size: int = 16
    max_len: int = 256
    compile: bool = False
    attn_implementation: str | None = None


class DensityConfig(LFMBaseConfig):
    """Density-aware resampling settings.

    Uses k-nearest-neighbor distance as a local density proxy: denser
    regions have smaller k-NN distances, sparser regions have larger.
    The resampling ``temperature`` blends between the natural density
    (``temperature=0``) and fully uniform on-manifold coverage
    (``temperature=1``).

    Attributes:
        enabled: Whether to apply density-aware resampling at all.
        knn_k: Which nearest neighbor to measure distance to.  Larger
            values are smoother but more expensive.
        temperature: ``0`` keeps natural density, ``1`` fully corrects
            toward uniform coverage.  Values between are interpolations.
        target_size: Final number of embeddings after resampling.  If
            greater than input size, sampling is with replacement.
    """

    enabled: bool = True
    knn_k: int = 10
    temperature: float = 0.7
    target_size: int = 300_000


class ChunkingConfig(LFMBaseConfig):
    """Sentence-aware document chunking settings.

    When enabled, every input document is split into one or more
    chunks that end at natural sentence (or paragraph) boundaries and
    fit within ``max_tokens`` of the extractor's tokenizer.  This
    guarantees that every pooled hidden state we extract represents a
    *complete* semantic unit, rather than a mid-sentence fragment
    produced by naive token truncation.

    Attributes:
        enabled: Whether to apply chunking at all.  Off → each
            document is passed to the extractor as a single input
            (truncated mid-sequence if necessary — not recommended).
        max_tokens: Maximum tokens per chunk.  Should equal — or sit
            slightly under — the extractor's ``max_len`` to leave
            headroom for tokenizer idiosyncrasies.
        max_chunks_per_doc: Cap on chunks emitted per document.
            Prevents a single very long source (a book, a long
            cosmopedia essay) from dominating the output distribution.
        min_chunk_tokens: Drop chunks shorter than this to avoid
            emitting single-word or single-phrase fragments.
    """

    enabled: bool = False
    max_tokens: int = 240
    max_chunks_per_doc: int = 3
    min_chunk_tokens: int = 30


class ClusterConfig(LFMBaseConfig):
    """K-means clustering settings (for hard-negative sampling downstream).

    Attributes:
        num_clusters: Number of clusters the dialogue game will use
            for hard-negative sampling.
        random_state: Seed for reproducibility.
        batch_size: MiniBatchKMeans batch size.
    """

    num_clusters: int = 2047
    random_state: int = 42
    batch_size: int = 4096


class QwenTargetsConfig(LFMBaseConfig):
    """End-to-end config for building a Qwen-latent target store.

    Attributes:
        sources: One or more :class:`CorpusSourceConfig`.  Texts are
            drawn from each source in proportion to its ``weight``,
            interleaved, and optionally capped.
        output_dir: Where to write the resulting
            :class:`~lfm.embeddings.store.EmbeddingStore`.
        extractor: Hidden-state extraction settings.
        density: Density-aware resampling settings.
        cluster: K-means settings.
        shuffle_seed: Seed for corpus interleaving.
        device: Compute device.
        max_extracted: Optional absolute cap on how many texts to
            process through the LLM before density resampling.
            Useful for small experiments.
    """

    sources: list[CorpusSourceConfig]
    output_dir: str = "data/embeddings_qwen"
    prefetch_dir: str = "data/qwen_targets_cache"
    extractor: ExtractorConfig = ExtractorConfig()
    chunking: ChunkingConfig = ChunkingConfig()
    density: DensityConfig = DensityConfig()
    cluster: ClusterConfig = ClusterConfig()
    shuffle_seed: int = 42
    device: str = "cuda"
    max_extracted: int | None = None

"""LFM embeddings subpackage -- offline pipeline and training-time games.

Provides components for:

**Offline precomputation** (built in parallel):
    Chunking text corpora, encoding passages into dense embeddings via frozen
    LLM encoders, clustering the resulting vectors, and storing everything on
    disk in a memory-mapped format suitable for fast training-time sampling.

**Training-time games** (this package):
    Stratified sampling from precomputed embeddings, asynchronous prefetching,
    reconstruction and referential game modules, loss functions, training
    phases, and evaluation metrics.  These drive the language faculty to
    produce messages from which embedding content can be approximately
    recovered.
"""

from __future__ import annotations

from lfm.embeddings.config import (
    ChunkerConfig,
    ClusterConfig,
    EmbeddingGameConfig,
    EmbeddingStoreConfig,
    PrecomputePipelineConfig,
    SamplerConfig,
    TextEncoderConfig,
)
from lfm.embeddings.encoder import SentenceTransformersEncoder, TextEncoder
from lfm.embeddings.games import EmbeddingReconstructionGame, EmbeddingReferentialGame
from lfm.embeddings.losses import EmbeddingReconstructionLoss, EmbeddingReferentialLoss
from lfm.embeddings.metrics import (
    CurriculumDifficulty,
    EmbeddingReconstructionSimilarity,
    EmbeddingReferentialAccuracy,
)
from lfm.embeddings.phases import (
    EmbeddingReconstructionGamePhase,
    EmbeddingReferentialGamePhase,
)
from lfm.embeddings.prefetcher import AsyncPrefetcher
from lfm.embeddings.sampler import StratifiedSampler

# Offline pipeline components are imported conditionally since they may not
# exist yet (being built in parallel).
try:
    from lfm.embeddings.chunker import TextChunker
except ImportError:
    TextChunker = None  # type: ignore[assignment,misc]

try:
    from lfm.embeddings.pipeline import PrecomputePipeline
except ImportError:
    PrecomputePipeline = None  # type: ignore[assignment,misc]

try:
    from lfm.embeddings.store import EmbeddingStore
except ImportError:
    EmbeddingStore = None  # type: ignore[assignment,misc]

__all__ = [
    # Config
    "ChunkerConfig",
    "ClusterConfig",
    "EmbeddingGameConfig",
    "EmbeddingStoreConfig",
    "PrecomputePipelineConfig",
    "SamplerConfig",
    "TextEncoderConfig",
    # Offline pipeline
    "EmbeddingStore",
    "PrecomputePipeline",
    "SentenceTransformersEncoder",
    "TextChunker",
    "TextEncoder",
    # Training-time: sampling
    "AsyncPrefetcher",
    "StratifiedSampler",
    # Training-time: games
    "EmbeddingReconstructionGame",
    "EmbeddingReferentialGame",
    # Training-time: losses
    "EmbeddingReconstructionLoss",
    "EmbeddingReferentialLoss",
    # Training-time: phases
    "EmbeddingReconstructionGamePhase",
    "EmbeddingReferentialGamePhase",
    # Training-time: metrics
    "CurriculumDifficulty",
    "EmbeddingReconstructionSimilarity",
    "EmbeddingReferentialAccuracy",
    # Convenience runners
    "run_embedding_reconstruction",
    "run_embedding_referential",
]


def run_embedding_reconstruction(
    store_dir: str,
    faculty_config: object | None = None,
    game_config: EmbeddingGameConfig | None = None,
    steps: int = 10000,
    lr: float = 1e-3,
    log_every: int = 100,
    device: str = "cuda",
) -> dict[str, float]:
    """End-to-end embedding reconstruction game training.

    Builds the embedding store, sampler, prefetcher, language faculty,
    and game module, then runs a training loop for the specified number
    of steps.

    Args:
        store_dir: Path to a directory containing precomputed embeddings
            (produced by ``PrecomputePipeline``).
        faculty_config: Optional ``FacultyConfig``. When ``None``, a default
            config is created with ``dim`` matching the embedding dimension.
        game_config: Optional ``EmbeddingGameConfig``. When ``None``, a
            default config is created.
        steps: Number of training steps.
        lr: Learning rate for the AdamW optimizer.
        log_every: Log metrics every *N* steps.
        device: Device string (e.g. ``"cuda"``, ``"cpu"``).

    Returns:
        Dictionary of final metric values.
    """
    import torch

    from lfm.embeddings.config import EmbeddingGameConfig, EmbeddingStoreConfig
    from lfm.embeddings.store import EmbeddingStore
    from lfm.faculty.config import FacultyConfig
    from lfm.faculty.model import LanguageFaculty
    from lfm.utils.logging import get_logger

    logger = get_logger(__name__)
    torch_device = torch.device(device)

    # 1. Load the embedding store
    store = EmbeddingStore(store_dir)
    store.load()
    logger.info(
        "Loaded embedding store: %d passages, dim=%d, %d clusters",
        store.num_passages,
        store.embedding_dim,
        store.num_clusters,
    )

    # 2. Build configs
    if game_config is None:
        game_config = EmbeddingGameConfig(
            store=EmbeddingStoreConfig(store_dir=store_dir),
            embedding_dim=store.embedding_dim,
        )

    if faculty_config is None:
        faculty_config = FacultyConfig(dim=store.embedding_dim)

    # 3. Build models
    faculty = LanguageFaculty(faculty_config)
    faculty.to(torch_device)

    game = EmbeddingReconstructionGame(game_config)
    game.to(torch_device)

    # 4. Build sampler and prefetcher
    sampler = StratifiedSampler(game_config.sampler, store)
    prefetcher = AsyncPrefetcher(
        sampler,
        torch_device,
        prefetch_batches=game_config.sampler.prefetch_batches,
        pin_memory=game_config.sampler.pin_memory,
    )

    # 5. Build optimizer
    params = list(faculty.parameters()) + list(game.parameters())
    optimizer = torch.optim.AdamW(params, lr=lr)

    # 6. Build metrics
    recon_sim = EmbeddingReconstructionSimilarity()
    curriculum_metric = CurriculumDifficulty()

    # 7. Build loss
    from lfm.embeddings.losses import EmbeddingReconstructionLoss

    recon_loss_fn = EmbeddingReconstructionLoss(weight=game_config.reconstruction_weight)

    # 8. Training loop
    logger.info("Starting embedding reconstruction training (%d steps)", steps)
    prefetcher.start()

    loss = torch.tensor(0.0)  # default in case steps == 0
    try:
        for step in range(steps):
            batch = next(prefetcher)
            agent_state = batch["agent_state"]

            # Forward
            lfm_outputs = faculty(agent_state)
            reconstructed = game.decode_message(lfm_outputs)

            outputs = dict(lfm_outputs)
            outputs["game.reconstructed_embedding"] = reconstructed
            outputs["game.original_embedding"] = agent_state

            targets = {"game.original_embedding": agent_state}

            # Loss
            loss = recon_loss_fn(outputs, targets)

            # Extra faculty losses
            for _k, v in faculty.extra_losses().items():
                loss = loss + v

            # Backward
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            optimizer.step()

            # Metrics
            if step % log_every == 0:
                recon_sim.update(outputs)
                curriculum_metric.current_difficulty = sampler.curriculum_difficulty
                curriculum_metric.update(outputs)

                logger.info(
                    "step=%d  loss=%.4f  cosine_sim=%.4f  difficulty=%.3f",
                    step,
                    loss.item(),
                    recon_sim.result(),
                    sampler.curriculum_difficulty,
                )
    finally:
        prefetcher.stop()

    results = {
        "reconstruction_similarity": recon_sim.result(),
        "curriculum_difficulty": sampler.curriculum_difficulty,
        "final_loss": loss.item(),
    }
    logger.info("Training complete. Results: %s", results)
    return results


def run_embedding_referential(
    store_dir: str,
    faculty_config: object | None = None,
    game_config: EmbeddingGameConfig | None = None,
    steps: int = 10000,
    lr: float = 1e-3,
    log_every: int = 100,
    device: str = "cuda",
) -> dict[str, float]:
    """End-to-end embedding referential game training.

    Builds the embedding store, sampler, prefetcher, language faculty,
    and game module, then runs a referential training loop for the
    specified number of steps.

    Args:
        store_dir: Path to a directory containing precomputed embeddings
            (produced by ``PrecomputePipeline``).
        faculty_config: Optional ``FacultyConfig``. When ``None``, a default
            config is created with ``dim`` matching the embedding dimension.
        game_config: Optional ``EmbeddingGameConfig``. When ``None``, a
            default config is created.
        steps: Number of training steps.
        lr: Learning rate for the AdamW optimizer.
        log_every: Log metrics every *N* steps.
        device: Device string (e.g. ``"cuda"``, ``"cpu"``).

    Returns:
        Dictionary of final metric values.
    """
    import torch

    from lfm.embeddings.config import EmbeddingGameConfig, EmbeddingStoreConfig
    from lfm.embeddings.store import EmbeddingStore
    from lfm.faculty.config import FacultyConfig
    from lfm.faculty.model import LanguageFaculty
    from lfm.utils.logging import get_logger

    logger = get_logger(__name__)
    torch_device = torch.device(device)

    # 1. Load the embedding store
    store = EmbeddingStore(store_dir)
    store.load()
    logger.info(
        "Loaded embedding store: %d passages, dim=%d, %d clusters",
        store.num_passages,
        store.embedding_dim,
        store.num_clusters,
    )

    # 2. Build configs
    if game_config is None:
        game_config = EmbeddingGameConfig(
            store=EmbeddingStoreConfig(store_dir=store_dir),
            embedding_dim=store.embedding_dim,
        )

    if faculty_config is None:
        faculty_config = FacultyConfig(dim=store.embedding_dim)

    # 3. Build models
    faculty = LanguageFaculty(faculty_config)
    faculty.to(torch_device)

    game = EmbeddingReferentialGame(game_config)
    game.to(torch_device)

    # 4. Build sampler (referential mode)
    sampler = StratifiedSampler(game_config.sampler, store)

    # 5. Build optimizer
    params = list(faculty.parameters()) + list(game.parameters())
    optimizer = torch.optim.AdamW(params, lr=lr)

    # 6. Build metrics
    ref_accuracy = EmbeddingReferentialAccuracy()
    curriculum_metric = CurriculumDifficulty()

    # 7. Build loss
    from lfm.embeddings.losses import EmbeddingReferentialLoss

    ref_loss_fn = EmbeddingReferentialLoss(weight=game_config.referential_weight)

    # 8. Training loop
    logger.info("Starting embedding referential training (%d steps)", steps)

    loss = torch.tensor(0.0)  # default in case steps == 0
    for step in range(steps):
        # Sample a referential batch directly from the sampler and move
        # tensors to the target device.
        ref_batch = sampler.sample_referential_batch()
        ref_batch = {k: v.to(torch_device) for k, v in ref_batch.items()}
        sampler.step_curriculum()

        agent_state = ref_batch["agent_state"]  # (B, dim)
        distractors = ref_batch["distractors"]  # (B, K, dim)
        bs = agent_state.shape[0]

        # Forward
        lfm_outputs = faculty(agent_state)

        # Assemble and shuffle candidates
        target_unsqueezed = agent_state.unsqueeze(1)  # (B, 1, dim)
        candidates = torch.cat([target_unsqueezed, distractors], dim=1)  # (B, K+1, dim)
        num_candidates = candidates.shape[1]

        perm = torch.stack([torch.randperm(num_candidates, device=torch_device) for _ in range(bs)])
        perm_expanded = perm.unsqueeze(-1).expand_as(candidates)
        candidates = torch.gather(candidates, 1, perm_expanded)
        target_idx = (perm == 0).long().argmax(dim=1)

        # Score candidates
        score_output = game.score_candidates(lfm_outputs, candidates)
        receiver_logits = score_output["receiver_logits"]

        outputs = dict(lfm_outputs)
        outputs["game.receiver_logits"] = receiver_logits
        outputs["game.target_idx"] = target_idx

        targets = {
            "game.receiver_logits": receiver_logits,
            "game.target_idx": target_idx,
        }

        # Loss
        loss = ref_loss_fn(outputs, targets)

        for _k, v in faculty.extra_losses().items():
            loss = loss + v

        # Backward
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        optimizer.step()

        # Metrics
        if step % log_every == 0:
            ref_accuracy.update(outputs)
            curriculum_metric.current_difficulty = sampler.curriculum_difficulty
            curriculum_metric.update(outputs)

            logger.info(
                "step=%d  loss=%.4f  accuracy=%.4f  difficulty=%.3f",
                step,
                loss.item(),
                ref_accuracy.result(),
                sampler.curriculum_difficulty,
            )

    results = {
        "referential_accuracy": ref_accuracy.result(),
        "curriculum_difficulty": sampler.curriculum_difficulty,
        "final_loss": loss.item(),
    }
    logger.info("Training complete. Results: %s", results)
    return results

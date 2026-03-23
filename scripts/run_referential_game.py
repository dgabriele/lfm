"""Run the LFM referential game with a frozen generator decoder.

Tests whether the linguistic bottleneck preserves *distinguishing*
information: can a receiver pick the correct embedding from among
distractors based on the message that went through the bottleneck?

Random chance = 1/(num_distractors+1) = 12.5% with 7 distractors.
Any accuracy above that means the bottleneck carries useful signal.

Usage::

    python scripts/generate_synthetic_embeddings.py  # if needed
    python scripts/run_referential_game.py
"""

from __future__ import annotations

import logging

import torch
import torch.nn.functional as F  # noqa: N812

from lfm.embeddings.config import EmbeddingGameConfig, EmbeddingStoreConfig, SamplerConfig
from lfm.embeddings.games import EmbeddingReferentialGame
from lfm.embeddings.store import EmbeddingStore
from lfm.faculty.config import FacultyConfig
from lfm.faculty.model import LanguageFaculty
from lfm.generator.config import GeneratorConfig

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(message)s",
)
logger = logging.getLogger(__name__)


def main(
    store_dir: str = "data/embeddings",
    decoder_path: str = "data/vae_decoder.pt",
    spm_path: str = "data/spm.model",
    steps: int = 50000,
    lr: float = 1e-3,
    batch_size: int = 64,
    num_distractors: int = 7,
    embedding_dim: int = 1024,
    device: str = "cuda",
    log_every: int = 50,
) -> dict[str, float]:
    """Run the referential game.

    The sender's embedding goes through the frozen generator bottleneck.
    The receiver scores the target + distractors via dot-product with
    the message. Cross-entropy loss on the receiver's choice.

    Random chance accuracy = 1/(num_distractors+1).
    """
    torch_device = torch.device(device)

    # Load embeddings
    store = EmbeddingStore(store_dir)
    store.load()
    logger.info(
        "Store: %d passages, dim=%d, %d clusters",
        store.num_passages,
        store.embedding_dim,
        store.num_clusters,
    )

    # Faculty with frozen generator
    faculty_config = FacultyConfig(
        dim=embedding_dim,
        generator=GeneratorConfig(
            pretrained_decoder_path=decoder_path,
            spm_model_path=spm_path,
            freeze_decoder=True,
        ),
        quantizer=None,
        phonology=None,
    )

    game_config = EmbeddingGameConfig(
        store=EmbeddingStoreConfig(store_dir=store_dir),
        sampler=SamplerConfig(batch_size=batch_size, num_negatives=num_distractors),
        embedding_dim=embedding_dim,
        num_distractors=num_distractors,
    )

    faculty = LanguageFaculty(faculty_config).to(torch_device)
    game = EmbeddingReferentialGame(game_config).to(torch_device)

    # Only train unfrozen params
    trainable = [p for p in faculty.parameters() if p.requires_grad]
    trainable += list(game.parameters())
    logger.info("Trainable params: %d", sum(p.numel() for p in trainable))

    optimizer = torch.optim.AdamW(trainable, lr=lr)

    embeddings_np = store._embeddings
    n = embeddings_np.shape[0]
    num_candidates = num_distractors + 1
    chance = 1.0 / num_candidates

    logger.info(
        "Referential game: %d distractors, chance=%.1f%%",
        num_distractors,
        chance * 100,
    )

    for step in range(steps):
        # Sample anchor + distractors
        idx = torch.randint(0, n, (batch_size,))
        anchor = torch.tensor(
            embeddings_np[idx], dtype=torch.float32
        ).to(torch_device)

        dist_idx = torch.randint(0, n, (batch_size, num_distractors))
        distractors = torch.tensor(
            embeddings_np[dist_idx.numpy()], dtype=torch.float32
        ).to(torch_device)

        # Forward through faculty
        lfm_outputs = faculty(anchor)

        # Assemble candidates: target at position 0, then distractors
        candidates = torch.cat(
            [anchor.unsqueeze(1), distractors], dim=1
        )  # (B, K, dim)

        # Shuffle per batch element
        perm = torch.stack(
            [torch.randperm(num_candidates, device=torch_device) for _ in range(batch_size)]
        )
        perm_expanded = perm.unsqueeze(-1).expand_as(candidates)
        candidates = torch.gather(candidates, 1, perm_expanded)
        target_idx = (perm == 0).long().argmax(dim=1)  # (B,)

        # Receiver scores
        score_output = game.score_candidates(lfm_outputs, candidates)
        logits = score_output["receiver_logits"]  # (B, K)

        # Loss
        loss = F.cross_entropy(logits, target_idx)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable, 1.0)
        optimizer.step()

        if step % log_every == 0:
            with torch.no_grad():
                preds = logits.argmax(dim=1)
                accuracy = (preds == target_idx).float().mean().item()

            logger.info(
                "step=%d  loss=%.4f  accuracy=%.1f%%  (chance=%.1f%%)",
                step,
                loss.item(),
                accuracy * 100,
                chance * 100,
            )

    # Final eval
    with torch.no_grad():
        preds = logits.argmax(dim=1)
        final_acc = (preds == target_idx).float().mean().item()

    results = {
        "final_accuracy": final_acc,
        "final_loss": loss.item(),
        "chance": chance,
    }
    logger.info("Final: %s", results)
    return results


if __name__ == "__main__":
    main()

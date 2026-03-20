"""Training phases for embedding-domain emergent communication games.

These phases consume precomputed embedding batches produced by the
``AsyncPrefetcher`` (which draws from an ``EmbeddingStore``), rather than
generating scenes on the fly like the scene-based game phases.

``EmbeddingReconstructionGamePhase``:
    batch["agent_state"] -> faculty -> game.decode_message -> reconstruction loss.

``EmbeddingReferentialGamePhase``:
    batch["agent_state"] + batch["distractors"] -> faculty -> game.score_candidates
    -> referential loss.

Both phases also compute any structural losses configured in the phase
config and incorporate auxiliary losses from the faculty.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from torch import Tensor

from lfm._registry import register
from lfm.embeddings.games import EmbeddingReconstructionGame, EmbeddingReferentialGame
from lfm.embeddings.losses import EmbeddingReconstructionLoss, EmbeddingReferentialLoss
from lfm.training.phase import TrainingPhase

if TYPE_CHECKING:
    from torch import nn

    from lfm.embeddings.config import EmbeddingGameConfig
    from lfm.training.config import PhaseConfig


def _infer_device(faculty: nn.Module) -> torch.device:
    """Infer the device from the faculty's parameters.

    Falls back to CPU if the faculty has no parameters.

    Args:
        faculty: The ``LanguageFaculty`` model.

    Returns:
        The device of the first parameter, or CPU.
    """
    p = next(faculty.parameters(), None)
    return p.device if p is not None else torch.device("cpu")


@register("phase", "embedding_reconstruction_game")
class EmbeddingReconstructionGamePhase(TrainingPhase):
    """Training phase for the embedding reconstruction game.

    At each step this phase:

    1. Receives ``batch["agent_state"]`` from the prefetcher (precomputed
       embeddings, not self-generated).
    2. Runs the ``LanguageFaculty`` forward pass.
    3. Decodes the LFM output back to an embedding via the game module.
    4. Computes reconstruction loss (cosine + MSE) plus any structural
       losses configured in the phase config.
    5. Returns combined outputs and losses.

    Args:
        config: Phase configuration (loss weights, frozen modules, etc.).
        faculty: The ``LanguageFaculty`` model being trained.
        game: The ``EmbeddingReconstructionGame`` module.
        game_config: Embedding game configuration.
    """

    def __init__(
        self,
        config: PhaseConfig,
        faculty: nn.Module,
        game: EmbeddingReconstructionGame,
        game_config: EmbeddingGameConfig,
    ) -> None:
        super().__init__(config, faculty)
        self.game = game
        self.game_config = game_config
        self._loss_fn = self.build_loss()
        self._recon_loss = EmbeddingReconstructionLoss(weight=game_config.reconstruction_weight)

    def step(self, batch: dict[str, Tensor]) -> tuple[dict[str, Tensor], dict[str, Tensor]]:
        """Execute one embedding reconstruction training step.

        Args:
            batch: Dictionary with ``"agent_state"`` tensor of shape
                ``(batch_size, embedding_dim)`` from the prefetcher.

        Returns:
            Tuple ``(outputs, losses)`` where *outputs* contains all
            faculty outputs plus the reconstructed embedding, and *losses*
            maps loss names to scalar tensors including ``"total"``.
        """
        agent_state = batch["agent_state"]

        # 1. Forward through the language faculty
        lfm_outputs = self.faculty(agent_state)

        # 2. Decode LFM output back to embedding space
        reconstructed = self.game.decode_message(lfm_outputs)

        # 3. Merge all outputs
        outputs: dict[str, Tensor] = dict(lfm_outputs)
        outputs["game.reconstructed_embedding"] = reconstructed
        outputs["game.original_embedding"] = agent_state

        # 4. Build targets for loss functions
        targets: dict[str, Tensor] = {
            "game.original_embedding": agent_state,
        }

        # 5. Compute losses
        # Structural losses from PhaseConfig (may be empty)
        total_loss, loss_dict = self._loss_fn(outputs, targets)

        # Reconstruction loss
        recon_loss = self._recon_loss(outputs, targets) * self._recon_loss.weight
        loss_dict["reconstruction"] = recon_loss
        total_loss = total_loss + recon_loss

        # Extra losses from the faculty (e.g. commitment loss)
        extra = self.faculty.extra_losses()
        for k, v in extra.items():
            loss_dict[f"extra.{k}"] = v
            total_loss = total_loss + v

        loss_dict["total"] = total_loss

        return outputs, loss_dict


@register("phase", "embedding_referential_game")
class EmbeddingReferentialGamePhase(TrainingPhase):
    """Training phase for the embedding referential game.

    At each step this phase:

    1. Receives ``batch["agent_state"]`` (anchor embeddings) and
       ``batch["distractors"]`` from the prefetcher.
    2. Runs the ``LanguageFaculty`` on the anchor embedding.
    3. Assembles candidates (target + distractors) and shuffles their order.
    4. The game's receiver scores each candidate via dot-product similarity.
    5. Computes referential cross-entropy loss plus structural losses.
    6. Returns combined outputs and losses.

    Args:
        config: Phase configuration (loss weights, frozen modules, etc.).
        faculty: The ``LanguageFaculty`` model being trained.
        game: The ``EmbeddingReferentialGame`` module.
        game_config: Embedding game configuration.
    """

    def __init__(
        self,
        config: PhaseConfig,
        faculty: nn.Module,
        game: EmbeddingReferentialGame,
        game_config: EmbeddingGameConfig,
    ) -> None:
        super().__init__(config, faculty)
        self.game = game
        self.game_config = game_config
        self._loss_fn = self.build_loss()
        self._ref_loss = EmbeddingReferentialLoss(weight=game_config.referential_weight)

    def step(self, batch: dict[str, Tensor]) -> tuple[dict[str, Tensor], dict[str, Tensor]]:
        """Execute one embedding referential training step.

        Args:
            batch: Dictionary with:
                - ``"agent_state"``: ``(batch_size, dim)`` anchor embeddings.
                - ``"distractors"``: ``(batch_size, num_neg, dim)`` distractor
                  embeddings.
                - ``"target_idx"``: ``(batch_size,)`` indices (typically
                  zeros before shuffling).

        Returns:
            Tuple ``(outputs, losses)`` where *outputs* contains all
            faculty outputs plus receiver logits, and *losses* maps loss
            names to scalar tensors including ``"total"``.
        """
        agent_state = batch["agent_state"]  # (B, dim)
        distractors = batch["distractors"]  # (B, K_dist, dim)
        bs = agent_state.shape[0]

        # 1. Forward through the language faculty
        lfm_outputs = self.faculty(agent_state)

        # 2. Assemble candidates: target + distractors
        # Target: (B, dim) -> (B, 1, dim)
        target_unsqueezed = agent_state.unsqueeze(1)

        # Concatenate: (B, 1 + K_dist, dim)
        candidates = torch.cat([target_unsqueezed, distractors], dim=1)
        num_candidates = candidates.shape[1]

        # 3. Shuffle candidate order independently per batch element
        device = candidates.device
        perm = torch.stack(
            [torch.randperm(num_candidates, device=device) for _ in range(bs)]
        )  # (B, K)

        # Gather shuffled candidates: expand perm for (B, K, dim) indexing
        perm_expanded = perm.unsqueeze(-1).expand_as(candidates)
        candidates = torch.gather(candidates, 1, perm_expanded)

        # Track where the target ended up (originally at index 0)
        target_idx = (perm == 0).long().argmax(dim=1)  # (B,)

        # 4. Receiver scores candidates
        score_output = self.game.score_candidates(lfm_outputs, candidates)
        receiver_logits = score_output["receiver_logits"]  # (B, K)

        # 5. Merge outputs
        outputs: dict[str, Tensor] = dict(lfm_outputs)
        outputs["game.receiver_logits"] = receiver_logits
        outputs["game.target_idx"] = target_idx

        # Build targets for loss functions
        targets: dict[str, Tensor] = {
            "game.receiver_logits": receiver_logits,
            "game.target_idx": target_idx,
        }

        # 6. Compute losses
        # Structural losses from PhaseConfig (may be empty)
        total_loss, loss_dict = self._loss_fn(outputs, targets)

        # Referential loss
        ref_loss = self._ref_loss(outputs, targets) * self._ref_loss.weight
        loss_dict["referential"] = ref_loss
        total_loss = total_loss + ref_loss

        # Extra losses from the faculty
        extra = self.faculty.extra_losses()
        for k, v in extra.items():
            loss_dict[f"extra.{k}"] = v
            total_loss = total_loss + v

        loss_dict["total"] = total_loss

        return outputs, loss_dict

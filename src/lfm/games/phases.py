"""Training phases for LFM game-based emergent communication.

Provides two registered training phases that integrate game modules with
the LanguageFaculty training pipeline:

- ``ReconstructionGamePhase``: trains via scene reconstruction (autoencoder).
- ``ReferentialGamePhase``: trains via a Lewis signaling game (referential).

Both phases generate scenes on the fly, run the encoder -> faculty -> decoder
pipeline, compute game-specific losses alongside structural losses, and
return the combined loss dictionary expected by the training loop.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from torch import Tensor

from lfm._registry import register
from lfm.games.config import ReconstructionGameConfig, ReferentialGameConfig
from lfm.games.losses import ReferentialLoss, SceneReconstructionLoss
from lfm.games.reconstruction import ReconstructionGame
from lfm.games.referential import ReferentialGame
from lfm.games.scenes import SceneGenerator
from lfm.training.phase import TrainingPhase

if TYPE_CHECKING:
    from torch import nn

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


@register("phase", "reconstruction_game")
class ReconstructionGamePhase(TrainingPhase):
    """Training phase for the reconstruction game.

    At each step this phase:
    1. Generates a random scene.
    2. Encodes it to an agent-state vector via the game's encoder.
    3. Runs the LanguageFaculty forward pass to produce a message.
    4. Decodes the message back to scene predictions via the game's decoder.
    5. Computes the scene reconstruction loss plus any structural losses
       configured in the phase.
    6. Returns the combined outputs and losses.

    Args:
        config: Phase configuration (loss weights, frozen modules, etc.).
        faculty: The ``LanguageFaculty`` model being trained.
        game: The ``ReconstructionGame`` module (encoder + decoder).
        game_config: Configuration for the reconstruction game.
    """

    def __init__(
        self,
        config: PhaseConfig,
        faculty: nn.Module,
        game: ReconstructionGame,
        game_config: ReconstructionGameConfig,
    ) -> None:
        super().__init__(config, faculty)
        self.game = game
        self.game_config = game_config

        device = _infer_device(faculty)
        self.scene_gen = SceneGenerator(game_config.scene, device=device)
        self._loss_fn = self.build_loss()
        self._recon_loss = SceneReconstructionLoss(weight=game_config.reconstruction_weight)

    def step(self, batch: dict[str, Tensor]) -> tuple[dict[str, Tensor], dict[str, Tensor]]:
        """Execute one reconstruction game training step.

        The ``batch`` argument is accepted for interface compatibility but
        is not used -- scenes are generated on the fly from the
        ``SceneGenerator``.

        Args:
            batch: Input batch dictionary (unused; scenes are generated).

        Returns:
            A tuple ``(outputs, losses)`` where *outputs* contains all
            faculty outputs plus game-specific predictions (prefixed with
            ``"game."``), and *losses* maps loss names to scalar tensors
            including a ``"total"`` key.
        """
        bs = self.game_config.batch_size

        # 1. Generate a random scene
        scene = self.scene_gen.generate(bs)

        # 2. Encode scene -> agent_state
        agent_state = self.game.encode_scene(scene)

        # 3. Run the LanguageFaculty forward pass
        lfm_outputs = self.faculty(agent_state)

        # 4. Decode LFM outputs -> scene predictions
        predictions = self.game.decode_message(lfm_outputs)

        # 5. Merge all outputs into a single dict with "game." prefix
        outputs: dict[str, Tensor] = dict(lfm_outputs)

        # Store per-attribute logits separately for the loss function
        attr_logits = predictions["attr_logits"]
        for d, logits in enumerate(attr_logits):
            outputs[f"game.attr_logits.{d}"] = logits
        outputs["game.relation_logits"] = predictions["relation_logits"]

        # Build targets dict for the loss functions
        targets: dict[str, Tensor] = {
            "game.object_attrs": scene["object_attrs"],
            "game.relations": scene["relations"],
        }

        # 6. Compute losses
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


@register("phase", "referential_game")
class ReferentialGamePhase(TrainingPhase):
    """Training phase for the referential game.

    At each step this phase:
    1. Generates a target scene and distractor scenes.
    2. Encodes the target to an agent-state vector via the sender encoder.
    3. Runs the LanguageFaculty forward pass to produce a message.
    4. Assembles candidate scenes (target + distractors) in shuffled order.
    5. The receiver scores all candidates against the message.
    6. Computes the referential loss plus any structural losses.
    7. Returns the combined outputs and losses.

    Args:
        config: Phase configuration (loss weights, frozen modules, etc.).
        faculty: The ``LanguageFaculty`` model being trained.
        game: The ``ReferentialGame`` module (sender encoder + receiver).
        game_config: Configuration for the referential game.
    """

    def __init__(
        self,
        config: PhaseConfig,
        faculty: nn.Module,
        game: ReferentialGame,
        game_config: ReferentialGameConfig,
    ) -> None:
        super().__init__(config, faculty)
        self.game = game
        self.game_config = game_config

        device = _infer_device(faculty)
        self.scene_gen = SceneGenerator(game_config.scene, device=device)
        self._loss_fn = self.build_loss()
        self._ref_loss = ReferentialLoss(weight=game_config.referential_weight)

    def step(self, batch: dict[str, Tensor]) -> tuple[dict[str, Tensor], dict[str, Tensor]]:
        """Execute one referential game training step.

        The ``batch`` argument is accepted for interface compatibility but
        is not used -- scenes are generated on the fly.

        Args:
            batch: Input batch dictionary (unused; scenes are generated).

        Returns:
            A tuple ``(outputs, losses)`` where *outputs* contains all
            faculty outputs plus receiver logits (prefixed with ``"game."``),
            and *losses* maps loss names to scalar tensors including a
            ``"total"`` key.
        """
        bs = self.game_config.batch_size
        num_distractors = self.game_config.num_distractors
        num_candidates = 1 + num_distractors  # target + distractors

        # 1. Generate target scene and distractor scenes
        target_scene = self.scene_gen.generate(bs)
        distractor_data = self.scene_gen.generate_distractors(bs, num_distractors)

        # 2. Encode target -> agent_state
        agent_state = self.game.encode_target(target_scene)

        # 3. Run the LanguageFaculty forward pass
        lfm_outputs = self.faculty(agent_state)

        # 4. Assemble candidates: target + distractors, then shuffle
        # Target attrs: (batch, N, D) -> (batch, 1, N, D)
        target_attrs = target_scene["object_attrs"].unsqueeze(1)
        # Target relations: (batch, N, N, R) -> (batch, 1, N, N, R)
        target_rels = target_scene["relations"].unsqueeze(1)

        # Distractor attrs: (batch, K_dist, N, D)
        dist_attrs = distractor_data["object_attrs"]
        # Distractor relations: (batch, K_dist, N, N, R)
        dist_rels = distractor_data["relations"]

        # Concatenate: (batch, K_total, N, D) and (batch, K_total, N, N, R)
        # Target is at index 0 before shuffling
        candidate_attrs = torch.cat([target_attrs, dist_attrs], dim=1)
        candidate_rels = torch.cat([target_rels, dist_rels], dim=1)

        # Shuffle candidate order independently per batch element
        device = candidate_attrs.device
        # Generate random permutation indices per batch element
        perm = torch.stack(
            [torch.randperm(num_candidates, device=device) for _ in range(bs)]
        )  # (batch, K_total)

        # Gather shuffled candidates
        # For attrs: expand perm to (batch, K_total, N, D)
        perm_attrs = perm.unsqueeze(-1).unsqueeze(-1).expand_as(candidate_attrs)
        candidate_attrs = torch.gather(candidate_attrs, 1, perm_attrs)

        # For relations: expand perm to (batch, K_total, N, N, R)
        perm_rels = perm.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).expand_as(candidate_rels)
        candidate_rels = torch.gather(candidate_rels, 1, perm_rels)

        # Track where the target ended up after shuffling
        # Target was originally at index 0, find its new position
        target_idx = (perm == 0).long().argmax(dim=1)  # (batch,)

        # 5. Receiver scores candidates
        score_output = self.game.score_candidates(lfm_outputs, candidate_attrs, candidate_rels)
        receiver_logits = score_output["receiver_logits"]

        # 6. Merge outputs
        outputs: dict[str, Tensor] = dict(lfm_outputs)
        outputs["game.receiver_logits"] = receiver_logits
        outputs["game.target_idx"] = target_idx

        # Build targets dict for the loss functions
        targets: dict[str, Tensor] = {
            "game.receiver_logits": receiver_logits,
            "game.target_idx": target_idx,
        }

        # Compute losses
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

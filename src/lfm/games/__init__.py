"""Games subpackage for the LFM framework.

Provides configurable scene-based games (reconstruction, referential) used to
train and evaluate emergent communication between agents.  Includes game
modules, training phases, loss functions, metrics, and convenience runner
functions for quick experimentation.

Submodules:
    config -- Pydantic configuration models for scenes, encoders, decoders, and games.
    scenes -- Procedural scene generator.
    encoder -- SceneEncoder, SceneDecoder, and MessagePooler modules.
    losses -- SceneReconstructionLoss and ReferentialLoss.
    reconstruction -- ReconstructionGame module.
    referential -- ReferentialGame and ReceiverModule.
    phases -- ReconstructionGamePhase and ReferentialGamePhase (registered).
    metrics -- Game-specific evaluation metrics.
"""

from __future__ import annotations

from typing import Any

import torch

from lfm.games.config import (
    ReconstructionGameConfig,
    ReferentialGameConfig,
    SceneConfig,
    SceneDecoderConfig,
    SceneEncoderConfig,
)
from lfm.games.phases import ReconstructionGamePhase, ReferentialGamePhase
from lfm.games.reconstruction import ReconstructionGame
from lfm.games.referential import ReferentialGame
from lfm.games.scenes import SceneGenerator

__all__ = [
    "ReconstructionGame",
    "ReconstructionGameConfig",
    "ReconstructionGamePhase",
    "ReferentialGame",
    "ReferentialGameConfig",
    "ReferentialGamePhase",
    "SceneConfig",
    "SceneDecoderConfig",
    "SceneEncoderConfig",
    "SceneGenerator",
    "run_reconstruction_game",
    "run_referential_game",
]


def run_reconstruction_game(
    faculty_config: Any | None = None,
    game_config: ReconstructionGameConfig | None = None,
    steps: int = 5000,
    lr: float = 1e-3,
    log_every: int = 100,
    device: str = "cuda",
) -> dict[str, float]:
    """Convenience function to run a reconstruction game experiment.

    Builds the LanguageFaculty, ReconstructionGame, optimizer, and training
    phase from configs, runs the training loop, and returns final metrics.

    Uses sensible defaults if configs are not provided.  The default faculty
    config uses only phonology (no quantizer or channel), which means the
    faculty operates in pass-through mode on the agent-state vector.

    Args:
        faculty_config: A ``FacultyConfig`` for the language faculty.
            If ``None``, a default config with ``dim`` matching the
            encoder output is used.
        game_config: A ``ReconstructionGameConfig``.  If ``None``, defaults
            are used.
        steps: Number of training steps.
        lr: Learning rate.
        log_every: Frequency of metric logging (in steps).
        device: Device string (e.g. ``"cuda"``, ``"cpu"``).

    Returns:
        Dictionary mapping metric names to their final values.
    """
    from lfm.faculty.config import FacultyConfig
    from lfm.faculty.model import LanguageFaculty
    from lfm.games.metrics import AttributeAccuracy, RelationAccuracy
    from lfm.training.config import PhaseConfig

    if game_config is None:
        game_config = ReconstructionGameConfig()

    if faculty_config is None:
        faculty_config = FacultyConfig(dim=game_config.encoder.output_dim)

    # Build the faculty and game
    faculty = LanguageFaculty(faculty_config)
    game = ReconstructionGame(game_config)

    # Move to device
    torch_device = torch.device(device)
    faculty.to(torch_device)
    game.to(torch_device)

    # Build optimizer over both faculty and game parameters
    all_params = list(faculty.parameters()) + list(game.parameters())
    optimizer = torch.optim.Adam(all_params, lr=lr)

    # Build the training phase
    phase_config = PhaseConfig(
        name="reconstruction_game",
        steps=steps,
        losses={},
    )
    phase = ReconstructionGamePhase(
        config=phase_config,
        faculty=faculty,
        game=game,
        game_config=game_config,
    )

    # Set up metrics
    attr_acc = AttributeAccuracy()
    rel_acc = RelationAccuracy()

    # Training loop
    metrics_log: dict[str, float] = {}
    dummy_batch: dict[str, torch.Tensor] = {}

    for step_i in range(steps):
        faculty.train()
        game.train()

        outputs, losses = phase.step(dummy_batch)
        total_loss = losses["total"]

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        if step_i % log_every == 0 or step_i == steps - 1:
            with torch.no_grad():
                faculty.eval()
                game.eval()

                eval_scene = phase.scene_gen.generate(game_config.batch_size)
                eval_state = game.encode_scene(eval_scene)
                eval_lfm = faculty(eval_state)
                eval_preds = game.decode_message(eval_lfm)

                eval_outputs: dict[str, Any] = {}
                for d, logits in enumerate(eval_preds["attr_logits"]):
                    eval_outputs[f"game.attr_logits.{d}"] = logits
                eval_outputs["game.relation_logits"] = eval_preds["relation_logits"]
                eval_outputs["game.object_attrs"] = eval_scene["object_attrs"]
                eval_outputs["game.relations"] = eval_scene["relations"]

                a_acc = attr_acc.compute(eval_outputs)
                r_acc = rel_acc.compute(eval_outputs)

                metrics_log = {
                    "step": float(step_i),
                    "total_loss": total_loss.item(),
                    "attribute_accuracy": a_acc,
                    "relation_accuracy": r_acc,
                }

                if step_i % log_every == 0:
                    print(
                        f"[step {step_i:>5d}] loss={total_loss.item():.4f}  "
                        f"attr_acc={a_acc:.4f}  rel_acc={r_acc:.4f}"
                    )

    return metrics_log


def run_referential_game(
    faculty_config: Any | None = None,
    game_config: ReferentialGameConfig | None = None,
    steps: int = 5000,
    lr: float = 1e-3,
    log_every: int = 100,
    device: str = "cuda",
) -> dict[str, float]:
    """Convenience function to run a referential game experiment.

    Builds the LanguageFaculty, ReferentialGame, optimizer, and training
    phase from configs, runs the training loop, and returns final metrics.

    Uses sensible defaults if configs are not provided.

    Args:
        faculty_config: A ``FacultyConfig`` for the language faculty.
            If ``None``, a default config with ``dim`` matching the
            encoder output is used.
        game_config: A ``ReferentialGameConfig``.  If ``None``, defaults
            are used.
        steps: Number of training steps.
        lr: Learning rate.
        log_every: Frequency of metric logging (in steps).
        device: Device string (e.g. ``"cuda"``, ``"cpu"``).

    Returns:
        Dictionary mapping metric names to their final values.
    """
    from lfm.faculty.config import FacultyConfig
    from lfm.faculty.model import LanguageFaculty
    from lfm.games.metrics import ReferentialAccuracy
    from lfm.training.config import PhaseConfig

    if game_config is None:
        game_config = ReferentialGameConfig()

    if faculty_config is None:
        faculty_config = FacultyConfig(dim=game_config.encoder.output_dim)

    # Build the faculty and game
    faculty = LanguageFaculty(faculty_config)
    game = ReferentialGame(game_config)

    # Move to device
    torch_device = torch.device(device)
    faculty.to(torch_device)
    game.to(torch_device)

    # Build optimizer over both faculty and game parameters
    all_params = list(faculty.parameters()) + list(game.parameters())
    optimizer = torch.optim.Adam(all_params, lr=lr)

    # Build the training phase
    phase_config = PhaseConfig(
        name="referential_game",
        steps=steps,
        losses={},
    )
    phase = ReferentialGamePhase(
        config=phase_config,
        faculty=faculty,
        game=game,
        game_config=game_config,
    )

    # Set up metrics
    ref_acc = ReferentialAccuracy()

    # Training loop
    metrics_log: dict[str, float] = {}
    dummy_batch: dict[str, torch.Tensor] = {}
    num_candidates = 1 + game_config.num_distractors

    for step_i in range(steps):
        faculty.train()
        game.train()

        outputs, losses = phase.step(dummy_batch)
        total_loss = losses["total"]

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        if step_i % log_every == 0 or step_i == steps - 1:
            with torch.no_grad():
                faculty.eval()
                game.eval()

                # Evaluate referential accuracy
                r_acc = ref_acc.compute(outputs)

                metrics_log = {
                    "step": float(step_i),
                    "total_loss": total_loss.item(),
                    "referential_accuracy": r_acc,
                    "num_candidates": float(num_candidates),
                    "chance_accuracy": 1.0 / num_candidates,
                }

                if step_i % log_every == 0:
                    print(
                        f"[step {step_i:>5d}] loss={total_loss.item():.4f}  "
                        f"ref_acc={r_acc:.4f}  "
                        f"(chance={1.0 / num_candidates:.4f})"
                    )

    return metrics_log

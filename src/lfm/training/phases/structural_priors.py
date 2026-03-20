"""Phase 1: Structural priors from multilingual corpora.

This phase trains the faculty to learn basic structural regularities —
compositionality, sequential consistency, and token-level coherence — from
raw corpus data before any corruption or emergence pressure is applied.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from lfm._registry import register
from lfm.training.phase import TrainingPhase

if TYPE_CHECKING:
    from torch import Tensor, nn

    from lfm.training.config import PhaseConfig


@register("phase", "structural_priors")
class StructuralPriorsPhase(TrainingPhase):
    """Phase 1: Learn structural priors from multilingual corpora.

    Uses the default forward-then-loss step with no additional
    transformations applied to the input batch.

    Args:
        config: Phase configuration.
        faculty: The ``LanguageFaculty`` model being trained.
    """

    def __init__(self, config: PhaseConfig, faculty: nn.Module) -> None:
        super().__init__(config, faculty)
        self._loss_fn = self.build_loss()

    def step(self, batch: dict[str, Tensor]) -> tuple[dict[str, Tensor], dict[str, Tensor]]:
        """Run one training step using the default forward + loss pipeline.

        Args:
            batch: Input batch dictionary.

        Returns:
            Tuple of ``(outputs, losses)``.
        """
        return self.default_step(batch, self._loss_fn)

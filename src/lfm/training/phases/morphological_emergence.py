"""Phase 3: Morphological emergence pressure.

This phase applies training pressure that encourages the emergence of
morphological structure — recurring sub-word patterns, affixation-like
regularities, and compositional morpheme reuse.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from lfm._registry import register
from lfm.training.phase import TrainingPhase

if TYPE_CHECKING:
    from torch import Tensor, nn

    from lfm.training.config import PhaseConfig


@register("phase", "morphological_emergence")
class MorphologicalEmergencePhase(TrainingPhase):
    """Phase 3: Morphological emergence pressure.

    Trains the faculty with loss weights that reward morphological
    regularity — recurring sub-token patterns and compositional reuse of
    morpheme-like units.

    Args:
        config: Phase configuration.
        faculty: The ``LanguageFaculty`` model being trained.
    """

    def __init__(self, config: PhaseConfig, faculty: nn.Module) -> None:
        super().__init__(config, faculty)
        self._loss_fn = self.build_loss()

    def step(self, batch: dict[str, Tensor]) -> tuple[dict[str, Tensor], dict[str, Tensor]]:
        """Run one training step with morphological emergence losses.

        Args:
            batch: Input batch dictionary.

        Returns:
            Tuple of ``(outputs, losses)``.
        """
        return self.default_step(batch, self._loss_fn)

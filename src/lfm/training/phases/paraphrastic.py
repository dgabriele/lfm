"""Phase 4: Paraphrastic generation diversity.

This phase trains the faculty to produce structurally diverse outputs for
similar inputs, encouraging paraphrastic capacity — the ability to express
the same underlying representation in multiple well-formed surface forms.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from lfm._registry import register
from lfm.training.phase import TrainingPhase

if TYPE_CHECKING:
    from torch import Tensor, nn

    from lfm.training.config import PhaseConfig


@register("phase", "paraphrastic")
class ParaphrasticPhase(TrainingPhase):
    """Phase 4: Paraphrastic generation diversity.

    Trains the faculty with losses that reward structural variation,
    encouraging multiple well-formed surface realizations for a given
    internal representation.

    Args:
        config: Phase configuration.
        faculty: The ``LanguageFaculty`` model being trained.
    """

    def __init__(self, config: PhaseConfig, faculty: nn.Module) -> None:
        super().__init__(config, faculty)
        self._loss_fn = self.build_loss()

    def step(self, batch: dict[str, Tensor]) -> tuple[dict[str, Tensor], dict[str, Tensor]]:
        """Run one training step with paraphrastic diversity losses.

        Args:
            batch: Input batch dictionary.

        Returns:
            Tuple of ``(outputs, losses)``.
        """
        return self.default_step(batch, self._loss_fn)

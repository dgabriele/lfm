"""Phase 2: Structural corruption and recomposition.

This phase applies controlled corruption to inputs before the forward pass,
training the faculty to reconstruct well-formed structure from degraded
input.  The corruption strategy is a pluggable component to be implemented
by downstream research.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from lfm._registry import register
from lfm.training.phase import TrainingPhase

if TYPE_CHECKING:
    from torch import Tensor, nn

    from lfm.training.config import PhaseConfig


@register("phase", "corruption")
class CorruptionPhase(TrainingPhase):
    """Phase 2: Structural corruption and recomposition.

    Applies a corruption transform to each batch before the standard
    forward pass.  The faculty is trained to produce well-formed outputs
    despite degraded inputs.

    Args:
        config: Phase configuration.
        faculty: The ``LanguageFaculty`` model being trained.
    """

    def __init__(self, config: PhaseConfig, faculty: nn.Module) -> None:
        super().__init__(config, faculty)
        self._loss_fn = self.build_loss()

    def step(self, batch: dict[str, Tensor]) -> tuple[dict[str, Tensor], dict[str, Tensor]]:
        """Run one training step with corruption applied to the batch.

        Args:
            batch: Input batch dictionary.

        Returns:
            Tuple of ``(outputs, losses)``.
        """
        corrupted_batch = self._corrupt(batch)
        return self.default_step(corrupted_batch, self._loss_fn)

    def _corrupt(self, batch: dict[str, Tensor]) -> dict[str, Tensor]:
        """Apply structural corruption to the input batch.

        This is a research-facing extension point.  Concrete corruption
        strategies (token dropping, span masking, permutation, etc.) should
        be implemented here or delegated to a registered corruption module.

        Args:
            batch: Original input batch.

        Returns:
            A corrupted copy of the batch.

        Raises:
            NotImplementedError: Always — must be overridden or extended with
                a concrete corruption strategy.
        """
        raise NotImplementedError(
            "CorruptionPhase._corrupt() must be implemented with a concrete "
            "corruption strategy (e.g. token dropping, span masking)."
        )

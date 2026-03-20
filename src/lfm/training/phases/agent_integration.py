"""Phase 5: Agent-integrated training.

This phase trains the faculty end-to-end within a multi-agent communication
loop.  Data is sourced from agent interactions rather than static corpora,
and the loss weights are tuned for communication effectiveness alongside
structural well-formedness.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from lfm._registry import register
from lfm.training.phase import TrainingPhase

if TYPE_CHECKING:
    from torch import Tensor, nn

    from lfm.training.config import PhaseConfig


@register("phase", "agent_integration")
class AgentIntegrationPhase(TrainingPhase):
    """Phase 5: Agent-integrated training.

    Trains the faculty with data drawn from agent interactions.  Loss
    weights typically emphasize communication success alongside
    structural quality.

    Args:
        config: Phase configuration.
        faculty: The ``LanguageFaculty`` model being trained.
    """

    def __init__(self, config: PhaseConfig, faculty: nn.Module) -> None:
        super().__init__(config, faculty)
        self._loss_fn = self.build_loss()

    def step(self, batch: dict[str, Tensor]) -> tuple[dict[str, Tensor], dict[str, Tensor]]:
        """Run one training step with agent-integration losses.

        Args:
            batch: Input batch dictionary.

        Returns:
            Tuple of ``(outputs, losses)``.
        """
        return self.default_step(batch, self._loss_fn)

"""Abstract base module for all LFM pipeline components.

Every trainable component in the LFM pipeline (quantizer, phonology, morphology,
syntax, sentence encoder, channel) inherits from ``LFMModule``.  This base class
combines ``torch.nn.Module`` with Python's ``ABC`` and enforces a uniform interface
for forward passes and auxiliary losses.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, ClassVar

import torch
from torch import Tensor, nn


class LFMModule(ABC, nn.Module):
    """Abstract base class for all LFM pipeline components.

    Subclasses must:
        1. Set the ``output_prefix`` class variable (e.g. ``"quantization"``).
           The ``LanguageFaculty`` orchestrator uses this to namespace output
           keys and avoid collisions between stages.
        2. Implement ``forward()`` returning a ``dict[str, Tensor]``.

    Attributes:
        output_prefix: String prefix used to namespace this module's outputs
            in the combined pipeline output dictionary.
        config: The configuration object for this module.
    """

    output_prefix: ClassVar[str] = ""

    def __init__(self, config: Any) -> None:
        super().__init__()
        self.config = config

    @abstractmethod
    def forward(self, *args: Any, **kwargs: Any) -> dict[str, Tensor]:
        """Run the module's forward pass.

        Returns:
            Dictionary mapping output names to tensors.  Keys should *not*
            include the ``output_prefix`` — namespacing is handled by the
            orchestrator.
        """

    def extra_losses(self) -> dict[str, Tensor]:
        """Return auxiliary / regularization losses produced by this module.

        The default implementation returns an empty dict.  Override this in
        subclasses that produce commitment losses, entropy penalties, etc.

        Returns:
            Dictionary mapping loss names to scalar tensors.
        """
        return {}

    def _zero_loss(self) -> Tensor:
        """Convenience helper that returns a zero scalar on the module's device.

        Useful inside ``extra_losses`` when a loss term is conditionally absent.
        """
        p = next(self.parameters(), None)
        device = p.device if p is not None else torch.device("cpu")
        return torch.zeros(1, device=device).squeeze()

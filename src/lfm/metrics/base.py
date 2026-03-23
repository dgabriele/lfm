"""Base metric interface for LFM evaluation.

Metrics track a running value that is updated per-batch and can be
queried for the latest result.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from torch import Tensor


class Metric(ABC):
    """Abstract base class for evaluation metrics.

    Subclasses implement :meth:`compute` which receives the combined
    output dictionary from a forward pass and returns a scalar value.
    The base class provides :meth:`update` / :meth:`result` for
    tracking the latest computed value.

    Args:
        name: Human-readable metric name used for logging.
    """

    def __init__(self, name: str) -> None:
        self.name = name
        self._value: float = 0.0

    @abstractmethod
    def compute(self, outputs: dict[str, Tensor]) -> float:
        """Compute the metric from model outputs.

        Args:
            outputs: Combined output dictionary from a forward pass.

        Returns:
            Scalar metric value.
        """

    def update(self, outputs: dict[str, Tensor]) -> float:
        """Compute and store the metric value.

        Args:
            outputs: Combined output dictionary from a forward pass.

        Returns:
            The computed metric value.
        """
        self._value = self.compute(outputs)
        return self._value

    def result(self) -> float:
        """Return the most recently computed value."""
        return self._value

    def reset(self) -> None:
        """Reset the stored value to zero."""
        self._value = 0.0

"""Abstract base class for evaluation metrics.

All metric implementations must subclass ``Metric`` and implement the
``compute`` method.  The base class provides accumulation, averaging,
and reset functionality for streaming metric computation over batches.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from torch import Tensor


class Metric(ABC):
    """Abstract base class for evaluation metrics.

    A metric computes a scalar value from a dictionary of model outputs,
    accumulates values over multiple batches, and provides an averaged
    result.

    Subclasses must implement:
        - ``compute``: calculate the metric value from a single batch of
          outputs.

    Usage::

        metric = SomeMetric("my_metric")
        for batch_outputs in evaluation_loop:
            metric.update(batch_outputs)
        print(metric.result())
        metric.reset()

    Attributes:
        name: Human-readable name for this metric.
    """

    def __init__(self, name: str) -> None:
        self.name = name
        self._values: list[float] = []

    @abstractmethod
    def compute(self, outputs: dict[str, Tensor]) -> float:
        """Compute the metric value from a single batch of outputs.

        Args:
            outputs: Dictionary mapping output names to tensors, as
                produced by the LFM pipeline.

        Returns:
            Scalar metric value for this batch.
        """
        ...

    def update(self, outputs: dict[str, Tensor]) -> None:
        """Compute and accumulate the metric for a single batch.

        Args:
            outputs: Dictionary mapping output names to tensors.
        """
        self._values.append(self.compute(outputs))

    def result(self) -> float:
        """Return the running average of all accumulated values.

        Returns:
            Mean of all accumulated metric values, or ``0.0`` if no
            values have been recorded.
        """
        if not self._values:
            return 0.0
        return sum(self._values) / len(self._values)

    def reset(self) -> None:
        """Clear all accumulated metric values."""
        self._values.clear()

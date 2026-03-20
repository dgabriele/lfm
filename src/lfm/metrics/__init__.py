"""Metrics subsystem for the LFM framework.

Provides the abstract ``Metric`` base class for defining evaluation metrics
that can be accumulated over batches and averaged.
"""

from __future__ import annotations

from lfm.metrics.base import Metric

__all__ = ["Metric"]

"""Weighted interleaving of multiple corpus sources."""

from __future__ import annotations

import logging
import random
from typing import Iterator

from lfm.qwen_targets.corpora.base import CorpusSource, CorpusText

logger = logging.getLogger(__name__)


class MixedCorpusLoader:
    """Interleave texts from several :class:`CorpusSource` objects.

    At construction time each source is given a relative sampling
    weight.  At iteration time a weighted random source is chosen for
    each draw.  When a source is exhausted it is removed from the mix
    and the remaining weights are renormalized.  Iteration stops when
    all sources are exhausted or when ``total_limit`` examples have
    been yielded.

    Args:
        sources: List of :class:`CorpusSource` instances to mix.
        weights: Parallel list of sampling weights.  Normalized
            internally.  ``None`` uses equal weights.
        total_limit: Optional hard cap on number of examples yielded.
        seed: RNG seed for deterministic shuffling.
    """

    def __init__(
        self,
        sources: list[CorpusSource],
        weights: list[float] | None = None,
        total_limit: int | None = None,
        seed: int = 42,
    ) -> None:
        if not sources:
            raise ValueError("MixedCorpusLoader needs at least one source")
        if weights is None:
            weights = [1.0] * len(sources)
        if len(weights) != len(sources):
            raise ValueError("weights length must match sources length")
        if any(w < 0 for w in weights):
            raise ValueError("weights must be non-negative")
        total = sum(weights)
        if total <= 0:
            raise ValueError("weights must sum to a positive value")

        self.sources = sources
        self.weights = [w / total for w in weights]
        self.total_limit = total_limit
        self._rng = random.Random(seed)

    def iterate(self) -> Iterator[CorpusText]:
        """Yield mixed texts until exhaustion or ``total_limit`` reached."""
        iterators = [iter(s) for s in self.sources]
        weights = list(self.weights)
        active = list(range(len(iterators)))
        total_emitted = 0

        while active:
            sub_weights = [weights[i] for i in active]
            picked_slot = self._rng.choices(range(len(active)), weights=sub_weights)[0]
            picked = active[picked_slot]
            try:
                yield next(iterators[picked])
                total_emitted += 1
                if self.total_limit is not None and total_emitted >= self.total_limit:
                    logger.info(
                        "MixedCorpusLoader: reached total_limit=%d", self.total_limit,
                    )
                    return
            except StopIteration:
                logger.info(
                    "Source %r exhausted after mixing",
                    self.sources[picked].name,
                )
                active.pop(picked_slot)

        logger.info(
            "MixedCorpusLoader: all %d sources exhausted after %d examples",
            len(self.sources), total_emitted,
        )

    def __iter__(self) -> Iterator[CorpusText]:
        return self.iterate()

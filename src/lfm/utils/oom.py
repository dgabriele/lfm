"""CUDA OOM auto-recovery helpers.

Training loops that share this pattern should all funnel through
:func:`shrink_on_oom` rather than re-implementing the try/except boiler-
plate.  The helper:

- Re-raises non-OOM ``RuntimeError`` untouched.
- Refuses to shrink below a floor (usually 4) and instead surfaces a
  clear error — OOM at the floor almost always means allocator
  fragmentation, not batch-size pressure.
- Empties the CUDA cache on every shrink (reduces reserved-but-
  unallocated fragmentation after aborted half-allocations).
- Returns the new capacity.  Callers are expected to apply the cap by
  slicing the current batch on subsequent iterations.
"""

from __future__ import annotations

import logging
from typing import Optional

import torch

logger = logging.getLogger(__name__)


def shrink_on_oom(
    exc: RuntimeError,
    current: int,
    *,
    floor: int = 4,
    ratio: float = 0.9,
    label: str = "",
    optimizer: Optional[torch.optim.Optimizer] = None,
) -> int:
    """Handle a CUDA OOM by returning a shrunk capacity.

    Args:
        exc: The caught exception.  Re-raised if not an OOM.
        current: Current batch-size (or batch-cap) that failed.
        floor: Minimum capacity we'll shrink to.  Going below this is
            treated as allocator fragmentation, not a sizing issue.
        ratio: Multiplicative shrinkage.
        label: Short context for the log line (e.g. ``"step 1375"``).
        optimizer: Optional — if passed, its ``zero_grad(set_to_none=True)``
            is called so a half-computed gradient doesn't leak into the
            next step.

    Returns:
        New capacity in ``[floor, current)``.

    Raises:
        RuntimeError: If ``exc`` isn't an OOM, or if ``current`` is
            already at ``floor``.
    """
    if "out of memory" not in str(exc):
        raise exc

    if optimizer is not None:
        optimizer.zero_grad(set_to_none=True)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    if current <= floor:
        raise RuntimeError(
            f"OOM {label} even at cap={floor}. "
            "This usually indicates allocator fragmentation; "
            "set PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True "
            "and/or reduce max sequence length.",
        ) from exc

    new_cap = max(floor, int(current * ratio))
    logger.warning("OOM %s — reducing cap %d → %d", label, current, new_cap)
    return new_cap

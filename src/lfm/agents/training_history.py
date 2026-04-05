"""Training history logger — persists per-step metrics as a DataFrame.

Captures all loss terms, accuracy, and metadata at each optimizer step
for post-hoc analysis.  Saves as compressed parquet alongside checkpoints.

Usage::

    history = TrainingHistory()
    history.record(step=100, loss=0.5, accuracy=0.9, hard_ratio=0.8,
                   ce_loss=0.3, surface_loss=1.2, turn_sim=0.4)
    history.save("data/dialogue_game/history.parquet")

    # Post-hoc analysis
    df = TrainingHistory.load("data/dialogue_game/history.parquet")
    df.plot(x="step", y=["loss", "ce_loss", "surface_loss"])
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd


class TrainingHistory:
    """Accumulate per-step metrics and persist as parquet."""

    def __init__(self) -> None:
        self._records: list[dict] = []

    def record(self, **kwargs) -> None:
        """Record one step's metrics.

        All keyword arguments become columns.  Typical keys:
        step, loss, accuracy, hard_ratio, ce_loss, surface_loss,
        turn_sim, num_phrases, tok_per_turn, total_tok, vram_mb.
        """
        self._records.append(kwargs)

    def save(self, path: str) -> None:
        """Save history to compressed parquet."""
        if not self._records:
            return
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        df = pd.DataFrame(self._records)
        df.to_parquet(p, index=False, compression="zstd")

    @staticmethod
    def load(path: str) -> pd.DataFrame:
        """Load history from parquet."""
        return pd.read_parquet(path)

    @property
    def num_records(self) -> int:
        return len(self._records)

    def to_dataframe(self) -> pd.DataFrame:
        """Convert current records to DataFrame (without saving)."""
        return pd.DataFrame(self._records)

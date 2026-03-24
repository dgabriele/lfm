"""Training session history tracking.

Records which config was used for which epoch ranges, enabling
post-hoc analysis of training runs across config changes and resumes.
Stored as a human-readable JSON file alongside checkpoints.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_FILENAME = "training_history.json"


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _config_to_dict(config: Any) -> dict:
    """Serialize a Pydantic config to a plain dict."""
    if hasattr(config, "model_dump"):
        return config.model_dump()
    if hasattr(config, "dict"):
        return config.dict()
    return {k: v for k, v in vars(config).items() if not k.startswith("_")}


class TrainingHistory:
    """Append-only log of training sessions and their configs.

    Each session records the epoch range, config snapshot, timestamps,
    and best validation loss.  A new session is opened on each
    ``start_session()`` call (typically at training start or resume)
    and closed by ``end_session()`` at completion or early stop.

    Args:
        output_dir: Directory containing checkpoints (e.g. ``data/``).
    """

    def __init__(self, output_dir: str | Path) -> None:
        self.path = Path(output_dir) / _FILENAME
        self._sessions: list[dict] = []
        if self.path.exists():
            try:
                with open(self.path) as f:
                    data = json.load(f)
                self._sessions = data.get("sessions", [])
            except (json.JSONDecodeError, KeyError):
                logger.warning("Corrupt training history at %s — starting fresh", self.path)
                self._sessions = []

    def start_session(
        self,
        start_epoch: int,
        config: Any,
        spm_hash: str | None = None,
    ) -> None:
        """Record the start of a new training session."""
        self._sessions.append({
            "start_epoch": start_epoch,
            "end_epoch": start_epoch,  # updated by end_session
            "started_at": _now_iso(),
            "ended_at": None,
            "config": _config_to_dict(config),
            "best_val_loss": None,
            "spm_hash": spm_hash,
        })
        self._save()
        logger.info(
            "Training history: session %d started at epoch %d",
            len(self._sessions), start_epoch,
        )

    def end_session(
        self,
        end_epoch: int,
        best_val_loss: float | None = None,
    ) -> None:
        """Record the end of the current training session."""
        if not self._sessions:
            return
        self._sessions[-1]["end_epoch"] = end_epoch
        self._sessions[-1]["ended_at"] = _now_iso()
        if best_val_loss is not None:
            self._sessions[-1]["best_val_loss"] = best_val_loss
        self._save()
        logger.info(
            "Training history: session %d ended at epoch %d (best_val=%.4f)",
            len(self._sessions), end_epoch,
            best_val_loss if best_val_loss is not None else float("nan"),
        )

    def update_epoch(self, epoch: int, best_val_loss: float) -> None:
        """Update the current session's end_epoch and best_val_loss.

        Called each epoch so the history stays current even if
        training is killed without a clean end_session().
        """
        if not self._sessions:
            return
        self._sessions[-1]["end_epoch"] = epoch
        self._sessions[-1]["best_val_loss"] = best_val_loss
        self._save()

    @property
    def sessions(self) -> list[dict]:
        return list(self._sessions)

    def _save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.path, "w") as f:
            json.dump({"sessions": self._sessions}, f, indent=2, default=str)

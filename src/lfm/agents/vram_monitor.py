"""Background VRAM monitoring via a separate process.

Samples GPU memory every N seconds in a daemon process using nvidia-smi
(no torch dependency in the monitor process) and records a time series
of (timestamp, allocated_mb, stage, step).  The training game sets stage
markers via shared memory to correlate VRAM usage with operations.

Usage::

    monitor = VRAMMonitor(save_path="data/dialogue_game/vram_trace.npz")
    monitor.start()

    monitor.stage = "phase2_turn0"
    # ... do work ...

    monitor.stop()  # final save
"""

from __future__ import annotations

import multiprocessing as mp
import subprocess
import time
from pathlib import Path

import numpy as np


def _monitor_loop(
    device: int,
    interval: float,
    save_path: str,
    save_every: int,
    stage_val: mp.Array,
    step_val: mp.Value,
    running: mp.Value,
) -> None:
    """Sampling loop (runs in daemon process)."""
    timestamps: list[float] = []
    allocated: list[int] = []
    stages: list[str] = []
    steps: list[int] = []
    start = time.monotonic()

    while running.value:
        try:
            result = subprocess.run(
                ["nvidia-smi",
                 f"--id={device}",
                 "--query-gpu=memory.used",
                 "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=5,
            )
            used_mb = int(result.stdout.strip())

            timestamps.append(time.monotonic() - start)
            allocated.append(used_mb)
            stages.append(stage_val.value.decode().rstrip("\x00"))
            steps.append(step_val.value)

            if len(timestamps) % save_every == 0 and save_path:
                _save(save_path, timestamps, allocated, stages, steps)

        except Exception:
            pass
        time.sleep(interval)

    # Final save
    if save_path and timestamps:
        _save(save_path, timestamps, allocated, stages, steps)


def _save(
    path: str,
    timestamps: list[float],
    allocated: list[int],
    stages: list[str],
    steps: list[int],
) -> None:
    """Write trace to .npz."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    unique_stages = sorted(set(stages))
    stage_to_idx = {s: i for i, s in enumerate(unique_stages)}
    np.savez_compressed(
        path,
        timestamps=np.array(timestamps, dtype=np.float64),
        allocated_mb=np.array(allocated, dtype=np.int32),
        stage_indices=np.array([stage_to_idx[s] for s in stages], dtype=np.int32),
        stage_names=np.array(unique_stages),
        steps=np.array(steps, dtype=np.int32),
    )


class VRAMMonitor:
    """Background GPU memory sampler using a separate process.

    Args:
        interval: Sampling interval in seconds.
        device: CUDA device index.
        save_path: Path for .npz trace file.
        save_every: Flush to disk every N samples (default 30 = ~1 min at 2s interval).
    """

    def __init__(
        self,
        interval: float = 2.0,
        device: int = 0,
        save_path: str | None = None,
        save_every: int = 30,
    ) -> None:
        self.interval = interval
        self.device = device
        self._save_path = save_path
        self._save_every = save_every

        # Shared state between main process and monitor process
        self._stage_val = mp.Array("c", 64)  # 64-byte char buffer for stage name
        self._step_val = mp.Value("i", 0)
        self._running = mp.Value("i", 0)
        self._process: mp.Process | None = None

    @property
    def stage(self) -> str:
        return self._stage_val.value.decode().rstrip("\x00")

    @stage.setter
    def stage(self, name: str) -> None:
        encoded = name.encode()[:63]  # truncate to fit buffer
        self._stage_val.value = encoded + b"\x00" * (64 - len(encoded))

    def set_step(self, step: int) -> None:
        self._step_val.value = step

    def start(self) -> None:
        """Start background monitor process."""
        if self._save_path is None:
            return
        self._running.value = 1
        self._process = mp.Process(
            target=_monitor_loop,
            args=(
                self.device,
                self.interval,
                self._save_path,
                self._save_every,
                self._stage_val,
                self._step_val,
                self._running,
            ),
            daemon=True,
        )
        self._process.start()

    def stop(self) -> None:
        """Stop monitor and wait for final save."""
        self._running.value = 0
        if self._process is not None:
            self._process.join(timeout=10.0)
            self._process = None

    def save(self, path: str) -> None:
        """Update save path (for compatibility with old call sites)."""
        self._save_path = path

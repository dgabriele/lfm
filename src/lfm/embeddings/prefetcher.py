"""Asynchronous batch prefetcher for embedding sampling.

Runs a background thread that continuously pulls batches from a
``StratifiedSampler``, optionally pins them to host memory, and enqueues
them for the main training thread.  The main thread dequeues and moves
each batch to the GPU with ``non_blocking=True``, keeping the device fed
while the next CPU-side mmap read happens concurrently.
"""

from __future__ import annotations

import queue
import threading
from typing import TYPE_CHECKING

import torch

from lfm.utils.logging import get_logger

if TYPE_CHECKING:
    from lfm.embeddings.sampler import StratifiedSampler

logger = get_logger(__name__)


class AsyncPrefetcher:
    """Background-thread prefetcher for ``StratifiedSampler`` batches.

    The worker thread runs an infinite loop:
    ``sampler.__next__()`` -> optional ``pin_memory()`` -> ``queue.put()``.

    The main thread calls ``__next__()`` which does:
    ``queue.get()`` -> ``.to(device, non_blocking=True)``.

    Args:
        sampler: An infinite ``StratifiedSampler`` instance.
        device: Target device for GPU transfer (e.g. ``torch.device("cuda")``).
        prefetch_batches: Maximum number of batches to buffer in the queue.
        pin_memory: Whether to pin each batch's tensors to page-locked memory
            for faster DMA transfers to the GPU.
    """

    def __init__(
        self,
        sampler: StratifiedSampler,
        device: torch.device,
        prefetch_batches: int = 4,
        pin_memory: bool = True,
    ) -> None:
        self._sampler = sampler
        self._device = device
        self._queue: queue.Queue[dict[str, torch.Tensor]] = queue.Queue(maxsize=prefetch_batches)
        self._pin_memory = pin_memory
        self._stop_event = threading.Event()
        self._worker: threading.Thread | None = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Start the background prefetch worker thread.

        Does nothing if the worker is already running.
        """
        if self._worker is not None and self._worker.is_alive():
            return
        self._stop_event.clear()
        self._worker = threading.Thread(target=self._worker_fn, daemon=True)
        self._worker.start()
        logger.info(
            "Prefetcher started (queue_size=%d, pin_memory=%s, device=%s)",
            self._queue.maxsize,
            self._pin_memory,
            self._device,
        )

    def stop(self) -> None:
        """Signal the worker to stop and drain the queue.

        Blocks until the worker thread has terminated.
        """
        self._stop_event.set()
        # Drain the queue so the worker is not blocked on a full put().
        while True:
            try:
                self._queue.get_nowait()
            except queue.Empty:
                break
        if self._worker is not None:
            self._worker.join(timeout=10.0)
            if self._worker.is_alive():
                logger.warning("Prefetcher worker did not terminate within timeout.")
            self._worker = None
        logger.info("Prefetcher stopped.")

    # ------------------------------------------------------------------
    # Iterator protocol
    # ------------------------------------------------------------------

    def __iter__(self) -> AsyncPrefetcher:
        """Start the worker (if not running) and return self as an iterator."""
        self.start()
        return self

    def __next__(self) -> dict[str, torch.Tensor]:
        """Dequeue the next batch and transfer it to the target device.

        Returns:
            Dictionary of GPU-resident tensors ready for the forward pass.

        Raises:
            StopIteration: If the worker has died or a timeout occurs.
        """
        if self._stop_event.is_set():
            raise StopIteration

        if self._worker is None or not self._worker.is_alive():
            # Worker died unexpectedly -- check if there are remaining batches.
            try:
                batch = self._queue.get_nowait()
            except queue.Empty:
                raise StopIteration
            return {k: v.to(self._device, non_blocking=True) for k, v in batch.items()}

        try:
            batch = self._queue.get(timeout=30.0)
        except queue.Empty:
            logger.error("Prefetcher queue timed out after 30 seconds.")
            raise StopIteration

        return {k: v.to(self._device, non_blocking=True) for k, v in batch.items()}

    def __del__(self) -> None:
        """Ensure the worker is stopped when the prefetcher is garbage-collected."""
        self.stop()

    # ------------------------------------------------------------------
    # Worker
    # ------------------------------------------------------------------

    def _worker_fn(self) -> None:
        """Background worker loop.

        Continuously pulls batches from the sampler, optionally pins them
        to page-locked memory, and enqueues them.  Exits cleanly when the
        stop event is set.
        """
        try:
            while not self._stop_event.is_set():
                batch = next(self._sampler)

                if self._pin_memory:
                    batch = {k: v.pin_memory() for k, v in batch.items()}

                # Try to enqueue; if the queue is full, retry with a short
                # timeout to allow checking the stop event.
                while not self._stop_event.is_set():
                    try:
                        self._queue.put(batch, timeout=1.0)
                        break
                    except queue.Full:
                        continue
        except Exception as e:
            logger.error("Prefetcher worker error: %s", e, exc_info=True)

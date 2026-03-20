"""Evaluation metrics for embedding-domain games.

Provides metrics for tracking reconstruction quality, referential accuracy,
and curriculum progression during embedding game training.
"""

from __future__ import annotations

import torch.nn.functional as F  # noqa: N812
from torch import Tensor

from lfm.metrics.base import Metric


class EmbeddingReconstructionSimilarity(Metric):
    """Mean cosine similarity between original and reconstructed embeddings.

    Reads the original and reconstructed embeddings from the output
    dictionary and computes the mean cosine similarity across the batch.
    A value of 1.0 indicates perfect directional alignment.

    Expected output keys:
        - ``"game.reconstructed_embedding"``: ``(batch, dim)`` float tensor.
        - ``"game.original_embedding"``: ``(batch, dim)`` float tensor.
    """

    def __init__(self) -> None:
        super().__init__("embedding_reconstruction_similarity")

    def compute(self, outputs: dict[str, Tensor]) -> float:
        """Compute mean cosine similarity for the batch.

        Args:
            outputs: Combined output dictionary containing the original
                and reconstructed embeddings.

        Returns:
            Scalar cosine similarity in ``[-1, 1]``, or ``0.0`` if the
            required keys are missing.
        """
        if (
            "game.reconstructed_embedding" not in outputs
            or "game.original_embedding" not in outputs
        ):
            return 0.0

        reconstructed = outputs["game.reconstructed_embedding"]
        original = outputs["game.original_embedding"]

        cosine_sim = F.cosine_similarity(reconstructed, original, dim=-1)
        return cosine_sim.mean().item()


class EmbeddingReferentialAccuracy(Metric):
    """Top-1 accuracy of the receiver identifying the target embedding.

    Compares the receiver's argmax prediction against the ground-truth
    target index.

    Expected output keys:
        - ``"game.receiver_logits"``: ``(batch, K)`` float logits.
        - ``"game.target_idx"``: ``(batch,)`` int64 target index.
    """

    def __init__(self) -> None:
        super().__init__("embedding_referential_accuracy")

    def compute(self, outputs: dict[str, Tensor]) -> float:
        """Compute top-1 referential accuracy for the batch.

        Args:
            outputs: Combined output dictionary containing receiver logits
                and the target index.

        Returns:
            Scalar accuracy in ``[0, 1]``, or ``0.0`` if the required
            keys are missing.
        """
        if "game.receiver_logits" not in outputs or "game.target_idx" not in outputs:
            return 0.0

        logits = outputs["game.receiver_logits"]  # (batch, K)
        target_idx = outputs["game.target_idx"]  # (batch,)

        preds = logits.argmax(dim=-1)  # (batch,)
        correct = (preds == target_idx).float().mean().item()
        return correct


class CurriculumDifficulty(Metric):
    """Tracks the current curriculum difficulty level.

    Unlike other metrics that read from model outputs, this metric is
    updated externally by setting ``current_difficulty`` before each
    ``update`` call.

    Expected output keys: none (uses the stored ``current_difficulty``).
    """

    def __init__(self) -> None:
        super().__init__("curriculum_difficulty")
        self.current_difficulty: float = 0.0

    def compute(self, outputs: dict[str, Tensor]) -> float:
        """Return the current curriculum difficulty.

        The ``outputs`` argument is ignored; the difficulty is set
        externally via the ``current_difficulty`` attribute.

        Args:
            outputs: Unused (required by ``Metric`` interface).

        Returns:
            Current curriculum difficulty in ``[0, 1]``.
        """
        return self.current_difficulty

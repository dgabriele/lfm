"""Loss functions for embedding-domain games.

``EmbeddingReconstructionLoss`` combines cosine similarity loss with
mean-squared error to drive the language faculty toward preserving
embedding content through its emergent language.

``EmbeddingReferentialLoss`` uses cross-entropy to reward the receiver
for correctly identifying the target embedding among distractors.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F  # noqa: N812
from torch import Tensor

from lfm._registry import register
from lfm.core.loss import LFMLoss


@register("loss", "embedding_reconstruction")
class EmbeddingReconstructionLoss(LFMLoss):
    """Cosine + MSE loss between original and reconstructed embeddings.

    Computes ``(1 - cosine_similarity) + 0.1 * MSE`` averaged over the
    batch.  The cosine term encourages directional alignment while the
    MSE term penalises magnitude differences.

    Expected output keys:
        - ``"game.reconstructed_embedding"``: ``(batch, dim)`` float tensor.

    Expected target keys:
        - ``"game.original_embedding"``: ``(batch, dim)`` float tensor.

    Args:
        config: Optional loss configuration (unused).
        weight: Multiplicative weight for this loss term.
        mse_weight: Relative weight of the MSE component.
    """

    def __init__(
        self,
        config: object = None,
        weight: float = 1.0,
        mse_weight: float = 0.1,
    ) -> None:
        super().__init__(config, weight)
        self.mse_weight = mse_weight

    def forward(
        self,
        outputs: dict[str, Tensor],
        targets: dict[str, Tensor] | None = None,
    ) -> Tensor:
        """Compute the embedding reconstruction loss.

        Args:
            outputs: Pipeline output dictionary containing the reconstructed
                embedding.
            targets: Ground-truth dictionary containing the original
                embedding.

        Returns:
            Scalar loss tensor.

        Raises:
            ValueError: If ``targets`` is ``None``.
        """
        if targets is None:
            raise ValueError("EmbeddingReconstructionLoss requires targets; received None.")

        reconstructed = outputs["game.reconstructed_embedding"]
        original = targets["game.original_embedding"]

        # Cosine similarity loss: mean(1 - cos_sim)
        cosine_sim = F.cosine_similarity(reconstructed, original, dim=-1)
        cosine_loss = (1.0 - cosine_sim).mean()

        # MSE loss
        mse_loss = F.mse_loss(reconstructed, original)

        return cosine_loss + self.mse_weight * mse_loss


@register("loss", "embedding_referential")
class EmbeddingReferentialLoss(LFMLoss):
    """Cross-entropy loss for the receiver choosing the target embedding.

    The receiver produces logits over candidate embeddings (one target plus
    several distractors).  This loss computes standard cross-entropy between
    those logits and the index of the true target.

    Expected output keys:
        - ``"game.receiver_logits"``: ``(batch, K)`` float logits.

    Expected target keys:
        - ``"game.target_idx"``: ``(batch,)`` int64 target index.

    Args:
        config: Optional loss configuration (unused).
        weight: Multiplicative weight for this loss term.
    """

    def __init__(self, config: object = None, weight: float = 1.0) -> None:
        super().__init__(config, weight)

    def forward(
        self,
        outputs: dict[str, Tensor],
        targets: dict[str, Tensor] | None = None,
    ) -> Tensor:
        """Compute the embedding referential loss.

        Args:
            outputs: Pipeline output dictionary containing the receiver's
                choice logits.
            targets: Ground-truth dictionary containing the target index.

        Returns:
            Scalar cross-entropy loss tensor.

        Raises:
            ValueError: If ``targets`` is ``None``.
        """
        if targets is None:
            raise ValueError("EmbeddingReferentialLoss requires targets; received None.")

        logits = outputs["game.receiver_logits"]  # (batch, K)
        target_idx = targets["game.target_idx"]  # (batch,)

        return F.cross_entropy(logits, target_idx)


def _infer_device(tensors: dict[str, Tensor]) -> torch.device:
    """Return the device of the first tensor in the dict, falling back to CPU."""
    for v in tensors.values():
        if isinstance(v, Tensor):
            return v.device
    return torch.device("cpu")

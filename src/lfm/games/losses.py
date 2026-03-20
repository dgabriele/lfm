"""Game-specific loss functions for LFM emergent communication games.

Provides two loss functions used in scene-based communication games:

``SceneReconstructionLoss`` combines per-attribute cross-entropy and
per-relation binary cross-entropy to measure how faithfully a receiver
can reconstruct the sender's scene from the transmitted message.

``ReferentialLoss`` measures whether a receiver can identify the target
scene among a set of distractors, using standard cross-entropy over
the receiver's choice logits.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F  # noqa: N812
from torch import Tensor

from lfm._registry import register
from lfm.core.loss import LFMLoss


@register("loss", "scene_reconstruction")
class SceneReconstructionLoss(LFMLoss):
    """Cross-entropy for attributes plus binary cross-entropy for relations.

    Computes per-attribute classification loss (cross-entropy between predicted
    logits and ground-truth attribute indices) summed over all attribute
    dimensions, plus a binary cross-entropy loss for pairwise relation
    predictions.  The total is the sum of attribute loss and relation loss,
    averaged over the batch.

    Expected output keys (written by the game):
        - ``"game.attr_logits.{d}"`` for each attribute dimension ``d``:
          ``(batch, N, C_d)`` float logits.
        - ``"game.relation_logits"``: ``(batch, N, N, R)`` float logits.

    Expected target keys:
        - ``"game.object_attrs"``: ``(batch, N, D)`` int64 attribute indices.
        - ``"game.relations"``: ``(batch, N, N, R)`` float32 relation labels.

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
        """Compute the scene reconstruction loss.

        Args:
            outputs: Combined pipeline output dictionary containing attribute
                logits and relation logits.
            targets: Ground-truth dictionary containing integer attribute
                indices and float relation labels.

        Returns:
            Scalar loss tensor (mean over batch).

        Raises:
            ValueError: If ``targets`` is ``None``.
        """
        if targets is None:
            raise ValueError("SceneReconstructionLoss requires targets; received None.")

        # --- Attribute loss ---
        # Discover how many attribute heads exist by scanning output keys.
        attr_loss = torch.tensor(0.0, device=_infer_device(outputs))
        d = 0
        while f"game.attr_logits.{d}" in outputs:
            # logits: (batch, N, C_d)
            logits = outputs[f"game.attr_logits.{d}"]
            # ground truth for this attribute: (batch, N) int64
            target = targets["game.object_attrs"][:, :, d]

            # Reshape for cross_entropy: (batch * N, C_d) vs (batch * N,)
            batch_n = logits.shape[0] * logits.shape[1]
            attr_loss = attr_loss + F.cross_entropy(
                logits.reshape(batch_n, -1),
                target.reshape(batch_n),
            )
            d += 1

        # --- Relation loss ---
        relation_logits = outputs["game.relation_logits"]  # (batch, N, N, R)
        relation_targets = targets["game.relations"]  # (batch, N, N, R)

        relation_loss = F.binary_cross_entropy_with_logits(relation_logits, relation_targets)

        return attr_loss + relation_loss


@register("loss", "referential_accuracy")
class ReferentialLoss(LFMLoss):
    """Cross-entropy loss for the receiver choosing the target scene.

    In the referential game the receiver produces a logit vector over the
    candidate scenes (one target plus several distractors).  This loss
    computes the standard cross-entropy between those logits and the
    index of the true target.

    Expected output keys:
        - ``"game.receiver_logits"``: ``(batch, K)`` float logits, where
          ``K`` is the number of candidate scenes.

    Expected target keys:
        - ``"game.target_idx"``: ``(batch,)`` int64 index of the target.

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
        """Compute the referential accuracy loss.

        Args:
            outputs: Combined pipeline output dictionary containing the
                receiver's choice logits.
            targets: Ground-truth dictionary containing the target index.

        Returns:
            Scalar cross-entropy loss tensor (mean over batch).

        Raises:
            ValueError: If ``targets`` is ``None``.
        """
        if targets is None:
            raise ValueError("ReferentialLoss requires targets; received None.")

        logits = outputs["game.receiver_logits"]  # (batch, K)
        target_idx = targets["game.target_idx"]  # (batch,)

        return F.cross_entropy(logits, target_idx)


def _infer_device(tensors: dict[str, Tensor]) -> torch.device:
    """Return the device of the first tensor in the dict, falling back to CPU."""
    for v in tensors.values():
        if isinstance(v, Tensor):
            return v.device
    return torch.device("cpu")

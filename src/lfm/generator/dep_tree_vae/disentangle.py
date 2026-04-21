"""Disentanglement losses for DepTreeVAE.

Ensures z_struct encodes syntactic structure and z_content encodes
lexical content, with minimal cross-contamination.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.autograd import Function

from lfm.generator.dep_tree_vae.config import (
    DisentanglementConfig,
    NUM_DEP_RELATIONS,
)


class _GradientReversal(Function):
    """Flip gradient sign during backward pass."""

    @staticmethod
    def forward(ctx: object, x: Tensor, scale: float) -> Tensor:
        ctx.scale = scale  # type: ignore[attr-defined]
        return x

    @staticmethod
    def backward(ctx: object, grad_output: Tensor) -> tuple[Tensor, None]:
        return -ctx.scale * grad_output, None  # type: ignore[attr-defined]


def gradient_reversal(x: Tensor, scale: float = 1.0) -> Tensor:
    """Apply gradient reversal layer."""
    return _GradientReversal.apply(x, scale)


def hsic(x: Tensor, y: Tensor) -> Tensor:
    """HSIC independence criterion with RBF kernels.

    Measures statistical dependence between x and y.
    Returns 0 when x and y are independent, positive otherwise.
    Uses the median heuristic for kernel bandwidth.

    Args:
        x: ``(B, D1)``
        y: ``(B, D2)``

    Returns:
        Scalar HSIC estimate.
    """
    n = x.size(0)
    if n < 4:
        return torch.tensor(0.0, device=x.device)

    # Pairwise squared distances
    dx = torch.cdist(x, x).pow(2)
    dy = torch.cdist(y, y).pow(2)

    # Median heuristic for bandwidth
    sigma_x = dx.median().clamp(min=1e-5)
    sigma_y = dy.median().clamp(min=1e-5)

    # RBF kernels
    kx = (-dx / (2 * sigma_x)).exp()
    ky = (-dy / (2 * sigma_y)).exp()

    # Centering matrix H = I - 1/n
    kx_c = kx - kx.mean(dim=0, keepdim=True) - kx.mean(dim=1, keepdim=True) + kx.mean()
    ky_c = ky - ky.mean(dim=0, keepdim=True) - ky.mean(dim=1, keepdim=True) + ky.mean()

    return (kx_c * ky_c).sum() / (n * n)


class DisentanglementModule(nn.Module):
    """Auxiliary heads + losses for struct/content disentanglement.

    Three components:
        1. **StructureClassifier**: z_struct → role sequence distribution.
           Supervised loss that z_struct should predict the dependency skeleton.
        2. **ContentPredictor**: z_content → bag-of-words over content tokens.
           Supervised loss that z_content should predict which words appear.
        3. **Adversarial**: z_content → role sequence (with gradient reversal).
           Forces z_content to NOT encode structural information.
    """

    def __init__(
        self,
        cfg: DisentanglementConfig,
        struct_dim: int,
        content_dim: int,
        content_vocab_size: int,
    ) -> None:
        super().__init__()
        self.cfg = cfg

        # z_struct → role sequence distribution (multi-label)
        self.struct_classifier = nn.Sequential(
            nn.Linear(struct_dim, struct_dim * 2),
            nn.GELU(),
            nn.Linear(struct_dim * 2, NUM_DEP_RELATIONS),
        )

        # z_content → bag-of-words (multi-label over content vocab)
        self.content_predictor = nn.Sequential(
            nn.Linear(content_dim, content_dim * 2),
            nn.GELU(),
            nn.Linear(content_dim * 2, content_vocab_size),
        )

        # z_content → structure (adversarial — gradient reversed)
        self.adversarial_head = nn.Sequential(
            nn.Linear(content_dim, content_dim),
            nn.GELU(),
            nn.Linear(content_dim, NUM_DEP_RELATIONS),
        )

    def forward(
        self,
        z_struct: Tensor,
        z_content: Tensor,
        role_bow: Tensor,
        content_bow: Tensor,
    ) -> dict[str, Tensor]:
        """Compute all disentanglement losses.

        Args:
            z_struct: ``(B, struct_dim)``
            z_content: ``(B, content_dim)``
            role_bow: ``(B, NUM_DEP_RELATIONS)`` multi-hot role presence.
            content_bow: ``(B, content_vocab_size)`` multi-hot word presence.

        Returns:
            Dict with keys ``struct_loss``, ``content_loss``,
            ``adversarial_loss``, and ``total`` (weighted sum).
        """
        cfg = self.cfg

        # 1. Structure classifier: z_struct should predict roles
        struct_logits = self.struct_classifier(z_struct)
        struct_loss = F.binary_cross_entropy_with_logits(
            struct_logits, role_bow.float(),
        )

        # 2. Content predictor: z_content should predict words
        content_logits = self.content_predictor(z_content)
        content_loss = F.binary_cross_entropy_with_logits(
            content_logits, content_bow.float(),
        )

        # 3. Adversarial: z_content should NOT predict structure
        if cfg.adversarial_weight > 0:
            z_content_rev = gradient_reversal(
                z_content, cfg.gradient_reversal_scale,
            )
            adv_logits = self.adversarial_head(z_content_rev)
            adv_loss = F.binary_cross_entropy_with_logits(
                adv_logits, role_bow.float(),
            )
        else:
            with torch.no_grad():
                adv_logits = self.adversarial_head(z_content)
                adv_loss = F.binary_cross_entropy_with_logits(
                    adv_logits, role_bow.float(),
                )

        # 4. HSIC: penalize statistical dependence between subspaces
        hsic_loss = (
            hsic(z_struct, z_content)
            if cfg.hsic_weight > 0
            else torch.tensor(0.0, device=z_struct.device)
        )

        total = (
            cfg.struct_cls_weight * struct_loss
            + cfg.content_bow_weight * content_loss
            + cfg.adversarial_weight * adv_loss
            + cfg.hsic_weight * hsic_loss
        ).clamp(min=-100, max=100)

        return {
            "struct_loss": struct_loss,
            "content_loss": content_loss,
            "adversarial_loss": adv_loss,
            "hsic_loss": hsic_loss,
            "total": total,
        }

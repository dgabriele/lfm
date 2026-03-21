"""Configuration for the surface phonology module.

Defines the ``PhonologyConfig`` used to parameterize implicit phonotactic
constraints via surface-form smoothness.  No explicit phonological categories
(vowels, consonants, sonority) are encoded — structure emerges from
communication pressure.
"""

from __future__ import annotations

from lfm.config.base import LFMBaseConfig


class PhonologyConfig(LFMBaseConfig):
    """Configuration for implicit surface phonology.

    Surface phonology maps token embeddings to continuous surface-form
    vectors and applies three implicit pressures — sequential smoothness,
    energy contour regularization, and batch diversity — to achieve
    phonotactic constraint without encoding any linguistic categories.

    Attributes:
        name: Registry name of the phonology implementation to use.
        surface_dim: Dimensionality of each surface vector (implicit
            articulatory/acoustic feature space).
        max_surface_len: Maximum number of surface vectors per token.
        smoothness_hidden_dim: Hidden size of the GRU that predicts
            sequential smoothness.
        smoothness_weight: Weight for the smoothness (pronounceability)
            loss term.
        energy_weight: Weight for the energy contour (SSP analog) loss.
        diversity_weight: Weight for the batch diversity (anti-collapse)
            loss.
        min_variance: Minimum batch variance floor for the diversity
            pressure.
        enrich: Whether to fold surface information back into embeddings.
        pretrained_smoothness_path: Path to pre-trained smoothness GRU
            checkpoint (``.pt`` file).  ``None`` means random init.
    """

    name: str = "surface"
    surface_dim: int = 12
    max_surface_len: int = 8
    smoothness_hidden_dim: int = 32
    smoothness_weight: float = 1.0
    energy_weight: float = 0.3
    diversity_weight: float = 0.5
    min_variance: float = 0.01
    enrich: bool = True
    pretrained_smoothness_path: str | None = None

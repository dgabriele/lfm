"""Shared configuration models for agent games."""

from __future__ import annotations

from lfm.config.base import LFMBaseConfig


class MessageEncoderConfig(LFMBaseConfig):
    """Configuration for the attention-based message encoder.

    Self-attention layers process the decoder's multi-scale hidden states,
    then a learned query cross-attention readout produces a fixed-size vector.
    """

    num_layers: int = 2
    num_heads: int = 8


class CurriculumConfig(LFMBaseConfig):
    """Configuration for curriculum learning over distractor difficulty.

    Linearly interpolates the fraction of hard (within-cluster) distractors
    from ``start_hard_ratio`` to ``end_hard_ratio`` over ``warmup_steps``.

    With stratified sampling (``medium_ratio > 0``), distractors are drawn
    from three tiers:
    - **Hard**: same cluster as anchor (fine-grained discrimination)
    - **Medium**: different cluster (inter-cluster contrasts)
    - **Easy**: uniform random (coarse-grained discrimination)

    This encourages a language flexible enough for both broad category
    markers and fine-grained instance-level distinctions.
    """

    enabled: bool = True
    warmup_steps: int = 500
    start_hard_ratio: float = 0.0
    end_hard_ratio: float = 1.0
    medium_ratio: float = 0.0  # fraction of distractors from different clusters

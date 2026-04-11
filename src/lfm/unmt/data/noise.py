"""Denoising autoencoder noise functions.

Three independent corruptions are applied to each training sequence,
following Lample et al. 2018 ("Phrase-Based & Neural Unsupervised
Machine Translation"):

1. **Word drop**: each token is independently dropped with probability
   ``drop_prob``.
2. **Word mask**: each token is independently replaced with the mask
   token with probability ``mask_prob``.
3. **Local shuffle**: token positions are permuted within a sliding
   window of size ``swap_window``.  Implemented by adding uniform noise
   to positional indices and re-sorting.

Noise is applied in-place on Python lists of token ids — these are
inner-loop operations during training, and avoiding tensor allocations
per-sample keeps dataloader workers fast.
"""

from __future__ import annotations

import random
from dataclasses import dataclass


@dataclass(frozen=True)
class NoiseConfig:
    """Parameters for DAE noise.

    Attributes:
        drop_prob: Per-token drop probability.
        mask_prob: Per-token mask probability.
        swap_window: Size of the sliding window for local shuffle.
            A value of ``1`` disables shuffling.
        mask_token_id: Vocabulary id of the mask token.
    """

    drop_prob: float
    mask_prob: float
    swap_window: int
    mask_token_id: int


def word_drop(tokens: list[int], prob: float, rng: random.Random) -> list[int]:
    """Drop each token independently with probability ``prob``.

    At least one token is always kept so the sequence is never empty —
    if every token would be dropped, one is retained at random.
    """
    if prob <= 0 or len(tokens) <= 1:
        return list(tokens)
    kept = [t for t in tokens if rng.random() >= prob]
    if not kept:
        kept = [rng.choice(tokens)]
    return kept


def word_mask(
    tokens: list[int],
    prob: float,
    mask_token_id: int,
    rng: random.Random,
) -> list[int]:
    """Replace each token with ``mask_token_id`` with probability ``prob``."""
    if prob <= 0:
        return list(tokens)
    return [
        mask_token_id if rng.random() < prob else t
        for t in tokens
    ]


def local_shuffle(
    tokens: list[int], window: int, rng: random.Random,
) -> list[int]:
    """Permute tokens locally via index jitter.

    Adds uniform noise in ``[0, window)`` to each token's position and
    re-sorts — this constrains every token to move at most ``window``
    positions, matching the noise operator used by Lample et al.
    """
    if window <= 1 or len(tokens) <= 1:
        return list(tokens)
    jittered = [(i + rng.uniform(0, window), t) for i, t in enumerate(tokens)]
    jittered.sort(key=lambda x: x[0])
    return [t for _, t in jittered]


def apply_noise(
    tokens: list[int], config: NoiseConfig, rng: random.Random,
) -> list[int]:
    """Apply all three noise operators in order: drop → mask → shuffle."""
    out = word_drop(tokens, config.drop_prob, rng)
    out = word_mask(out, config.mask_prob, config.mask_token_id, rng)
    out = local_shuffle(out, config.swap_window, rng)
    return out

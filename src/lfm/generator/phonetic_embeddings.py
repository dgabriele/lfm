"""Phonetically-structured token embeddings and label smoothing.

Leverages the natural topology of IPA: sounds that are articulatorily
similar (differing by one feature, like /p/ vs /b/) get similar
embeddings and lower substitution cost during training.

Uses PanPhon articulatory feature vectors (24-dim, ternary) as the
foundation.  Each BPE token's feature vector is the mean of its
constituent IPA segments' features.
"""

from __future__ import annotations

import logging
import unicodedata

import torch
import torch.nn.functional as F  # noqa: N812
from torch import Tensor, nn

logger = logging.getLogger(__name__)

# IPA combining characters and modifiers (not standalone segments)
_COMBINING_CATEGORIES = {"Mn", "Mc", "Me"}  # Mark nonspacing/spacing/enclosing
_SUPRASEGMENTALS = set("ːˈˌˑ̃̆̈")


def build_token_feature_matrix(
    sp: object,
    vocab_size: int,
    num_special: int = 2,
) -> Tensor:
    """Build a (full_vocab, 24) matrix of panphon features per BPE token.

    Each BPE token is decoded to its IPA text, segmented into characters,
    and each character's 24-dim articulatory feature vector is looked up
    via panphon.  The token's feature vector is the mean of its segments.

    Tokens with no valid IPA segments (specials, whitespace-only, etc.)
    get a zero vector.

    Args:
        sp: Sentencepiece processor.
        vocab_size: SPM vocab size (excludes BOS/EOS).
        num_special: Number of extra tokens (BOS, EOS) appended past vocab.

    Returns:
        Float tensor of shape ``(vocab_size + num_special, 24)``.
    """
    import panphon

    ft = panphon.FeatureTable()
    full_vocab = vocab_size + num_special
    n_features = len(ft.names)
    features = torch.zeros(full_vocab, n_features)

    mapped = 0
    for token_id in range(vocab_size):
        piece = sp.id_to_piece(token_id)  # type: ignore[union-attr]
        # Strip sentencepiece word-boundary marker
        piece = piece.replace("▁", "")
        if not piece:
            continue

        # Segment piece into individual IPA characters
        # (skip combining marks — they modify the preceding segment)
        seg_vectors: list[list[int]] = []
        for char in piece:
            cat = unicodedata.category(char)
            if cat in _COMBINING_CATEGORIES or char in _SUPRASEGMENTALS:
                continue
            if char.isspace():
                continue
            seg = ft.fts(char)
            if seg is not None and hasattr(seg, "numeric"):
                vec = seg.numeric()
                if vec and len(vec) == n_features:
                    seg_vectors.append(vec)

        if seg_vectors:
            features[token_id] = torch.tensor(seg_vectors, dtype=torch.float32).mean(dim=0)
            mapped += 1

    logger.info(
        "Phonetic features: %d/%d tokens mapped (%.1f%%)",
        mapped, vocab_size, 100 * mapped / max(vocab_size, 1),
    )
    return features


def init_embeddings_from_features(
    embedding: nn.Embedding,
    feature_matrix: Tensor,
    scale: float = 0.5,
) -> None:
    """Initialize token embeddings from phonetic feature vectors.

    Projects the 24-dim feature vectors to the embedding dimension via
    a random projection, then blends with the existing (random) init.
    This gives phonetically similar tokens similar starting embeddings
    while preserving enough randomness for the model to learn.

    Args:
        embedding: The nn.Embedding to initialize in-place.
        feature_matrix: ``(vocab_size, 24)`` from ``build_token_feature_matrix``.
        scale: Blend factor (0=fully random, 1=fully phonetic).
    """
    vocab_size, n_features = feature_matrix.shape
    embed_dim = embedding.embedding_dim

    # Random projection: 24 → embed_dim (on same device as embedding)
    device = embedding.weight.device
    proj = torch.randn(n_features, embed_dim, device=device) * (1.0 / n_features ** 0.5)
    phonetic_embeds = feature_matrix.to(device) @ proj  # (vocab, embed_dim)

    # Normalize both to same scale before blending
    ph_std = phonetic_embeds.std()
    rand_std = embedding.weight.data.std()
    if ph_std > 1e-6:
        phonetic_embeds = phonetic_embeds * (rand_std / ph_std)

    with torch.no_grad():
        # Only blend tokens that have features (non-zero rows)
        has_features = feature_matrix.to(device).abs().sum(dim=-1) > 0
        n = min(vocab_size, len(has_features))
        blended = (
            scale * phonetic_embeds + (1 - scale) * embedding.weight.data
        )
        mask = has_features[:n]
        embedding.weight.data[:n][mask] = blended[:n][mask]

    logger.info(
        "Initialized %d token embeddings from phonetic features (scale=%.2f)",
        has_features.sum().item(), scale,
    )


def build_phonetic_similarity_matrix(
    feature_matrix: Tensor,
    temperature: float = 2.0,
) -> Tensor:
    """Build a soft label distribution based on phonetic similarity.

    For each token, computes cosine similarity to all other tokens in
    the articulatory feature space, then applies softmax with temperature
    to create a smooth target distribution.

    Used for phonetic label smoothing: instead of one-hot targets, the
    loss function sees a distribution that gives partial credit for
    phonetically similar predictions.

    Args:
        feature_matrix: ``(vocab_size, 24)`` phonetic features.
        temperature: Softmax temperature (higher = smoother).

    Returns:
        ``(vocab_size, vocab_size)`` row-normalized similarity matrix.
    """
    # Normalize features
    norms = feature_matrix.norm(dim=-1, keepdim=True).clamp(min=1e-6)
    normalized = feature_matrix / norms

    # Cosine similarity matrix
    sim = normalized @ normalized.T  # (V, V)

    # Tokens with no features (zero vectors) get uniform zero similarity
    has_features = feature_matrix.abs().sum(dim=-1) > 0
    no_features = ~has_features

    # Softmax with temperature → soft targets
    sim = sim / temperature
    # Mask out no-feature tokens from being targets
    sim[:, no_features] = float("-inf")
    # Self-similarity should be highest
    soft_targets = F.softmax(sim, dim=-1)

    # Tokens without features get one-hot (identity row)
    soft_targets[no_features] = 0.0
    soft_targets[no_features, no_features] = 1.0 / max(no_features.sum().item(), 1)

    return soft_targets


def phonetic_ce_loss(
    logits: Tensor,
    targets: Tensor,
    similarity_matrix: Tensor,
    smoothing: float = 0.1,
) -> Tensor:
    """Cross-entropy with phonetic label smoothing.

    Blends one-hot targets with a phonetic similarity distribution:
    ``target_dist = (1 - α) * one_hot + α * phonetic_similarity[target]``

    A prediction of /b/ when the target is /p/ incurs less loss than
    predicting /ʒ/, because /b/ is phonetically closer to /p/.

    Args:
        logits: ``(N, vocab_size)`` raw logits from the decoder.
        targets: ``(N,)`` target token IDs.
        similarity_matrix: ``(vocab_size, vocab_size)`` from
            ``build_phonetic_similarity_matrix``.
        smoothing: Blend factor (0 = standard CE, 1 = fully phonetic).

    Returns:
        Scalar loss (mean over batch).
    """
    # Standard log probabilities
    log_probs = F.log_softmax(logits, dim=-1)

    # One-hot targets
    one_hot = torch.zeros_like(log_probs).scatter_(1, targets.unsqueeze(1), 1.0)

    # Phonetic soft targets — look up each target's similarity row
    soft_targets = similarity_matrix[targets]  # (N, V)

    # Blend
    blended = (1 - smoothing) * one_hot + smoothing * soft_targets

    # KL divergence (equivalent to CE with soft targets)
    loss = -(blended * log_probs).sum(dim=-1).mean()
    return loss

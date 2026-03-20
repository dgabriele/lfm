"""Embedding-domain game modules for LFM training.

Unlike the scene-based games in ``lfm.games``, these games operate directly
on precomputed dense embedding vectors.  No scene encoder is needed because
the embedding *is* the agent state.

``EmbeddingReconstructionGame``:  Embedding -> LFM -> reconstruct embedding.
``EmbeddingReferentialGame``:  Sender embedding -> LFM -> message -> receiver
scores candidates via dot-product similarity.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from torch import Tensor, nn

from lfm.games.encoder import MessagePooler

if TYPE_CHECKING:
    from lfm.embeddings.config import EmbeddingGameConfig


class EmbeddingReconstructionGame(nn.Module):
    """Reconstruction game: embedding -> LFM -> reconstructed embedding.

    The input embedding is used directly as the agent state (no encoder).
    After the LFM faculty processes it, a ``MessagePooler`` extracts a
    fixed-size vector from the faculty outputs and an MLP projects it
    back to the original embedding dimensionality.

    Args:
        config: Embedding game configuration specifying embedding
            dimensionality and network architecture.
    """

    def __init__(self, config: EmbeddingGameConfig) -> None:
        super().__init__()
        self.config = config
        dim = config.embedding_dim

        # Pool LFM outputs into a fixed-size representation
        self.pooler = MessagePooler(target_dim=dim)

        # Project back to embedding space for reconstruction
        self.projection = nn.Sequential(
            nn.Linear(dim, dim * 2),
            nn.ReLU(),
            nn.Linear(dim * 2, dim),
        )

    def decode_message(self, lfm_outputs: dict[str, Tensor]) -> Tensor:
        """Pool LFM outputs and project back to embedding space.

        Args:
            lfm_outputs: Dictionary of named tensors from the
                ``LanguageFaculty`` forward pass.

        Returns:
            Reconstructed embedding tensor of shape ``(batch, embedding_dim)``.
        """
        pooled = self.pooler(lfm_outputs)
        return self.projection(pooled)

    @property
    def device(self) -> torch.device:
        """Infer device from model parameters."""
        p = next(self.parameters(), None)
        return p.device if p is not None else torch.device("cpu")


class EmbeddingReferentialGame(nn.Module):
    """Referential game: sender embeds via LFM, receiver scores candidates.

    The sender's input embedding is the agent state.  After the LFM faculty
    processes it, a ``MessagePooler`` extracts a message vector and a learned
    projection maps it to a scoring space.  The receiver computes dot-product
    similarity between the projected message and each candidate embedding.

    Args:
        config: Embedding game configuration specifying embedding
            dimensionality, number of distractors, etc.
    """

    def __init__(self, config: EmbeddingGameConfig) -> None:
        super().__init__()
        self.config = config
        dim = config.embedding_dim

        # Pool LFM outputs into a fixed-size message representation
        self.pooler = MessagePooler(target_dim=dim)

        # Learned projection from message space to scoring space
        self.message_projection = nn.Sequential(
            nn.Linear(dim, dim),
            nn.ReLU(),
            nn.Linear(dim, dim),
        )

    def score_candidates(
        self,
        lfm_outputs: dict[str, Tensor],
        candidates: Tensor,
    ) -> dict[str, Tensor]:
        """Score candidate embeddings against the LFM message.

        Args:
            lfm_outputs: Dictionary of named tensors from the
                ``LanguageFaculty`` forward pass.
            candidates: Candidate embedding tensor of shape
                ``(batch, K, embedding_dim)`` where ``K`` is the number
                of candidates (target + distractors).

        Returns:
            Dictionary with ``"receiver_logits"`` of shape ``(batch, K)``.
        """
        pooled = self.pooler(lfm_outputs)  # (B, dim)
        projected = self.message_projection(pooled)  # (B, dim)

        # Dot-product scoring: (B, 1, dim) @ (B, dim, K) -> (B, 1, K) -> (B, K)
        logits = torch.bmm(
            projected.unsqueeze(1),
            candidates.transpose(1, 2),
        ).squeeze(1)

        return {"receiver_logits": logits}

    @property
    def device(self) -> torch.device:
        """Infer device from model parameters."""
        p = next(self.parameters(), None)
        return p.device if p is not None else torch.device("cpu")

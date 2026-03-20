"""Referential game module for LFM emergent communication.

Lewis signaling game: a sender encodes a target scene, passes it through
the LanguageFaculty, and a receiver must identify the target among a set
of distractors based on the transmitted message.  Communication success
drives both the faculty and the game modules to develop informative,
discriminative language.

The game owns a sender encoder, a receiver module (with its own encoder),
and a message pooler.  The LanguageFaculty is external.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from torch import Tensor, nn

from lfm.games.config import ReferentialGameConfig
from lfm.games.encoder import MessagePooler, SceneEncoder

if TYPE_CHECKING:
    pass


class ReceiverModule(nn.Module):
    """Scores candidate scenes against a received message.

    The receiver has its own scene encoder (separate from the sender) that
    encodes each candidate scene independently.  Scoring is performed via
    dot-product similarity between the message representation and each
    candidate's encoding.

    Args:
        config: Referential game configuration.
    """

    def __init__(self, config: ReferentialGameConfig) -> None:
        super().__init__()
        self.config = config

        # Receiver's own encoder for candidate scenes
        self.encoder = SceneEncoder(config.scene, config.encoder)

        encoder_out_dim = config.encoder.output_dim

        # Project message to encoder output space for dot-product scoring
        self.message_proj = nn.Linear(config.receiver_hidden_dim, encoder_out_dim)

    def forward(
        self,
        message_repr: Tensor,
        candidate_attrs: Tensor,
        candidate_relations: Tensor,
    ) -> Tensor:
        """Score candidate scenes against the received message.

        Args:
            message_repr: Pooled message vector of shape ``(batch, message_dim)``.
            candidate_attrs: Integer attribute indices of shape
                ``(batch, K, N, D)`` where ``K`` is the number of candidates.
            candidate_relations: Relation indicators of shape
                ``(batch, K, N, N, R)``.

        Returns:
            Logits tensor of shape ``(batch, K)`` scoring each candidate.
        """
        batch, num_candidates = candidate_attrs.shape[:2]

        # Flatten candidates into a single batch dimension for encoding
        # (batch * K, N, D) and (batch * K, N, N, R)
        flat_attrs = candidate_attrs.reshape(batch * num_candidates, *candidate_attrs.shape[2:])
        flat_rels = candidate_relations.reshape(
            batch * num_candidates, *candidate_relations.shape[2:]
        )

        # Encode all candidates: (batch * K, encoder_out_dim)
        flat_encoded = self.encoder(flat_attrs, flat_rels)

        # Reshape back: (batch, K, encoder_out_dim)
        candidate_encodings = flat_encoded.view(batch, num_candidates, -1)

        # Project message to matching space: (batch, encoder_out_dim)
        message_projected = self.message_proj(message_repr)

        # Dot-product scoring: (batch, K)
        # (batch, 1, encoder_out_dim) @ (batch, encoder_out_dim, K) -> (batch, 1, K) -> (batch, K)
        logits = torch.bmm(
            message_projected.unsqueeze(1),
            candidate_encodings.transpose(1, 2),
        ).squeeze(1)

        return logits


class ReferentialGame(nn.Module):
    """Lewis signaling game: sender describes target, receiver identifies it.

    The sender uses a ``SceneEncoder`` to produce an agent-state vector from
    the target scene.  After the LanguageFaculty processes this state, the
    receiver scores each candidate scene against the transmitted message.

    Args:
        config: Referential game configuration.
    """

    def __init__(self, config: ReferentialGameConfig) -> None:
        super().__init__()
        self.config = config

        # Sender encoder: target scene -> agent-state
        self.sender_encoder = SceneEncoder(config.scene, config.encoder)

        # Receiver module (with its own encoder): scores candidates
        self.receiver = ReceiverModule(config)

        # Message pooler: LFM outputs -> fixed-size vector for the receiver
        self.pooler = MessagePooler(target_dim=config.receiver_hidden_dim)

    def encode_target(self, scene: dict[str, Tensor]) -> Tensor:
        """Encode the target scene to an agent-state vector.

        Args:
            scene: Dictionary with ``"object_attrs"`` of shape
                ``(batch, N, D)`` and ``"relations"`` of shape
                ``(batch, N, N, R)``.

        Returns:
            Agent-state tensor of shape ``(batch, output_dim)``.
        """
        return self.sender_encoder(scene["object_attrs"], scene["relations"])

    def score_candidates(
        self,
        lfm_outputs: dict[str, Tensor],
        candidate_attrs: Tensor,
        candidate_relations: Tensor,
    ) -> dict[str, Tensor]:
        """Score candidate scenes given the LFM message.

        Pools the LFM output into a fixed-size message representation and
        uses the receiver module to score each candidate.

        Args:
            lfm_outputs: Dictionary of named tensors from the
                ``LanguageFaculty`` forward pass.
            candidate_attrs: Integer attribute indices of shape
                ``(batch, K, N, D)`` where ``K`` is the number of candidates
                (target + distractors).
            candidate_relations: Relation indicators of shape
                ``(batch, K, N, N, R)``.

        Returns:
            Dictionary with ``"receiver_logits"`` of shape ``(batch, K)``.
        """
        message_repr = self.pooler(lfm_outputs)
        logits = self.receiver(message_repr, candidate_attrs, candidate_relations)
        return {"receiver_logits": logits}

    @property
    def device(self) -> torch.device:
        """Infer the device from model parameters."""
        p = next(self.parameters(), None)
        return p.device if p is not None else torch.device("cpu")

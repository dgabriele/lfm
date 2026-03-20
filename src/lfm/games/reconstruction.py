"""Reconstruction game module for LFM emergent communication.

Self-play autoencoder game: a scene is encoded into an agent state, passed
through the LanguageFaculty to produce a message, then decoded back into a
reconstructed scene.  The reconstruction error drives the faculty to transmit
scene-relevant information through its emergent language.

The game owns its own encoder and decoder networks.  The LanguageFaculty
(quantizer, phonology, morphology, syntax, sentence, channel) is external
and supplied at call time by the training phase.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from torch import Tensor, nn

from lfm.games.config import ReconstructionGameConfig
from lfm.games.encoder import MessagePooler, SceneDecoder, SceneEncoder

if TYPE_CHECKING:
    pass


class ReconstructionGame(nn.Module):
    """Self-play autoencoder: scene -> encoder -> LFM -> decoder -> reconstructed scene.

    Owns the encoder and decoder.  The LanguageFaculty is external and called
    by the training phase between ``encode_scene`` and ``decode_message``.

    Args:
        config: Reconstruction game configuration controlling scene structure,
            encoder/decoder architecture, and training parameters.
    """

    def __init__(self, config: ReconstructionGameConfig) -> None:
        super().__init__()
        self.config = config

        # Scene encoder: structured scene -> flat agent-state vector
        self.encoder = SceneEncoder(config.scene, config.encoder)

        # Scene decoder: pooled message vector -> scene predictions
        # The decoder's input_dim equals its hidden_dim by convention; the
        # MessagePooler projects the LFM output to match this.
        decoder_input_dim = config.decoder.hidden_dim
        self.decoder = SceneDecoder(config.scene, config.decoder, input_dim=decoder_input_dim)

        # Message pooler: extracts a fixed-size vector from LFM outputs and
        # projects it to the decoder's expected input dimension.
        self.pooler = MessagePooler(target_dim=decoder_input_dim)

    def encode_scene(self, scene: dict[str, Tensor]) -> Tensor:
        """Encode a structured scene to an agent-state vector.

        Args:
            scene: Dictionary with ``"object_attrs"`` of shape
                ``(batch, N, D)`` and ``"relations"`` of shape
                ``(batch, N, N, R)``.

        Returns:
            Agent-state tensor of shape ``(batch, output_dim)``.
        """
        return self.encoder(scene["object_attrs"], scene["relations"])

    def decode_message(self, lfm_outputs: dict[str, Tensor]) -> dict[str, Tensor]:
        """Decode LFM outputs back to scene predictions.

        Pools the LFM output dictionary into a fixed-size vector and passes
        it through the scene decoder to produce per-attribute logits and
        relation logits.

        Args:
            lfm_outputs: Dictionary of named tensors from the
                ``LanguageFaculty`` forward pass.

        Returns:
            Dictionary with ``"attr_logits"`` (list of tensors, one per
            attribute dimension) and ``"relation_logits"`` tensor.
        """
        message_repr = self.pooler(lfm_outputs)
        return self.decoder(message_repr)

    @property
    def device(self) -> torch.device:
        """Infer the device from model parameters."""
        p = next(self.parameters(), None)
        return p.device if p is not None else torch.device("cpu")

"""Scene encoder, decoder, and message pooler modules for LFM games.

Provides neural network components that convert between structured scene
representations (object attributes and pairwise relations) and flat vector
representations suitable for the LFM language faculty pipeline.

``SceneEncoder`` maps a structured scene to a flat agent-state vector.
``SceneDecoder`` maps a pooled message vector back to per-attribute logits
and relation logits.  ``MessagePooler`` extracts a fixed-size vector from
the dictionary output of a ``LanguageFaculty`` forward pass.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F  # noqa: N812
from torch import Tensor, nn

from lfm.games.config import SceneConfig, SceneDecoderConfig, SceneEncoderConfig


def _build_mlp(
    input_dim: int,
    hidden_dim: int,
    output_dim: int,
    num_layers: int,
    dropout: float,
) -> nn.Sequential:
    """Build a feedforward MLP with ReLU activations and dropout.

    Architecture: ``input -> [hidden + ReLU + Dropout] * num_layers -> output``.

    Args:
        input_dim: Dimensionality of the input features.
        hidden_dim: Width of each hidden layer.
        output_dim: Dimensionality of the output features.
        num_layers: Number of hidden layers (each consisting of Linear, ReLU,
            and Dropout).  Must be >= 1.
        dropout: Dropout probability applied after each hidden layer.

    Returns:
        An ``nn.Sequential`` module implementing the MLP.
    """
    layers: list[nn.Module] = []
    prev_dim = input_dim
    for _ in range(num_layers):
        layers.extend(
            [
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            ]
        )
        prev_dim = hidden_dim
    layers.append(nn.Linear(prev_dim, output_dim))
    return nn.Sequential(*layers)


class SceneEncoder(nn.Module):
    """Encodes structured scene tensors into a flat agent-state vector.

    Architecture: one-hot encode each attribute, concatenate all objects,
    flatten relations, concatenate everything, then project through an MLP to
    produce the agent-state vector.

    Args:
        scene_config: Configuration describing the scene structure (number of
            objects, attribute cardinalities, relation types).
        encoder_config: Configuration for the encoder network (hidden dim,
            number of layers, output dim, dropout).
    """

    def __init__(self, scene_config: SceneConfig, encoder_config: SceneEncoderConfig) -> None:
        super().__init__()
        self.scene_config = scene_config
        self.encoder_config = encoder_config

        cardinalities = scene_config.attribute_cardinalities
        num_objects = scene_config.num_objects
        num_relation_types = scene_config.num_relation_types

        # One-hot per attribute per object, then flattened across objects
        self._cardinalities = cardinalities
        onehot_dim = sum(cardinalities) * num_objects

        # Flattened relation tensor: N * N * R
        relation_dim = num_objects * num_objects * num_relation_types

        input_dim = onehot_dim + relation_dim

        self.mlp = _build_mlp(
            input_dim=input_dim,
            hidden_dim=encoder_config.hidden_dim,
            output_dim=encoder_config.output_dim,
            num_layers=encoder_config.num_layers,
            dropout=encoder_config.dropout,
        )

    def forward(self, object_attrs: Tensor, relations: Tensor) -> Tensor:
        """Encode a scene into a flat agent-state vector.

        Args:
            object_attrs: Integer attribute indices of shape
                ``(batch, N, D)`` with dtype ``int64``, where ``N`` is the
                number of objects and ``D`` is the number of attribute
                dimensions.
            relations: Pairwise relation indicators of shape
                ``(batch, N, N, R)`` with dtype ``float32``, where ``R`` is
                the number of relation types.

        Returns:
            Agent-state tensor of shape ``(batch, output_dim)``.
        """
        batch = object_attrs.shape[0]

        # One-hot encode each attribute dimension and concatenate
        onehot_parts: list[Tensor] = []
        for d, card in enumerate(self._cardinalities):
            # (batch, N, card)
            onehot_parts.append(F.one_hot(object_attrs[:, :, d], num_classes=card).float())

        # (batch, N, sum(cardinalities))
        onehot_cat = torch.cat(onehot_parts, dim=-1)

        # Flatten N dimension: (batch, N * sum(cardinalities))
        onehot_flat = onehot_cat.reshape(batch, -1)

        # Flatten relations: (batch, N * N * R)
        relations_flat = relations.reshape(batch, -1)

        # Concatenate and project through MLP
        combined = torch.cat([onehot_flat, relations_flat], dim=-1)
        return self.mlp(combined)


class SceneDecoder(nn.Module):
    """Decodes a message representation back to scene predictions.

    Uses a shared trunk MLP followed by separate output heads: one linear
    head per attribute dimension (producing per-object classification logits)
    and one linear head for pairwise relation logits.

    Args:
        scene_config: Configuration describing the scene structure.
        decoder_config: Configuration for the decoder network.
        input_dim: Dimensionality of the incoming pooled message vector.
    """

    def __init__(
        self,
        scene_config: SceneConfig,
        decoder_config: SceneDecoderConfig,
        input_dim: int,
    ) -> None:
        super().__init__()
        self.scene_config = scene_config
        self.decoder_config = decoder_config

        num_objects = scene_config.num_objects
        cardinalities = scene_config.attribute_cardinalities
        num_relation_types = scene_config.num_relation_types

        self._num_objects = num_objects
        self._cardinalities = cardinalities
        self._num_relation_types = num_relation_types

        # Shared trunk: input_dim -> hidden_dim (num_layers hidden blocks)
        self.trunk = _build_mlp(
            input_dim=input_dim,
            hidden_dim=decoder_config.hidden_dim,
            output_dim=decoder_config.hidden_dim,
            num_layers=decoder_config.num_layers,
            dropout=decoder_config.dropout,
        )

        # One head per attribute dimension: hidden_dim -> N * C_d
        self.attr_heads = nn.ModuleList(
            [nn.Linear(decoder_config.hidden_dim, num_objects * card) for card in cardinalities]
        )

        # Relation head: hidden_dim -> N * N * R
        self.relation_head = nn.Linear(
            decoder_config.hidden_dim,
            num_objects * num_objects * num_relation_types,
        )

    def forward(self, message_repr: Tensor) -> dict[str, Tensor]:
        """Decode a message representation into scene predictions.

        Args:
            message_repr: Pooled message vector of shape ``(batch, input_dim)``.

        Returns:
            Dictionary with keys:

            - ``"attr_logits"``: list of tensors of shape ``(batch, N, C_d)``,
              one per attribute dimension, containing classification logits.
            - ``"relation_logits"``: tensor of shape ``(batch, N, N, R)``
              containing binary relation logits.
        """
        batch = message_repr.shape[0]

        hidden = self.trunk(message_repr)

        attr_logits: list[Tensor] = []
        for d, head in enumerate(self.attr_heads):
            # (batch, N * C_d) -> (batch, N, C_d)
            logits = head(hidden).view(batch, self._num_objects, self._cardinalities[d])
            attr_logits.append(logits)

        # (batch, N * N * R) -> (batch, N, N, R)
        relation_logits = self.relation_head(hidden).view(
            batch, self._num_objects, self._num_objects, self._num_relation_types
        )

        return {
            "attr_logits": attr_logits,
            "relation_logits": relation_logits,
        }


class MessagePooler(nn.Module):
    """Extracts a fixed-size vector from ``LanguageFaculty`` outputs.

    Searches the output dictionary for a suitable sequence tensor and
    mean-pools over the sequence dimension to produce a single vector.
    If the pooled dimension does not match ``target_dim``, a learned linear
    projection is lazily created on the first forward pass.

    Lookup order:

    1. ``"channel.message"`` -- the transmitted message.
    2. ``"quantization.embeddings"`` -- pre-channel quantized embeddings.
    3. Falls back to any key ending in ``".embeddings"``.

    Args:
        target_dim: Desired output dimensionality.  A learned projection is
            applied when the source tensor's feature dimension differs.
    """

    def __init__(self, target_dim: int) -> None:
        super().__init__()
        self.target_dim = target_dim
        # Projection is lazily initialised on first forward since we do not
        # know the source dimensionality until we see the data.
        self.projection: nn.Linear | None = None

    def _find_tensor(self, lfm_outputs: dict[str, Tensor]) -> Tensor:
        """Locate the best tensor to pool from the output dictionary.

        Args:
            lfm_outputs: Dictionary of named tensors from the language
                faculty forward pass.

        Returns:
            A tensor of shape ``(batch, seq_len, dim)`` or
            ``(batch, seq_len)`` suitable for mean-pooling.

        Raises:
            KeyError: If no suitable tensor is found.
        """
        # Priority 1: channel message
        if "channel.message" in lfm_outputs:
            return lfm_outputs["channel.message"]

        # Priority 2: quantized embeddings
        if "quantization.embeddings" in lfm_outputs:
            return lfm_outputs["quantization.embeddings"]

        # Priority 3: any key ending in ".embeddings"
        for key in lfm_outputs:
            if key.endswith(".embeddings"):
                return lfm_outputs[key]

        raise KeyError(
            "MessagePooler could not find a suitable tensor in lfm_outputs. "
            f"Available keys: {sorted(lfm_outputs.keys())}"
        )

    def forward(self, lfm_outputs: dict[str, Tensor]) -> Tensor:
        """Pool and project language faculty outputs to a fixed-size vector.

        Args:
            lfm_outputs: Dictionary of named tensors from the language
                faculty forward pass.

        Returns:
            Tensor of shape ``(batch, target_dim)``.
        """
        tensor = self._find_tensor(lfm_outputs)

        # If 2-D (batch, seq_len), unsqueeze to (batch, seq_len, 1) so
        # mean-pooling produces (batch, 1).
        if tensor.ndim == 2:
            tensor = tensor.unsqueeze(-1)

        # Mean-pool over the sequence dimension (dim=1)
        pooled = tensor.mean(dim=1)  # (batch, dim)

        source_dim = pooled.shape[-1]

        # Lazily create the projection layer if dimensions don't match
        if source_dim != self.target_dim:
            if self.projection is None:
                self.projection = nn.Linear(source_dim, self.target_dim).to(pooled.device)
            pooled = self.projection(pooled)

        return pooled

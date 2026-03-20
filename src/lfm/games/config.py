"""Configuration models for LFM games.

Defines Pydantic configuration classes for scene generation, scene encoding /
decoding, and the reconstruction and referential game variants.
"""

from __future__ import annotations

from pydantic import model_validator

from lfm.config.base import LFMBaseConfig


class SceneConfig(LFMBaseConfig):
    """Configuration for procedural scene generation.

    Attributes:
        num_objects: Number of objects in each generated scene.
        num_attributes: Number of attribute dimensions per object (e.g. shape,
            color, size, material).
        attribute_cardinalities: Number of distinct values for each attribute
            dimension.  Must have length equal to ``num_attributes``.
        num_relation_types: Number of binary relation types between object pairs
            (e.g. left-of, behind, above).
        relation_density: Probability that any given relation is active between
            an ordered pair of distinct objects.
        symmetric_relations: When ``True``, relations are forced to be symmetric
            by copying the upper triangle of the adjacency matrix to the lower
            triangle.
    """

    num_objects: int = 3
    num_attributes: int = 4
    attribute_cardinalities: tuple[int, ...] = (5, 8, 3, 4)
    num_relation_types: int = 3
    relation_density: float = 0.3
    symmetric_relations: bool = False

    @model_validator(mode="after")
    def _check_cardinalities_length(self) -> SceneConfig:
        if len(self.attribute_cardinalities) != self.num_attributes:
            raise ValueError(
                f"len(attribute_cardinalities) = {len(self.attribute_cardinalities)} "
                f"does not match num_attributes = {self.num_attributes}"
            )
        return self


class SceneEncoderConfig(LFMBaseConfig):
    """Configuration for the scene encoder network.

    Attributes:
        hidden_dim: Width of hidden layers in the encoder MLP.
        num_layers: Number of hidden layers.
        output_dim: Dimensionality of the encoder output.  Should match
            ``faculty.dim`` or ``quantizer.input_dim`` for downstream
            compatibility.
        dropout: Dropout probability applied after each hidden layer.
    """

    hidden_dim: int = 256
    num_layers: int = 2
    output_dim: int = 256
    dropout: float = 0.1


class SceneDecoderConfig(LFMBaseConfig):
    """Configuration for the scene decoder network.

    Attributes:
        hidden_dim: Width of hidden layers in the decoder MLP.
        num_layers: Number of hidden layers.
        dropout: Dropout probability applied after each hidden layer.
    """

    hidden_dim: int = 256
    num_layers: int = 2
    dropout: float = 0.1


class ReconstructionGameConfig(LFMBaseConfig):
    """Configuration for the reconstruction game.

    In the reconstruction game an encoder maps a scene to a latent message and
    a decoder reconstructs the original scene from that message.

    Attributes:
        scene: Scene generation parameters.
        encoder: Scene encoder parameters.
        decoder: Scene decoder parameters.
        batch_size: Number of scenes per training batch.
        reconstruction_weight: Scalar weight for the reconstruction loss.
    """

    scene: SceneConfig = SceneConfig()
    encoder: SceneEncoderConfig = SceneEncoderConfig()
    decoder: SceneDecoderConfig = SceneDecoderConfig()
    batch_size: int = 64
    reconstruction_weight: float = 1.0


class ReferentialGameConfig(LFMBaseConfig):
    """Configuration for the referential game.

    In the referential game a sender encodes a target scene and a receiver must
    identify it among a set of distractors.

    Attributes:
        scene: Scene generation parameters.
        encoder: Scene encoder parameters.
        num_distractors: Number of distractor scenes presented alongside the
            target scene.
        batch_size: Number of game episodes per training batch.
        receiver_hidden_dim: Width of hidden layers in the receiver network.
        referential_weight: Scalar weight for the referential loss.
    """

    scene: SceneConfig = SceneConfig()
    encoder: SceneEncoderConfig = SceneEncoderConfig()
    num_distractors: int = 3
    batch_size: int = 64
    receiver_hidden_dim: int = 256
    referential_weight: float = 1.0

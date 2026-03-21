"""Tests for the games subpackage."""

from __future__ import annotations

import pytest
import torch

from lfm.games.config import (
    ReconstructionGameConfig,
    ReferentialGameConfig,
    SceneConfig,
    SceneDecoderConfig,
    SceneEncoderConfig,
)
from lfm.games.encoder import MessagePooler, SceneDecoder, SceneEncoder
from lfm.games.reconstruction import ReconstructionGame
from lfm.games.referential import ReferentialGame
from lfm.games.scenes import SceneGenerator

# -- Scene generation --


def test_scene_config_validation():
    """attribute_cardinalities length must match num_attributes."""
    with pytest.raises(Exception):
        SceneConfig(num_attributes=3, attribute_cardinalities=(5, 8))  # mismatch


def test_scene_generation_shapes():
    """SceneGenerator produces correctly shaped tensors."""
    cfg = SceneConfig(num_objects=3, num_attributes=4, attribute_cardinalities=(5, 8, 3, 4))
    gen = SceneGenerator(cfg, device="cpu")
    scene = gen.generate(16)

    assert scene["object_attrs"].shape == (16, 3, 4)
    assert scene["object_attrs"].dtype == torch.int64
    assert scene["relations"].shape == (16, 3, 3, 3)
    assert scene["relations"].dtype == torch.float32


def test_scene_attribute_ranges():
    """Attribute values are within their cardinality ranges."""
    cfg = SceneConfig(num_objects=2, num_attributes=3, attribute_cardinalities=(5, 8, 3))
    gen = SceneGenerator(cfg, device="cpu")
    scene = gen.generate(100)

    for d, card in enumerate(cfg.attribute_cardinalities):
        vals = scene["object_attrs"][:, :, d]
        assert vals.min() >= 0
        assert vals.max() < card


def test_scene_no_self_relations():
    """Diagonal of relations tensor should be zero."""
    cfg = SceneConfig(num_objects=4, num_relation_types=3, relation_density=1.0)
    gen = SceneGenerator(cfg, device="cpu")
    scene = gen.generate(32)

    for i in range(cfg.num_objects):
        assert (scene["relations"][:, i, i, :] == 0).all()


def test_distractor_generation_shapes():
    """generate_distractors produces correctly shaped tensors."""
    cfg = SceneConfig(num_objects=2)
    gen = SceneGenerator(cfg, device="cpu")
    dist = gen.generate_distractors(8, 5)

    assert dist["object_attrs"].shape == (8, 5, 2, 4)
    assert dist["relations"].shape == (8, 5, 2, 2, 3)


# -- Encoder / Decoder --


def test_encoder_output_shape():
    """SceneEncoder produces the correct output dimension."""
    scene_cfg = SceneConfig(num_objects=3)
    enc_cfg = SceneEncoderConfig(output_dim=128)
    encoder = SceneEncoder(scene_cfg, enc_cfg)

    gen = SceneGenerator(scene_cfg, device="cpu")
    scene = gen.generate(8)
    out = encoder(scene["object_attrs"], scene["relations"])
    assert out.shape == (8, 128)


def test_decoder_output_shapes():
    """SceneDecoder produces correctly shaped attribute and relation logits."""
    scene_cfg = SceneConfig(num_objects=3, num_attributes=4, attribute_cardinalities=(5, 8, 3, 4))
    dec_cfg = SceneDecoderConfig()
    decoder = SceneDecoder(scene_cfg, dec_cfg, input_dim=128)

    fake_message = torch.randn(8, 128)
    out = decoder(fake_message)

    assert len(out["attr_logits"]) == 4
    assert out["attr_logits"][0].shape == (8, 3, 5)
    assert out["attr_logits"][1].shape == (8, 3, 8)
    assert out["attr_logits"][2].shape == (8, 3, 3)
    assert out["attr_logits"][3].shape == (8, 3, 4)
    assert out["relation_logits"].shape == (8, 3, 3, 3)


def test_message_pooler():
    """MessagePooler extracts and pools from LFM output dict."""
    pooler = MessagePooler(target_dim=64)

    # Simulate LFM output with quantization embeddings
    lfm_outputs = {
        "quantization.embeddings": torch.randn(8, 16, 32),
    }
    result = pooler(lfm_outputs)
    assert result.shape == (8, 64)


# -- Game modules --


def test_reconstruction_game_encode():
    """ReconstructionGame.encode_scene returns correct shape."""
    game = ReconstructionGame(
        ReconstructionGameConfig(
            encoder=SceneEncoderConfig(output_dim=128),
        )
    )
    gen = SceneGenerator(SceneConfig(), device="cpu")
    scene = gen.generate(8)
    agent_state = game.encode_scene(scene)
    assert agent_state.shape == (8, 128)


def test_referential_game_encode():
    """ReferentialGame.encode_target returns correct shape."""
    game = ReferentialGame(
        ReferentialGameConfig(
            encoder=SceneEncoderConfig(output_dim=128),
        )
    )
    gen = SceneGenerator(SceneConfig(), device="cpu")
    scene = gen.generate(8)
    agent_state = game.encode_target(scene)
    assert agent_state.shape == (8, 128)


# -- Single object control condition --


def test_single_object_scene():
    """Single-object scenes have trivial relations."""
    cfg = SceneConfig(num_objects=1, num_attributes=4, attribute_cardinalities=(5, 8, 3, 4))
    gen = SceneGenerator(cfg, device="cpu")
    scene = gen.generate(16)

    assert scene["object_attrs"].shape == (16, 1, 4)
    assert scene["relations"].shape == (16, 1, 1, 3)
    assert (scene["relations"] == 0).all()  # no self-relations

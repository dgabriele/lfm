"""Tests for the VAE pretraining pipeline."""

from __future__ import annotations

import torch

from lfm.data.corpus import MultilingualCorpusDataset


def test_multilingual_corpus_dataset_basic():
    """MultilingualCorpusDataset pads and stores sequences correctly."""
    token_ids = [[1, 2, 3], [4, 5, 6, 7, 8]]
    eos_id = 99
    max_seq_len = 8

    ds = MultilingualCorpusDataset(token_ids, max_seq_len, eos_id)

    assert len(ds) == 2

    tokens0, length0 = ds[0]
    assert tokens0.shape == (8,)
    assert length0 == 4  # [1, 2, 3, 99] + padding
    assert tokens0[3].item() == eos_id
    assert tokens0[4].item() == 0  # padding

    tokens1, length1 = ds[1]
    assert length1 == 6  # [4, 5, 6, 7, 8, 99] + padding


def test_multilingual_corpus_dataset_truncation():
    """Long sequences are truncated to max_seq_len - 1 + EOS."""
    token_ids = [list(range(100))]  # 100 tokens
    eos_id = 999
    max_seq_len = 10

    ds = MultilingualCorpusDataset(token_ids, max_seq_len, eos_id)
    tokens, length = ds[0]

    assert tokens.shape == (10,)
    assert length == 10  # 9 content tokens + EOS
    assert tokens[9].item() == eos_id


def test_variable_length_collate():
    """variable_length_collate stacks and returns tensors."""
    from lfm.data.collation import variable_length_collate

    batch = [
        (torch.tensor([1, 2, 3, 0, 0]), 3),
        (torch.tensor([4, 5, 6, 7, 0]), 4),
    ]

    tokens, lengths = variable_length_collate(batch)

    assert tokens.shape == (2, 5)
    assert lengths.shape == (2,)
    assert lengths[0].item() == 3
    assert lengths[1].item() == 4


def test_pretrain_config_defaults():
    """VAEPretrainConfig has conservative defaults for 6GB GPU."""
    from lfm.generator.pretrain import VAEPretrainConfig

    cfg = VAEPretrainConfig(corpus_paths=[])
    assert cfg.decoder_hidden_dim == 256
    assert cfg.decoder_num_layers == 2
    assert cfg.decoder_num_heads == 4
    assert cfg.latent_dim == 128
    assert cfg.kl_free_bits == 2.0
    assert cfg.batch_size == 32
    assert cfg.gradient_accumulation_steps == 2
    assert cfg.use_amp is True
    assert cfg.decoder_dropout == 0.2


def test_generator_config_conservative_defaults():
    """GeneratorConfig defaults match conservative pretrain config."""
    from lfm.generator.config import GeneratorConfig

    cfg = GeneratorConfig()
    assert cfg.decoder_hidden_dim == 256
    assert cfg.decoder_num_layers == 2
    assert cfg.decoder_num_heads == 4
    assert cfg.latent_dim == 128
    assert cfg.kl_free_bits == 2.0
    assert cfg.decoder_dropout == 0.2

#!/usr/bin/env python
"""Smoke-test the phoneme VAE pretraining pipeline.

Instantiates VAEPretrainer with the phoneme config, loads data, builds
model, runs 3 training steps, and exits.  Validates that the entire
pipeline wires up before committing to a full pretrain.
"""

from __future__ import annotations

from pathlib import Path

import yaml


def main() -> None:
    from lfm.generator.pretrain.config import VAEPretrainConfig

    cfg_path = Path("configs/pretrain_phoneme_vae_v1.yaml")
    with cfg_path.open() as f:
        raw = yaml.safe_load(f)

    # Tiny settings so we can exit after a handful of steps
    raw["num_epochs"] = 1
    raw["batch_size"] = 8
    raw["gradient_accumulation_steps"] = 1
    raw["diagnostic_every"] = 2
    raw["checkpoint_every_steps"] = 10_000_000  # don't bother checkpointing
    raw["val_fraction"] = 0.02  # enough val samples for diagnostic batching
    raw["output_path"] = "data/models/phoneme_v1_smoke/vae_decoder.pt"
    cfg = VAEPretrainConfig(**raw)

    # Hack: limit to a tiny subset by monkey-patching load_and_preprocess
    from lfm.generator.pretrain import data_setup as _ds
    _orig = _ds._load_and_preprocess_phoneme_h5

    def _small_loader(c):
        data, c = _orig(c)
        # Keep only the first 1024 sequences so one "epoch" is 128 steps
        import torch
        from torch.utils.data import DataLoader, random_split
        from lfm.data.corpus import MultilingualCorpusDataset
        small_ids = data.token_ids_list[:1024]
        small_langs = data.languages_list[:1024]
        data.token_ids_list = small_ids
        data.languages_list = small_langs
        data.dataset = MultilingualCorpusDataset(
            small_ids, c.max_seq_len, data.eos_id,
            word_boundary_ids=set(),
        )
        val_size = max(1, int(len(data.dataset) * c.val_fraction))
        train_size = len(data.dataset) - val_size
        data.train_dataset, data.val_dataset = random_split(
            data.dataset, [train_size, val_size],
            generator=torch.Generator().manual_seed(c.seed),
        )
        data.train_loader = DataLoader(
            data.train_dataset, batch_size=c.batch_size,
            shuffle=True, drop_last=True,
        )
        data.val_loader = DataLoader(
            data.val_dataset, batch_size=c.batch_size,
            shuffle=False, drop_last=False,
        )
        return data, c

    _ds._load_and_preprocess_phoneme_h5 = _small_loader

    # Cap steps via an early exit via num_epochs+tiny dataset → loop exits fast
    from lfm.generator.pretrain.trainer import VAEPretrainer
    trainer = VAEPretrainer(cfg)
    metrics = trainer.pretrain()
    print("Smoke pretrain metrics:", metrics)


if __name__ == "__main__":
    main()

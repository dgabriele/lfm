#!/usr/bin/env python3
"""Interpolate in the diffusion VAE's latent space.

Shows two types of interpolation:
1. Structure interpolation: vary z_struct, hold z_content constant
2. Content interpolation: vary z_content, hold z_struct constant

This reveals whether the disentangled latent split is meaningful —
structure changes should alter syntax while preserving topic,
content changes should alter vocabulary while preserving structure.

Usage:
    poetry run python scripts/interpolate_diffusion.py \
        --checkpoint data/models/dep_tree_diffusion_v1_vast.pt \
        --n-pairs 5 --steps 5
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import sentencepiece as spm
import torch

from lfm.generator.dep_tree_diffusion.config import DepTreeDiffusionConfig
from lfm.generator.dep_tree_diffusion.data import DiffusionDepTreeDataset
from lfm.generator.dep_tree_diffusion.model import DepTreeDiffusionVAE
from lfm.translator.romanize import respell

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--dataset", default="data/datasets/english-dep-trees-v16")
    parser.add_argument("--spm-path", default="data/models/v15b_ipa/spm.model")
    parser.add_argument("--n-pairs", type=int, default=5)
    parser.add_argument("--steps", type=int, default=5, help="interpolation steps")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    device = torch.device(args.device)

    cfg = DepTreeDiffusionConfig(
        dataset_path=args.dataset,
        spm_model_path=args.spm_path,
        max_seq_len=160,
        completeness_scorer_path="",
    )

    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model = DepTreeDiffusionVAE(cfg, 8050)
    model.load_state_dict(ckpt["model_state"], strict=False)
    model.eval().to(device)

    sp = spm.SentencePieceProcessor()
    sp.Load(args.spm_path)

    ds = DiffusionDepTreeDataset(Path(args.dataset) / "diffusion_cache")
    struct_dim = cfg.latent.struct_dim

    def decode_z(z, ref_sample):
        tokens = torch.tensor(np.array(ref_sample["interleaved"]).astype(np.int64)).unsqueeze(0).to(device)
        lengths = torch.tensor([len(ref_sample["interleaved"])]).to(device)
        depths = torch.tensor(np.array(ref_sample["depths"]).astype(np.int64)).unsqueeze(0).to(device)
        roles = model._extract_per_token_roles(tokens, lengths)

        z_mem = model._z_to_memory(z.unsqueeze(0))
        tok_ids = model.diffusion_decoder.sample(
            tokens.size(1), roles, depths, z_mem,
            num_steps=8, depth_scale=1.0, min_noise=1.0,
        )
        ids = [int(t) for t in tok_ids[0].tolist() if 0 < t < sp.GetPieceSize()]
        return sp.DecodeIds(ids)

    # Pick pairs with some distance between them
    rng = np.random.default_rng(42)
    indices = rng.choice(len(ds), size=args.n_pairs * 2, replace=False)

    with torch.no_grad():
        for pair_i in range(args.n_pairs):
            a_idx = int(indices[pair_i * 2])
            b_idx = int(indices[pair_i * 2 + 1])
            s_a, s_b = ds[a_idx], ds[b_idx]

            t_a = torch.tensor(np.array(s_a["interleaved"]).astype(np.int64)).unsqueeze(0).to(device)
            l_a = torch.tensor([len(s_a["interleaved"])]).to(device)
            t_b = torch.tensor(np.array(s_b["interleaved"]).astype(np.int64)).unsqueeze(0).to(device)
            l_b = torch.tensor([len(s_b["interleaved"])]).to(device)

            mu_a, _ = model.encoder(t_a, l_a)
            mu_b, _ = model.encoder(t_b, l_b)
            z_a, z_b = mu_a[0], mu_b[0]

            # Original sentences
            ids_a = [int(t) for t in s_a["interleaved"] if 0 < int(t) < sp.GetPieceSize()]
            ids_b = [int(t) for t in s_b["interleaved"] if 0 < int(t) < sp.GetPieceSize()]
            logger.info("=" * 80)
            logger.info("Pair %d", pair_i)
            logger.info("  Original A: %s", respell(sp.DecodeIds(ids_a)))
            logger.info("  Original B: %s", respell(sp.DecodeIds(ids_b)))

            # Structure interpolation (content from A held constant)
            logger.info("")
            logger.info("  --- Structure interpolation (content=A, structure A→B) ---")
            z_a_struct, z_a_content = z_a[:struct_dim], z_a[struct_dim:]
            z_b_struct = z_b[:struct_dim]
            for step in range(args.steps + 1):
                alpha = step / args.steps
                z_struct = (1 - alpha) * z_a_struct + alpha * z_b_struct
                z = torch.cat([z_struct, z_a_content])
                text = respell(decode_z(z, s_a))
                logger.info("    α=%.1f: %s", alpha, text)

            # Content interpolation (structure from A held constant)
            logger.info("")
            logger.info("  --- Content interpolation (structure=A, content A→B) ---")
            z_b_content = z_b[struct_dim:]
            for step in range(args.steps + 1):
                alpha = step / args.steps
                z_content = (1 - alpha) * z_a_content + alpha * z_b_content
                z = torch.cat([z_a_struct, z_content])
                text = respell(decode_z(z, s_a))
                logger.info("    α=%.1f: %s", alpha, text)

            logger.info("")


if __name__ == "__main__":
    main()

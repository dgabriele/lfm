"""Direct runner for SynthContrastiveGame training.

Bypasses AgentTrainer's cluster-based hard-negative sampling (the embedding
store doesn't yet have cluster_labels.npy for our last_k_concat embeddings)
and runs a simpler training loop with random in-batch distractors.

Once the game is validated end-to-end via this runner, AgentTrainer integration
+ cluster-aware sampling can be added properly.

Usage:
  poetry run python scripts/run_synth_contrastive.py CONFIG_PATH
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.optim import Adam
from transformers import PreTrainedTokenizerFast

from lfm.agents.games.synth_contrastive import (
    SynthContrastiveGame, SynthContrastiveGameConfig,
)
from lfm.synth.backend import CausalDecoderBackend
from lfm.synth.config import SynthConfig
from lfm.synth.model import SynthLM


def _setup_logger(out_dir: Path) -> logging.Logger:
    out_dir.mkdir(parents=True, exist_ok=True)
    fh = logging.FileHandler(out_dir / "train.log", mode="a")
    fh.setLevel(logging.INFO)
    fh.setFormatter(logging.Formatter("%(asctime)s %(message)s"))
    sh = logging.StreamHandler(sys.stdout)
    sh.setLevel(logging.INFO)
    sh.setFormatter(logging.Formatter("%(asctime)s %(message)s"))
    root = logging.getLogger()
    root.handlers = [fh, sh]
    root.setLevel(logging.INFO)
    return logging.getLogger(__name__)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("config")
    p.add_argument("--resume", default=None)
    args = p.parse_args()

    game_cfg = SynthContrastiveGameConfig(**yaml.safe_load(Path(args.config).read_text()))
    out_dir = Path(game_cfg.output_dir)
    log = _setup_logger(out_dir)
    log.info("game config: %s", args.config)
    log.info("phase1 checkpoint: %s", game_cfg.phase1_checkpoint)
    log.info("synth config: %s", game_cfg.synth_config)

    # Build SynthLM
    synth_cfg = SynthConfig(**yaml.safe_load(Path(game_cfg.synth_config).read_text()))
    alien_tok = PreTrainedTokenizerFast.from_pretrained(
        str(Path(synth_cfg.output_dir) / "alien_tokenizer")
    )
    backend = CausalDecoderBackend(
        synth_cfg.base_model_name, alien_vocab_size=len(alien_tok), with_reference_body=False,
    )
    synth_lm = SynthLM(backend, synth_cfg)
    raw = torch.load(game_cfg.phase1_checkpoint, map_location="cpu", weights_only=True)
    state = raw["model"] if isinstance(raw, dict) and "model" in raw else raw
    synth_lm.load_phase1_state(state)
    log.info("loaded Phase 1 from %s", game_cfg.phase1_checkpoint)

    device = torch.device(game_cfg.device)
    synth_lm.to(device)
    game = SynthContrastiveGame(game_cfg, synth_lm, synth_cfg).to(device)

    # Load embedding store. embeddings.npy is a raw memmap (no header) of
    # shape (N, n_source_positions, source_dim) float16. Compute N from file
    # size given known per-sample shape.
    emb_path = Path(game_cfg.embedding_store_dir) / "embeddings.npy"
    log.info("loading embedding store: %s", emb_path)
    bytes_per_sample = game_cfg.n_source_positions * game_cfg.source_dim * 2  # float16
    file_size = emb_path.stat().st_size
    if file_size % bytes_per_sample:
        raise ValueError(
            f"embeddings.npy size {file_size} not divisible by per-sample bytes {bytes_per_sample}"
        )
    n_passages = file_size // bytes_per_sample
    embeddings = np.memmap(
        emb_path, dtype=np.float16, mode="r",
        shape=(n_passages, game_cfg.n_source_positions, game_cfg.source_dim),
    )
    log.info("  shape=%s  dtype=%s", embeddings.shape, embeddings.dtype)
    rng = np.random.default_rng(game_cfg.seed)

    # Optimizer
    param_groups = game.trainable_param_groups()
    n_params = sum(sum(p.numel() for p in g["params"]) for g in param_groups)
    log.info("trainable params: %d", n_params)
    opt = Adam(param_groups)

    start_step = 0
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        game.load_checkpoint_state(ckpt)
        if "optimizer" in ckpt:
            opt.load_state_dict(ckpt["optimizer"])
        start_step = ckpt.get("step", 0)
        log.info("resumed from %s at step %d", args.resume, start_step)

    log.info("\n=== game training start (random in-batch distractors) ===")
    log.info(
        "batch=%d  num_distractors=%d  steps=%d  effective contrastive pool=%d-way",
        game_cfg.batch_size, game_cfg.num_distractors, game_cfg.steps,
        game_cfg.batch_size * (1 + game_cfg.num_distractors),
    )

    for step in range(start_step, game_cfg.steps):
        # Sample anchor + random distractors (no cluster-based hard negatives yet)
        idx = rng.integers(0, n_passages, size=game_cfg.batch_size)
        dist_idx = rng.integers(
            0, n_passages, size=(game_cfg.batch_size, game_cfg.num_distractors),
        )
        anchor = torch.tensor(embeddings[idx].astype(np.float32)).to(device)
        distractors = torch.tensor(embeddings[dist_idx].astype(np.float32)).to(device)

        try:
            opt.zero_grad()
            out = game(anchor, distractors, step=step)
            loss = out["loss"]
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                [p for g in param_groups for p in g["params"]], game_cfg.max_grad_norm,
            )
            opt.step()
        except torch.cuda.OutOfMemoryError as e:
            log.warning("OOM at step %d: %s", step, e)
            torch.cuda.empty_cache()
            if game_cfg.batch_size <= 4:
                raise
            new_bs = max(4, game_cfg.batch_size // 2)
            log.warning("OOM recovery: batch %d → %d", game_cfg.batch_size, new_bs)
            object.__setattr__(game_cfg, "batch_size", new_bs)
            continue

        if (step + 1) % game_cfg.log_every == 0:
            log.info(
                "step=%d  loss=%.4f  acc=%.3f  ttr=%.3f  info_nce=%.3f  topology=%.3f  "
                "bigram_kl=%.3f  adj_div=%.3f",
                step + 1, loss.item(), out["accuracy"].item(), out["ttr"].item(),
                out["info_nce"].item(), out["topology"].item(),
                out["bigram_kl"].item(), out["adj_diversity"].item(),
            )

        if (step + 1) % game_cfg.checkpoint_every == 0:
            ckpt_state = game.checkpoint_state()
            ckpt_state["step"] = step + 1
            ckpt_state["optimizer"] = opt.state_dict()
            torch.save(ckpt_state, out_dir / "latest.pt")
            log.info("checkpoint -> %s (step %d)", out_dir / "latest.pt", step + 1)

    # Final save — overwrite latest.pt; copy manually if you want a snapshot
    ckpt_state = game.checkpoint_state()
    ckpt_state["step"] = game_cfg.steps
    ckpt_state["optimizer"] = opt.state_dict()
    torch.save(ckpt_state, out_dir / "latest.pt")
    log.info("training complete -> %s (step %d)", out_dir / "latest.pt", game_cfg.steps)


if __name__ == "__main__":
    main()

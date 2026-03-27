#!/usr/bin/env python3
"""Referential game with tree-structured expression generation.

The sender produces a constituency tree of z vectors decoded through the
frozen LFM decoder as one continuous autoregressive sequence with z-switching
at segment boundaries.  Tree topology (depth, branching) is learned alongside
leaf content via REINFORCE.

Usage::

    python scripts/run_tree_game.py
    python scripts/run_tree_game.py \
        --max-depth 3 \
        --vq-codebook data/models/v1/vq_codebook.pt
"""

from __future__ import annotations

import logging
import sys

import numpy as np
import torch
import torch.nn.functional as F  # noqa: N812
from torch import Tensor, nn

from lfm.embeddings.store import EmbeddingStore
from lfm.expression import ExpressionEncoder, ExpressionGenerator
from lfm.faculty.config import FacultyConfig
from lfm.faculty.model import LanguageFaculty
from lfm.generator.config import GeneratorConfig

# Force unbuffered output
sys.stderr = open(sys.stderr.fileno(), "w", buffering=1, closefd=False)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(message)s",
    stream=sys.stderr,
)
logger = logging.getLogger(__name__)


class Receiver(nn.Module):
    """Score candidates against the message via learned dot-product."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.proj = nn.Linear(dim, dim)

    def forward(self, message: Tensor, candidates: Tensor) -> Tensor:
        projected = self.proj(message)
        return torch.bmm(
            projected.unsqueeze(1), candidates.transpose(1, 2),
        ).squeeze(1)


def main(
    store_dir: str = "data/embeddings",
    decoder_path: str = "data/vae_decoder.pt",
    spm_path: str = "data/spm.model",
    steps: int = 3000,
    sender_lr: float = 1e-4,
    receiver_lr: float = 1e-3,
    batch_size: int = 64,
    num_distractors: int = 15,
    embedding_dim: int = 384,
    device: str = "cuda",
    log_every: int = 50,
    checkpoint_every: int = 100,
    baseline_decay: float = 0.99,
    curriculum: bool = True,
    curriculum_warmup: int = 1500,
    curriculum_start: float = 0.0,
    curriculum_end: float = 1.0,
    resume: str | None = None,
    vq_codebook_path: str | None = None,
    vq_residual_alpha: float = 1.0,
    length_cost: float = 0.3,
    target_length: int = 96,
    tree_complexity_cost: float = 0.05,
    max_depth: int = 3,
    min_depth: int = 1,
    max_tokens_per_leaf: int = 96,
    max_output_len: int = 96,
) -> dict[str, float]:
    """Run the tree-structured referential game."""
    torch_device = torch.device(device)

    # Load embeddings
    store = EmbeddingStore(store_dir)
    store.load()
    logger.info(
        "Store: %d passages, dim=%d, %d clusters",
        store.num_passages, store.embedding_dim, store.num_clusters,
    )

    # Build generator (frozen decoder)
    gen_config = GeneratorConfig(
        pretrained_decoder_path=decoder_path,
        spm_model_path=spm_path,
        freeze_decoder=True,
        max_output_len=max_output_len,
        vq_codebook_path=vq_codebook_path,
        vq_residual_alpha=vq_residual_alpha,
    )
    faculty = LanguageFaculty(FacultyConfig(dim=embedding_dim, generator=gen_config))
    faculty.to(torch_device)
    faculty.generator.eval()

    # Trigger lazy init
    with torch.no_grad():
        faculty(torch.randn(1, embedding_dim).to(torch_device))

    decoder_hidden = gen_config.decoder_hidden_dim

    # Expression generator (topology + continuous z-switching decode)
    expr_gen = ExpressionGenerator(
        generator=faculty.generator,
        input_dim=embedding_dim,
        latent_dim=gen_config.latent_dim,
        hidden_dim=decoder_hidden,
        max_depth=max_depth,
        min_depth=min_depth,
        max_tokens_per_leaf=max_tokens_per_leaf,
    ).to(torch_device)

    # Expression encoder (segment pooling + bottom-up Merge)
    expr_enc = ExpressionEncoder(
        hidden_dim=decoder_hidden,
        output_dim=embedding_dim,
        max_depth=max_depth,
    ).to(torch_device)

    receiver = Receiver(embedding_dim).to(torch_device)

    # Sender params (REINFORCE)
    sender_params = list(expr_gen.parameters())
    sender_optimizer = torch.optim.Adam(sender_params, lr=sender_lr)

    # Receiver params (backprop)
    receiver_params = list(expr_enc.parameters()) + list(receiver.parameters())
    receiver_optimizer = torch.optim.Adam(receiver_params, lr=receiver_lr)

    logger.info(
        "Expression generator: %d params, Receiver: %d params, max_nodes=%d",
        sum(p.numel() for p in sender_params),
        sum(p.numel() for p in receiver_params),
        expr_gen.max_nodes,
    )

    embeddings_np = store._embeddings
    cluster_labels = store._cluster_labels
    n = embeddings_np.shape[0]
    num_candidates = num_distractors + 1
    chance = 1.0 / num_candidates
    rng = np.random.default_rng(42)

    # Per-depth baselines for reduced REINFORCE variance
    depth_baselines = [0.0] * max_depth

    # Resume
    start_step = 0
    if resume is not None:
        ckpt = torch.load(resume, map_location=torch_device, weights_only=False)
        expr_gen.load_state_dict(ckpt["expr_gen"])
        expr_enc.load_state_dict(ckpt["expr_enc"])
        receiver.load_state_dict(ckpt["receiver"])
        start_step = ckpt.get("step", 0)
        depth_baselines = ckpt.get("depth_baselines", depth_baselines)
        logger.info("Resumed from step %d", start_step)

    logger.info(
        "Expression referential game: %d distractors, chance=%.1f%%, "
        "max_depth=%d, curriculum=%d steps",
        num_distractors, chance * 100, max_depth, curriculum_warmup,
    )

    for step in range(start_step, steps):
        # --- Curriculum difficulty ---
        if curriculum:
            frac = min(step / max(curriculum_warmup, 1), 1.0)
            hard_ratio = curriculum_start + frac * (curriculum_end - curriculum_start)
        else:
            hard_ratio = 0.0

        # --- Sample batch ---
        idx = rng.integers(0, n, size=batch_size)
        anchor = torch.tensor(
            embeddings_np[idx], dtype=torch.float32,
        ).to(torch_device)

        n_hard = int(num_distractors * hard_ratio)
        n_easy = num_distractors - n_hard
        dist_indices = np.empty((batch_size, num_distractors), dtype=np.intp)
        for i in range(batch_size):
            hard_idx = np.array([], dtype=np.intp)
            if n_hard > 0:
                anchor_cluster = int(cluster_labels[idx[i]])
                hard_idx = store.sample_from_cluster(
                    anchor_cluster, n_hard, rng=rng,
                )
            easy_idx = rng.integers(0, n, size=n_easy)
            dist_indices[i] = np.concatenate([hard_idx, easy_idx])

        distractors = torch.tensor(
            embeddings_np[dist_indices], dtype=torch.float32,
        ).to(torch_device)

        # --- Sender: generate expression (topology + continuous decode) ---
        expr = expr_gen(anchor)

        # --- Receiver: encode expression → score candidates ---
        message = expr_enc(expr)

        candidates = torch.cat([anchor.unsqueeze(1), distractors], dim=1)
        perm = torch.stack(
            [torch.randperm(num_candidates, device=torch_device)
             for _ in range(batch_size)]
        )
        perm_expanded = perm.unsqueeze(-1).expand_as(candidates)
        candidates = torch.gather(candidates, 1, perm_expanded)
        target_idx = (perm == 0).long().argmax(dim=1)

        logits = receiver(message, candidates)

        # --- Receiver loss (backprop) ---
        receiver_loss = F.cross_entropy(logits, target_idx)
        receiver_optimizer.zero_grad()
        receiver_loss.backward()
        nn.utils.clip_grad_norm_(receiver_params, 1.0)
        receiver_optimizer.step()

        # --- Sender loss (REINFORCE) ---
        with torch.no_grad():
            preds = logits.argmax(dim=1)
            correct = (preds == target_idx).float()

            # Length cost: total tokens in the continuous decode
            total_tokens = expr.lengths.float()
            length_penalty = length_cost * total_tokens / target_length

            # Tree complexity cost: penalize more active nodes
            num_active = expr.num_active_nodes.float()
            complexity_penalty = tree_complexity_cost * num_active / expr_gen.max_nodes

            reward = correct * (1.0 - length_penalty - complexity_penalty).clamp(min=0)

            # Per-depth baselines
            for d in range(max_depth):
                depth_mask = expr.depth == d
                if depth_mask.any():
                    depth_baselines[d] = (
                        baseline_decay * depth_baselines[d]
                        + (1 - baseline_decay) * reward.mean().item()
                    )

            global_baseline = sum(depth_baselines) / max(len(depth_baselines), 1)
            advantage = reward - global_baseline

        # REINFORCE: push leaf mu vectors toward high-reward expressions
        active_mu = expr.leaf_mu * expr.is_leaf.unsqueeze(-1).float()
        sender_loss = -(
            advantage.unsqueeze(-1).unsqueeze(-1) * active_mu
        ).sum(dim=(-2, -1)).mean()

        sender_optimizer.zero_grad()
        sender_loss.backward()
        nn.utils.clip_grad_norm_(sender_params, 1.0)
        sender_optimizer.step()

        # --- Logging ---
        if step % log_every == 0:
            acc = correct.mean().item()
            total_len = total_tokens.mean().item()
            avg_nodes = num_active.mean().item()
            avg_leaves = expr.num_leaves.float().mean().item()
            logger.info(
                "step=%d  loss=%.3f  acc=%.1f%%  "
                "nodes=%.1f  leaves=%.1f  tokens=%.0f  "
                "hard=%.0f%%  (chance=%.1f%%)",
                step, receiver_loss.item(), acc * 100,
                avg_nodes, avg_leaves, total_len,
                hard_ratio * 100, chance * 100,
            )

        # --- Checkpoint ---
        if step > 0 and step % checkpoint_every == 0:
            ckpt = {
                "expr_gen": expr_gen.state_dict(),
                "expr_enc": expr_enc.state_dict(),
                "receiver": receiver.state_dict(),
                "step": step,
                "depth_baselines": depth_baselines,
            }
            torch.save(ckpt, "data/tree_game.pt")
            logger.info("Checkpoint saved at step %d", step)

    # Final checkpoint
    torch.save({
        "expr_gen": expr_gen.state_dict(),
        "expr_enc": expr_enc.state_dict(),
        "receiver": receiver.state_dict(),
        "step": steps,
        "depth_baselines": depth_baselines,
    }, "data/tree_game.pt")
    logger.info("Saved final checkpoint to data/tree_game.pt")

    return {
        "final_accuracy": correct.mean().item(),
        "final_receiver_loss": receiver_loss.item(),
        "chance": chance,
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Expression referential game")
    parser.add_argument("--resume", default=None)
    parser.add_argument("--steps", type=int, default=3000)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--vq-codebook", default=None)
    parser.add_argument("--vq-alpha", type=float, default=1.0)
    parser.add_argument("--max-depth", type=int, default=3)
    parser.add_argument("--min-depth", type=int, default=1)
    parser.add_argument("--max-tokens-per-leaf", type=int, default=96)
    parser.add_argument("--max-output-len", type=int, default=96)
    parser.add_argument("--length-cost", type=float, default=0.3)
    parser.add_argument("--target-length", type=int, default=96)
    parser.add_argument("--tree-complexity-cost", type=float, default=0.05)
    parser.add_argument("--curriculum-warmup", type=int, default=1500)
    args = parser.parse_args()
    main(
        resume=args.resume, steps=args.steps, batch_size=args.batch_size,
        device=args.device, vq_codebook_path=args.vq_codebook,
        vq_residual_alpha=args.vq_alpha, max_depth=args.max_depth,
        min_depth=args.min_depth, max_tokens_per_leaf=args.max_tokens_per_leaf,
        max_output_len=args.max_output_len, length_cost=args.length_cost,
        target_length=args.target_length,
        tree_complexity_cost=args.tree_complexity_cost,
        curriculum_warmup=args.curriculum_warmup,
    )

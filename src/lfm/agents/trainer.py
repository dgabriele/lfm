"""Game-agnostic training orchestrator for agent communication games."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import torch
from torch import nn

from lfm.embeddings.store import EmbeddingStore

logger = logging.getLogger(__name__)


class AgentTrainer:
    """Train an agent game with curriculum learning, logging, and checkpointing.

    The trainer is game-agnostic: any game module that implements
    ``forward(anchor, distractors) -> dict`` and ``trainable_param_groups()``
    can be plugged in.

    Args:
        game: A game module (e.g. ``ReferentialGame``).
        config: The game's config (must have training/curriculum/checkpoint fields).
    """

    def __init__(self, game: nn.Module, config) -> None:
        self.game = game
        self.config = config
        self.device = torch.device(config.device)

        # Load embedding store
        self.store = EmbeddingStore(config.embedding_store_dir)
        self.store.load()
        logger.info(
            "Store: %d passages, dim=%d, %d clusters",
            self.store.num_passages, self.store.embedding_dim,
            self.store.num_clusters,
        )

        self._embeddings = self.store._embeddings
        self._cluster_labels = self.store._cluster_labels
        self._n = self._embeddings.shape[0]
        self._rng = np.random.default_rng(config.seed)

        # Optimizer
        param_groups = game.trainable_param_groups()
        self._all_params = [p for g in param_groups for p in g["params"]]
        self.optimizer = torch.optim.Adam(param_groups)

        total_params = sum(
            sum(p.numel() for p in g["params"]) for g in param_groups
        )
        logger.info("Total trainable params: %d", total_params)

        # Output directory
        self._output_dir = Path(config.output_dir)
        self._output_dir.mkdir(parents=True, exist_ok=True)
        self._best_acc = 0.0

    def _sample_batch(
        self, step: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Sample anchor + distractors with curriculum difficulty."""
        cfg = self.config
        curriculum = cfg.curriculum

        if curriculum.enabled:
            frac = min(step / max(curriculum.warmup_steps, 1), 1.0)
            hard_ratio = (
                curriculum.start_hard_ratio
                + frac * (curriculum.end_hard_ratio - curriculum.start_hard_ratio)
            )
        else:
            hard_ratio = 0.0

        idx = self._rng.integers(0, self._n, size=cfg.batch_size)
        anchor = torch.tensor(
            self._embeddings[idx], dtype=torch.float32,
        ).to(self.device)

        dist_indices = np.empty(
            (cfg.batch_size, cfg.num_distractors), dtype=np.intp,
        )
        for i in range(cfg.batch_size):
            n_hard = int(hard_ratio * cfg.num_distractors)
            n_easy = cfg.num_distractors - n_hard

            hard_idx = np.empty(0, dtype=np.intp)
            if n_hard > 0:
                cluster = int(self._cluster_labels[idx[i]])
                hard_idx = self.store.sample_from_cluster(
                    cluster, n_hard, rng=self._rng,
                )
            easy_idx = self._rng.integers(0, self._n, size=n_easy)
            dist_indices[i] = np.concatenate([hard_idx, easy_idx])

        distractors = torch.tensor(
            self._embeddings[dist_indices], dtype=torch.float32,
        ).to(self.device)

        return anchor, distractors, hard_ratio

    def _save_checkpoint(self, step: int, accuracy: float) -> None:
        """Save latest checkpoint, and best if accuracy improved."""
        game = self.game
        gen = game.gen

        ckpt = {
            "input_proj": gen._input_proj.state_dict(),
            "input_refine": gen._input_refine.state_dict(),
            "msg_encoder": game.msg_encoder.state_dict(),
            "receiver": game.receiver.state_dict(),
            "step": step,
            "accuracy": accuracy,
        }

        torch.save(ckpt, str(self._output_dir / "latest.pt"))

        if accuracy > self._best_acc:
            self._best_acc = accuracy
            torch.save(ckpt, str(self._output_dir / "best.pt"))
            logger.info(
                "Checkpoint step %d — new best acc=%.1f%%",
                step, accuracy * 100,
            )
        else:
            logger.info(
                "Checkpoint step %d — acc=%.1f%% (best=%.1f%%)",
                step, accuracy * 100, self._best_acc * 100,
            )

    def train(self, resume: str | None = None) -> dict[str, float]:
        """Run the training loop.

        Args:
            resume: Optional path to a checkpoint to resume from.

        Returns:
            Dict with final metrics.
        """
        cfg = self.config
        game = self.game
        num_candidates = cfg.num_distractors + 1
        chance = 1.0 / num_candidates

        start_step = 0
        if resume is not None:
            ckpt = torch.load(resume, map_location=self.device, weights_only=False)
            game.gen._input_proj.load_state_dict(ckpt["input_proj"])
            game.gen._input_refine.load_state_dict(ckpt["input_refine"])
            game.msg_encoder.load_state_dict(ckpt["msg_encoder"])
            game.receiver.load_state_dict(ckpt["receiver"])
            start_step = ckpt.get("step", 0)
            logger.info("Resumed from %s at step %d", resume, start_step)

        logger.info(
            "Backprop referential game: %d distractors, chance=%.1f%%, "
            "curriculum=%s",
            cfg.num_distractors, chance * 100, cfg.curriculum.enabled,
        )
        if cfg.curriculum.enabled:
            logger.info(
                "  curriculum: %.0f%% → %.0f%% hard negatives over %d steps",
                cfg.curriculum.start_hard_ratio * 100,
                cfg.curriculum.end_hard_ratio * 100,
                cfg.curriculum.warmup_steps,
            )

        results: dict[str, float] = {}

        for step in range(start_step, cfg.steps):
            anchor, distractors, hard_ratio = self._sample_batch(step)

            out = game(anchor, distractors)
            loss = out["loss"]

            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self._all_params, cfg.max_grad_norm)
            self.optimizer.step()

            # Logging
            if step % cfg.log_every == 0:
                logger.info(
                    "step=%d  loss=%.3f  acc=%.1f%%  "
                    "avg_msg_len=%.0f  hard=%.0f%%  "
                    "(chance=%.1f%%)",
                    step,
                    loss.item(),
                    out["accuracy"].item() * 100,
                    out["msg_lengths"].item(),
                    hard_ratio * 100,
                    chance * 100,
                )

            # Checkpoint
            if step > 0 and step % cfg.checkpoint_every == 0:
                self._save_checkpoint(step, out["accuracy"].item())

        # Final checkpoint
        if cfg.steps > 0:
            with torch.no_grad():
                final_acc = out["accuracy"].item()
            self._save_checkpoint(cfg.steps, final_acc)

            results = {
                "final_accuracy": final_acc,
                "final_loss": loss.item(),
                "chance": chance,
            }
            logger.info("Final: %s", results)

        return results

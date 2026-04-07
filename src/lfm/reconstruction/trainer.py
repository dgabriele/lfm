"""Reconstruction trainer — embedding recovery through the linguistic bottleneck.

Uses cluster-based hard negative sampling: each batch is drawn from
a small number of semantic clusters, so the contrastive loss requires
fine-grained discrimination — the same pressure that drives the
dialogue game to 98%+ accuracy.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import torch
from torch import nn

from lfm.embeddings.store import EmbeddingStore
from lfm.faculty.model import LanguageFaculty
from lfm.reconstruction.config import ReconstructionConfig
from lfm.reconstruction.model import ReconstructionModel

logger = logging.getLogger(__name__)


class ReconstructionTrainer:
    """Train a reconstruction model on embeddings.

    Args:
        config: Reconstruction configuration.
    """

    def __init__(self, config: ReconstructionConfig) -> None:
        self.config = config
        self.device = torch.device(config.device)

        # Load embeddings with cluster info
        self.store = EmbeddingStore(config.embedding_store_dir)
        self.store.load()
        self._embeddings = np.array(self.store._embeddings)
        self._cluster_labels = self.store._cluster_labels
        self._n = len(self._embeddings)
        self._num_clusters = int(self._cluster_labels.max()) + 1
        self._rng = np.random.default_rng(config.seed)

        logger.info(
            "Loaded %d embeddings, dim=%d, %d clusters",
            self._n, self._embeddings.shape[1], self._num_clusters,
        )

        # Build model
        faculty = LanguageFaculty(
            config.build_faculty_config(),
        ).to(self.device)
        self.model = ReconstructionModel(config, faculty).to(self.device)

        # Optimizer
        groups = self.model.param_groups()
        self._all_params = [p for g in groups for p in g["params"]]
        self.optimizer = torch.optim.Adam(groups)

        total_params = sum(p.numel() for p in self._all_params)
        logger.info("Trainable params: %d", total_params)

        # Output
        self._output_dir = Path(config.output_dir)
        self._output_dir.mkdir(parents=True, exist_ok=True)
        self._best_sim = 0.0
        self._batch_size = config.batch_size

    def _sample_batch(self, hard_ratio: float = 1.0) -> torch.Tensor:
        """Sample a batch with cluster-based hard negatives.

        Args:
            hard_ratio: Fraction of batch drawn from same-cluster
                hard negatives. 0.0 = fully random, 1.0 = all hard.

        Returns:
            ``(batch_size, dim)`` tensor of embeddings where nearby
            items are semantically similar (hard to distinguish).
        """
        bs = self._batch_size
        n_hard = int(hard_ratio * bs)
        n_easy = bs - n_hard

        indices = np.empty(bs, dtype=np.intp)

        if n_hard > 0:
            # Pick a few clusters and fill the hard portion from them
            n_clusters = max(1, n_hard // 16)  # ~16 samples per cluster
            clusters = self._rng.integers(0, self._num_clusters, size=n_clusters)
            per_cluster = n_hard // n_clusters
            remainder = n_hard - per_cluster * n_clusters

            pos = 0
            for i, c in enumerate(clusters):
                n = per_cluster + (1 if i < remainder else 0)
                indices[pos:pos + n] = self.store.sample_from_cluster(
                    int(c), n, rng=self._rng,
                )
                pos += n

        if n_easy > 0:
            indices[n_hard:] = self._rng.integers(0, self._n, size=n_easy)

        # Shuffle so cluster boundaries aren't at fixed positions
        self._rng.shuffle(indices)

        return torch.tensor(
            self._embeddings[indices], dtype=torch.float32,
        ).to(self.device)

    def _save_checkpoint(self, step: int, cos_sim: float) -> None:
        ckpt = self.model.checkpoint_state()
        ckpt["step"] = step
        ckpt["cosine_sim"] = cos_sim
        torch.save(ckpt, str(self._output_dir / "latest.pt"))

        if cos_sim > self._best_sim:
            self._best_sim = cos_sim
            torch.save(ckpt, str(self._output_dir / "best.pt"))
            logger.info(
                "Checkpoint step %d — new best cos_sim=%.4f",
                step, cos_sim,
            )
        else:
            logger.info(
                "Checkpoint step %d — cos_sim=%.4f (best=%.4f)",
                step, cos_sim, self._best_sim,
            )

    def train(self, resume: str | None = None) -> dict[str, float]:
        """Run the training loop.

        Args:
            resume: Optional path to checkpoint.

        Returns:
            Dict with final metrics.
        """
        cfg = self.config
        model = self.model

        start_step = 0
        if resume is not None:
            ckpt = torch.load(resume, map_location=self.device, weights_only=False)
            model.load_checkpoint_state(ckpt)
            start_step = ckpt.get("step", 0)
            self._best_sim = ckpt.get("cosine_sim", 0.0)
            logger.info("Resumed from %s at step %d", resume, start_step)

        # LR warmup schedule
        def lr_lambda(step):
            if step < cfg.warmup_steps:
                return step / max(cfg.warmup_steps, 1)
            return 1.0

        scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.optimizer, lr_lambda,
        )

        logger.info(
            "Training: %d steps, batch=%d, z_gen_lr=%.1e, inverse_lr=%.1e",
            cfg.steps, cfg.batch_size, cfg.z_gen_lr, cfg.inverse_lr,
        )

        for step in range(start_step, cfg.steps):
            # Ramp hard ratio over warmup
            frac = min(step / max(cfg.warmup_steps, 1), 1.0)
            hard_ratio = frac  # 0→1 over warmup

            try:
                self.optimizer.zero_grad()
                embeddings = self._sample_batch(hard_ratio)
                out = model(embeddings)
                out["loss"].backward()
                nn.utils.clip_grad_norm_(self._all_params, cfg.max_grad_norm)
                self.optimizer.step()
                scheduler.step()
            except RuntimeError as e:
                if "out of memory" not in str(e):
                    raise
                torch.cuda.empty_cache()
                new_bs = max(4, int(self._batch_size * 0.9))
                logger.warning(
                    "OOM at step %d — reducing batch_size %d → %d",
                    step, self._batch_size, new_bs,
                )
                self._batch_size = new_bs
                self.optimizer.zero_grad()
                continue

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            # Logging
            if step % cfg.log_every == 0:
                sim = out["cosine_sim"].item()
                z_sim = out.get("z_cos_sim", torch.tensor(0.0)).item()
                loss = out["loss"].item()
                hs = out.get("hs_loss", torch.tensor(0.0)).item()
                d_acc = out.get("direct_acc", torch.tensor(0.0)).item()
                s_acc = out.get("surface_acc", torch.tensor(0.0)).item()
                n_phr = out["num_phrases"].item()
                n_tok = out["total_tokens"].item()
                vram = ""
                if torch.cuda.is_available():
                    vram_mb = torch.cuda.max_memory_allocated() // (1024 * 1024)
                    vram = f"  vram={vram_mb}MB"
                    torch.cuda.reset_peak_memory_stats()
                logger.info(
                    "step=%d  loss=%.3f  z_sim=%.3f  surf_sim=%.3f  "
                    "d_acc=%.1f%%  s_acc=%.1f%%  hs=%.4f  "
                    "phr=%.1f  tok=%.0f  hard=%.0f%%%s",
                    step, loss, z_sim, sim,
                    d_acc * 100, s_acc * 100, hs,
                    n_phr, n_tok, hard_ratio * 100, vram,
                )

            # Save best immediately
            sim = out["cosine_sim"].item()
            if sim > self._best_sim:
                self._best_sim = sim
                ckpt = model.checkpoint_state()
                ckpt["step"] = step
                ckpt["cosine_sim"] = sim
                torch.save(ckpt, str(self._output_dir / "best.pt"))
                logger.info(
                    "  New best cos_sim=%.4f at step %d",
                    sim, step,
                )

            # Periodic checkpoint
            if step > 0 and step % cfg.checkpoint_every == 0:
                self._save_checkpoint(step, sim)

        # Final
        final_sim = out["cosine_sim"].item()
        self._save_checkpoint(cfg.steps, final_sim)
        logger.info("Final: cos_sim=%.4f", final_sim)

        return {"cosine_sim": final_sim, "loss": out["loss"].item()}

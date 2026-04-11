"""Main training loop for unsupervised NMT.

The trainer coordinates four per-step losses on a single shared model:

* ``dae_ng`` — denoising autoencoder on Neuroglot
* ``dae_en`` — denoising autoencoder on English
* ``bt_ng2en`` — backtranslation ng → en → ng (after warmup)
* ``bt_en2ng`` — backtranslation en → ng → en (after warmup)

Only DAE runs during the warmup window ``[0, bt_start_step)``.  After
that, BT is added with its own loss weight.  This delayed-BT schedule
matches Lample et al. 2018 and is important for stability: early
backtranslations are random, and training on them aggressively can
destabilize the shared encoder before it has even learned denoising.

All of this is built around the plain ``MonolingualDataset`` + the
collator from :mod:`lfm.unmt.data.monolingual`.  Each training step
reads one batch from each language and computes all four losses.

Checkpointing is minimal: every ``checkpoint_every`` steps the model
weights, optimizer state, and training step counter are saved to
``<output_dir>/latest.pt``.  Rerunning with the same config resumes
from that checkpoint automatically.
"""

from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass, field
from pathlib import Path

import torch
import torch.nn as nn
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader

from lfm.unmt.config import UNMTConfig
from lfm.unmt.data.monolingual import (
    MonolingualDataset,
    build_noise_config,
    pad_batch,
)
from lfm.unmt.model.transformer import (
    SharedNMTTransformer,
    build_model,
    initialize_from_muse,
)
from lfm.unmt.tokenizer import (
    BilingualTokenizer,
    EN_TAG_ID,
    NG_TAG_ID,
    load_tokenizer,
)
from lfm.unmt.training.backtranslation import compute_bt_loss
from lfm.unmt.training.dae import compute_dae_loss

logger = logging.getLogger(__name__)

_CHECKPOINT_NAME = "latest.pt"


@dataclass
class TrainingState:
    """Mutable training state saved/loaded across resumes."""

    step: int = 0
    best_round_trip_bleu: float = 0.0
    last_log_time: float = field(default_factory=time.time)


def _infinite_iter(loader: DataLoader):
    """Yield batches forever by cycling through the loader."""
    while True:
        for batch in loader:
            yield batch


def _move_to_device(
    batch: dict[str, torch.Tensor], device: torch.device,
) -> dict[str, torch.Tensor]:
    return {k: v.to(device, non_blocking=True) for k, v in batch.items()}


def _build_scheduler(
    optimizer: torch.optim.Optimizer,
    warmup_steps: int,
    total_steps: int,
) -> LambdaLR:
    """Linear warmup then cosine decay to 10% of peak."""
    def _lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return step / max(warmup_steps, 1)
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        return 0.1 + 0.9 * 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))
    return LambdaLR(optimizer, _lr_lambda)


class UNMTTrainer:
    """Trains a shared-weight seq2seq transformer with DAE + BT losses.

    Args:
        config: UNMT configuration.
        checkpoint_every: Save frequency in optimizer steps.
        log_every: Log frequency in optimizer steps.
    """

    def __init__(
        self,
        config: UNMTConfig,
        checkpoint_every: int = 1000,
        log_every: int = 50,
    ) -> None:
        self.config = config
        self.checkpoint_every = checkpoint_every
        self.log_every = log_every

        self.device = torch.device(
            config.device if torch.cuda.is_available() else "cpu",
        )
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.tokenizer: BilingualTokenizer = load_tokenizer(config)
        self.model = build_model(config, self.tokenizer).to(self.device)
        initialize_from_muse(self.model, config, self.tokenizer)

        noise_cfg = build_noise_config(config)
        self.ng_dataset = MonolingualDataset(
            corpus_path=config.neuroglot_corpus,
            tokenizer=self.tokenizer,
            lang="ng",
            max_len=config.max_len,
            noise=noise_cfg,
            seed=config.seed,
        )
        self.en_dataset = MonolingualDataset(
            corpus_path=config.english_corpus,
            tokenizer=self.tokenizer,
            lang="en",
            max_len=config.max_len,
            noise=noise_cfg,
            seed=config.seed,
        )

        self.ng_loader = DataLoader(
            self.ng_dataset,
            batch_size=config.batch_size,
            shuffle=True,
            collate_fn=pad_batch,
            num_workers=2,
            drop_last=True,
        )
        self.en_loader = DataLoader(
            self.en_dataset,
            batch_size=config.batch_size,
            shuffle=True,
            collate_fn=pad_batch,
            num_workers=2,
            drop_last=True,
        )

        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=config.lr,
            betas=(0.9, 0.98),
            eps=1e-9,
        )
        self.scheduler = _build_scheduler(
            self.optimizer,
            warmup_steps=config.warmup_steps,
            total_steps=config.total_steps,
        )

        self.state = TrainingState()
        self._try_resume()

    # -- checkpoint i/o --

    def _checkpoint_path(self) -> Path:
        return self.output_dir / _CHECKPOINT_NAME

    def _save_checkpoint(self) -> None:
        torch.save(
            {
                "step": self.state.step,
                "best_round_trip_bleu": self.state.best_round_trip_bleu,
                "model": self.model.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "scheduler": self.scheduler.state_dict(),
            },
            self._checkpoint_path(),
        )
        logger.info("Saved checkpoint at step %d", self.state.step)

    def _try_resume(self) -> None:
        path = self._checkpoint_path()
        if not path.exists():
            return
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(ckpt["model"])
        self.optimizer.load_state_dict(ckpt["optimizer"])
        self.scheduler.load_state_dict(ckpt["scheduler"])
        self.state.step = ckpt.get("step", 0)
        self.state.best_round_trip_bleu = ckpt.get("best_round_trip_bleu", 0.0)
        logger.info("Resumed from %s at step %d", path, self.state.step)

    # -- one training step --

    def _compute_losses(
        self,
        ng_batch: dict[str, torch.Tensor],
        en_batch: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        cfg = self.config
        losses: dict[str, torch.Tensor] = {}

        # DAE on both languages
        losses["dae_ng"] = compute_dae_loss(self.model, ng_batch)
        losses["dae_en"] = compute_dae_loss(self.model, en_batch)

        total = cfg.dae_weight * (losses["dae_ng"] + losses["dae_en"])

        if self.state.step >= cfg.bt_start_step and cfg.bt_weight > 0:
            # BT in both directions.  Each direction consumes its own
            # language's clean batch and generates into the other.
            losses["bt_ng2en"] = compute_bt_loss(
                self.model,
                ng_batch,
                target_lang_tag_id=EN_TAG_ID,
                target_token_range=self.tokenizer.english_range,
                max_len=cfg.max_len,
            )
            losses["bt_en2ng"] = compute_bt_loss(
                self.model,
                en_batch,
                target_lang_tag_id=NG_TAG_ID,
                target_token_range=self.tokenizer.neuroglot_range,
                max_len=cfg.max_len,
            )
            total = total + cfg.bt_weight * (
                losses["bt_ng2en"] + losses["bt_en2ng"]
            )

        losses["total"] = total
        return losses

    # -- main loop --

    def train(self) -> None:
        """Run the full training loop from current step to total_steps."""
        cfg = self.config
        self.model.train()

        ng_iter = _infinite_iter(self.ng_loader)
        en_iter = _infinite_iter(self.en_loader)

        logger.info(
            "UNMT training: target_steps=%d warmup=%d bt_start=%d "
            "batch=%d device=%s",
            cfg.total_steps, cfg.warmup_steps, cfg.bt_start_step,
            cfg.batch_size, self.device,
        )

        running: dict[str, float] = {}
        start_time = time.time()
        step_start = self.state.step

        while self.state.step < cfg.total_steps:
            ng_batch = _move_to_device(next(ng_iter), self.device)
            en_batch = _move_to_device(next(en_iter), self.device)

            losses = self._compute_losses(ng_batch, en_batch)
            total = losses["total"]

            self.optimizer.zero_grad(set_to_none=True)
            total.backward()
            nn.utils.clip_grad_norm_(
                self.model.parameters(), cfg.max_grad_norm,
            )
            self.optimizer.step()
            self.scheduler.step()

            # Running means for logging.
            for k, v in losses.items():
                running[k] = running.get(k, 0.0) + float(v.detach())

            self.state.step += 1

            if self.state.step % self.log_every == 0:
                elapsed = time.time() - start_time
                steps_done = self.state.step - step_start
                rate = steps_done / max(elapsed, 1e-6)
                avg = {k: v / self.log_every for k, v in running.items()}
                lr = self.scheduler.get_last_lr()[0]
                msg_parts = [
                    f"step={self.state.step}/{cfg.total_steps}",
                    f"lr={lr:.2e}",
                    f"rate={rate:.1f}/s",
                    f"total={avg.get('total', 0):.3f}",
                    f"dae_ng={avg.get('dae_ng', 0):.3f}",
                    f"dae_en={avg.get('dae_en', 0):.3f}",
                ]
                if "bt_ng2en" in avg:
                    msg_parts.append(f"bt_ng2en={avg['bt_ng2en']:.3f}")
                    msg_parts.append(f"bt_en2ng={avg['bt_en2ng']:.3f}")
                logger.info("  " + " ".join(msg_parts))
                running.clear()

            if self.state.step % self.checkpoint_every == 0:
                self._save_checkpoint()

        self._save_checkpoint()
        logger.info("Training complete.")

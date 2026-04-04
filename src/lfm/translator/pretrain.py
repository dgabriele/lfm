"""Self-supervised pretraining on romanized IPA corpus.

Standard causal language model training: the LLM learns to predict
the next token in the alien language.  No paired translations needed.
After pretraining, translation emerges via few-shot cross-lingual transfer.
"""

from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass, field
from pathlib import Path

import torch
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, Dataset

from lfm.translator.config import PretrainConfig

logger = logging.getLogger(__name__)

_CHECKPOINT_NAME = "training_state.pt"


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------


class PlainTextDataset(Dataset):
    """Tokenized plain text lines for causal LM training."""

    def __init__(self, lines: list[str], tokenizer, max_len: int = 128) -> None:
        self.examples: list[dict[str, torch.Tensor]] = []
        for line in lines:
            line = line.strip()
            if not line:
                continue
            enc = tokenizer(
                line, max_length=max_len, padding="max_length",
                truncation=True, return_tensors="pt",
            )
            ids = enc["input_ids"].squeeze(0)
            mask = enc["attention_mask"].squeeze(0)
            labels = ids.clone()
            labels[mask == 0] = -100
            self.examples.append({"input_ids": ids, "attention_mask": mask, "labels": labels})

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> dict:
        return self.examples[idx]


# ---------------------------------------------------------------------------
# Training state
# ---------------------------------------------------------------------------


@dataclass
class TrainingState:
    """Mutable training state, saved/loaded for resume."""

    epoch: int = 0
    global_step: int = 0
    best_val_loss: float = float("inf")

    def save(
        self, path: Path, optimizer, scheduler, scaler, model_path: str,
    ) -> None:
        torch.save({
            "epoch": self.epoch,
            "global_step": self.global_step,
            "best_val_loss": self.best_val_loss,
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "scaler": scaler.state_dict(),
            "model_path": model_path,
        }, path)
        logger.info("Saved training state (epoch=%d, step=%d)", self.epoch, self.global_step)

    @classmethod
    def load(cls, path: Path, optimizer, scheduler, scaler) -> "TrainingState":
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        state = cls(
            epoch=ckpt["epoch"],
            global_step=ckpt["global_step"],
            best_val_loss=ckpt.get("best_val_loss", float("inf")),
        )
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        if ckpt.get("scaler"):
            try:
                scaler.load_state_dict(ckpt["scaler"])
            except Exception:
                pass
        logger.info(
            "Restored: epoch=%d, step=%d, best_val=%.4f",
            state.epoch, state.global_step, state.best_val_loss,
        )
        return state


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------


class SelfSupervisedTrainer:
    """Continue pretraining a causal LM on romanized IPA corpus.

    Supports full resume: model weights, optimizer, scheduler, epoch,
    and step are all saved per-epoch and restored automatically.

    Args:
        config: Pretraining configuration.
    """

    def __init__(self, config: PretrainConfig) -> None:
        self.config = config
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    # -- Setup --

    def _load_model(self):
        """Load model and tokenizer, preferring latest checkpoint."""
        from transformers import AutoModelForCausalLM, AutoTokenizer

        # Prefer saved model for resume, fall back to config
        latest_path = self.output_dir / "model_latest"
        ckpt_path = self.output_dir / _CHECKPOINT_NAME
        if latest_path.exists() and ckpt_path.exists():
            model_path = str(latest_path)
            logger.info("Resuming from %s", model_path)
        else:
            model_path = self.config.model_name
            logger.info("Loading base model: %s", model_path)

        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForCausalLM.from_pretrained(model_path)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        return model, tokenizer

    def _build_dataloaders(self, tokenizer):
        """Load corpus, split, tokenize, return train/val loaders."""
        cfg = self.config
        logger.info("Loading corpus from %s", cfg.corpus_path)
        with open(cfg.corpus_path, "r", encoding="utf-8") as f:
            lines = f.readlines()

        n_val = max(1, int(len(lines) * 0.05))
        logger.info("Tokenizing %d train + %d val lines...", len(lines) - n_val, n_val)

        train_ds = PlainTextDataset(lines[n_val:], tokenizer, cfg.max_len)
        val_ds = PlainTextDataset(lines[:n_val], tokenizer, cfg.max_len)

        train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=cfg.batch_size)

        logger.info("Train: %d examples, Val: %d examples", len(train_ds), len(val_ds))
        return train_loader, val_loader

    def _build_optimizer(self, model, total_steps):
        """Create optimizer, scheduler, and grad scaler."""
        cfg = self.config
        optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=0.01)
        warmup_steps = int(total_steps * cfg.warmup_fraction)

        def _lr_lambda(step: int) -> float:
            if step < warmup_steps:
                return step / max(warmup_steps, 1)
            progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
            return 0.1 + 0.9 * 0.5 * (1.0 + math.cos(math.pi * progress))

        scheduler = LambdaLR(optimizer, _lr_lambda)

        model_dtype = next(model.parameters()).dtype
        use_scaler = cfg.use_amp and model_dtype != torch.bfloat16
        scaler = torch.amp.GradScaler(enabled=use_scaler)

        return optimizer, scheduler, scaler, model_dtype

    # -- Training loop --

    def _train_epoch(
        self, model, tokenizer, train_loader, optimizer, scheduler, scaler,
        model_dtype, state: TrainingState,
    ) -> float:
        """Run one training epoch with periodic checkpointing. Returns mean train loss."""
        cfg = self.config
        device = torch.device(cfg.device)
        model.train()
        epoch_loss = 0.0
        optimizer.zero_grad()

        num_batches = len(train_loader)
        checkpoint_interval = max(1, int(num_batches * cfg.checkpoint_fraction))

        for i, batch in enumerate(train_loader):
            ids = batch["input_ids"].to(device)
            mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            with torch.amp.autocast(device_type=device.type, enabled=cfg.use_amp, dtype=model_dtype):
                out = model(input_ids=ids, attention_mask=mask, labels=labels)
                loss = out.loss / cfg.gradient_accumulation_steps

            scaler.scale(loss).backward()

            if (i + 1) % cfg.gradient_accumulation_steps == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad()
                state.global_step += 1

            epoch_loss += out.loss.item()

            if (i + 1) % 50 == 0:
                lr = optimizer.param_groups[0]["lr"]
                logger.info(
                    "  epoch=%d step=%d loss=%.4f lr=%.2e",
                    state.epoch, state.global_step, epoch_loss / (i + 1), lr,
                )

            # Mid-epoch checkpoint
            if (i + 1) % checkpoint_interval == 0 and (i + 1) < num_batches:
                avg_loss = epoch_loss / (i + 1)
                logger.info(
                    "  Mid-epoch checkpoint (%.0f%%) — step=%d loss=%.4f",
                    (i + 1) / num_batches * 100, state.global_step, avg_loss,
                )
                model.save_pretrained(self.output_dir / "model_latest")
                tokenizer.save_pretrained(self.output_dir / "model_latest")
                state.save(
                    self.output_dir / _CHECKPOINT_NAME,
                    optimizer, scheduler, scaler,
                    str(self.output_dir / "model_latest"),
                )

        return epoch_loss / num_batches

    @torch.no_grad()
    def _validate(self, model, val_loader, model_dtype) -> float:
        """Run validation. Returns mean val loss."""
        cfg = self.config
        device = torch.device(cfg.device)
        model.eval()
        total = 0.0
        for batch in val_loader:
            ids = batch["input_ids"].to(device)
            mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            out = model(input_ids=ids, attention_mask=mask, labels=labels)
            total += out.loss.item()
        return total / max(len(val_loader), 1)

    def _save_checkpoint(self, model, tokenizer, optimizer, scheduler, scaler,
                         state: TrainingState, val_loss: float) -> None:
        """Save model weights + full training state."""
        # Always save latest (for resume)
        model.save_pretrained(self.output_dir / "model_latest")
        tokenizer.save_pretrained(self.output_dir / "model_latest")
        state.save(
            self.output_dir / _CHECKPOINT_NAME,
            optimizer, scheduler, scaler,
            str(self.output_dir / "model_latest"),
        )

        # Save best (for inference)
        if val_loss < state.best_val_loss:
            state.best_val_loss = val_loss
            model.save_pretrained(self.output_dir / "model")
            tokenizer.save_pretrained(self.output_dir / "model")
            logger.info("  Saved best model (val_loss=%.4f)", val_loss)

    # -- Entry point --

    def train(self) -> dict[str, float]:
        """Run self-supervised pretraining with automatic resume."""
        cfg = self.config
        model, tokenizer = self._load_model()
        model.to(torch.device(cfg.device))

        total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logger.info("Model: %d params (%.1fM)", total_params, total_params / 1e6)

        train_loader, val_loader = self._build_dataloaders(tokenizer)
        steps_per_epoch = len(train_loader) // cfg.gradient_accumulation_steps
        total_steps = steps_per_epoch * cfg.epochs

        optimizer, scheduler, scaler, model_dtype = self._build_optimizer(model, total_steps)

        # Resume state if checkpoint exists
        ckpt_path = self.output_dir / _CHECKPOINT_NAME
        if ckpt_path.exists():
            state = TrainingState.load(ckpt_path, optimizer, scheduler, scaler)
        else:
            state = TrainingState()

        logger.info(
            "Training: epochs %d→%d, %d steps/epoch, lr=%.1e, batch=%d×%d",
            state.epoch, cfg.epochs, steps_per_epoch, cfg.lr,
            cfg.batch_size, cfg.gradient_accumulation_steps,
        )

        train_loss = 0.0
        for epoch in range(state.epoch, cfg.epochs):
            state.epoch = epoch
            epoch_start = time.time()

            train_loss = self._train_epoch(
                model, tokenizer, train_loader, optimizer, scheduler, scaler, model_dtype, state,
            )
            val_loss = self._validate(model, val_loader, model_dtype)

            logger.info(
                "Epoch %d/%d (%.0fs) — train=%.4f val=%.4f ppl=%.2f",
                epoch + 1, cfg.epochs, time.time() - epoch_start,
                train_loss, val_loss, math.exp(min(val_loss, 20)),
            )

            state.epoch = epoch + 1
            self._save_checkpoint(model, tokenizer, optimizer, scheduler, scaler, state, val_loss)

        logger.info("Training complete. Best val_loss=%.4f", state.best_val_loss)
        return {"best_val_loss": state.best_val_loss, "final_train_loss": train_loss}

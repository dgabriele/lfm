"""Self-supervised pretraining on romanized IPA corpus.

Standard causal language model training: the LLM learns to predict
the next token in the alien language.  No paired translations needed.
After pretraining, translation emerges via few-shot cross-lingual transfer.
"""

from __future__ import annotations

import logging
import math
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset

from lfm.translator.config import PretrainConfig

logger = logging.getLogger(__name__)


class PlainTextDataset(Dataset):
    """Simple dataset for causal LM training on plain text lines."""

    def __init__(self, lines: list[str], tokenizer, max_len: int = 128):
        self.examples = []
        for line in lines:
            line = line.strip()
            if not line:
                continue
            enc = tokenizer(
                line,
                max_length=max_len,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            )
            input_ids = enc["input_ids"].squeeze(0)
            attention_mask = enc["attention_mask"].squeeze(0)

            # Labels = input_ids shifted (causal LM), masked where padding
            labels = input_ids.clone()
            labels[attention_mask == 0] = -100

            self.examples.append({
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "labels": labels,
            })

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> dict:
        return self.examples[idx]


class SelfSupervisedTrainer:
    """Continue pretraining a causal LM on romanized IPA corpus.

    Args:
        config: Pretraining configuration.
    """

    def __init__(self, config: PretrainConfig) -> None:
        self.config = config

    def train(self) -> dict[str, float]:
        """Run self-supervised pretraining.

        Returns:
            Dict with final metrics.
        """
        cfg = self.config
        device = torch.device(cfg.device)
        torch.manual_seed(cfg.seed)

        # Load model + tokenizer
        from transformers import AutoModelForCausalLM, AutoTokenizer

        logger.info("Loading model: %s", cfg.model_name)
        tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)
        model = AutoModelForCausalLM.from_pretrained(cfg.model_name)

        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        model.to(device)
        total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logger.info("Model loaded: %d trainable params (%.1fM)", total_params, total_params / 1e6)

        # Load corpus
        logger.info("Loading corpus from %s", cfg.corpus_path)
        with open(cfg.corpus_path, "r", encoding="utf-8") as f:
            lines = f.readlines()
        logger.info("Loaded %d lines", len(lines))

        # Train/val split
        n_val = max(1, int(len(lines) * 0.05))
        val_lines = lines[:n_val]
        train_lines = lines[n_val:]

        logger.info("Tokenizing %d train + %d val lines...", len(train_lines), len(val_lines))
        train_dataset = PlainTextDataset(train_lines, tokenizer, cfg.max_len)
        val_dataset = PlainTextDataset(val_lines, tokenizer, cfg.max_len)

        train_loader = DataLoader(train_dataset, batch_size=cfg.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=cfg.batch_size)

        logger.info("Train: %d examples, Val: %d examples", len(train_dataset), len(val_dataset))

        # Optimizer + scheduler (linear warmup + cosine decay)
        optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=0.01)
        total_steps = len(train_loader) // cfg.gradient_accumulation_steps * cfg.epochs
        warmup_steps = int(total_steps * cfg.warmup_fraction)

        def _lr_lambda(step: int) -> float:
            if step < warmup_steps:
                return step / max(warmup_steps, 1)
            progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
            return 0.1 + 0.9 * 0.5 * (1.0 + math.cos(math.pi * progress))

        from torch.optim.lr_scheduler import LambdaLR
        scheduler = LambdaLR(optimizer, _lr_lambda)

        # Disable GradScaler if model uses bfloat16 (Qwen default)
        model_dtype = next(model.parameters()).dtype
        use_scaler = cfg.use_amp and model_dtype != torch.bfloat16
        scaler = torch.amp.GradScaler(enabled=use_scaler)

        # Output dir
        output_dir = Path(cfg.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        logger.info(
            "Training: %d epochs, %d steps, lr=%.1e, batch=%d×%d",
            cfg.epochs, total_steps, cfg.lr, cfg.batch_size, cfg.gradient_accumulation_steps,
        )

        best_val_loss = float("inf")
        global_step = 0

        for epoch in range(cfg.epochs):
            model.train()
            epoch_loss = 0.0
            epoch_start = time.time()
            optimizer.zero_grad()

            for i, batch in enumerate(train_loader):
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["labels"].to(device)

                with torch.amp.autocast(device_type=device.type, enabled=cfg.use_amp, dtype=model_dtype):
                    outputs = model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        labels=labels,
                    )
                    loss = outputs.loss / cfg.gradient_accumulation_steps

                scaler.scale(loss).backward()

                if (i + 1) % cfg.gradient_accumulation_steps == 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)
                    scaler.step(optimizer)
                    scaler.update()
                    scheduler.step()
                    optimizer.zero_grad()
                    global_step += 1

                epoch_loss += outputs.loss.item()

                if (i + 1) % 50 == 0:
                    avg = epoch_loss / (i + 1)
                    lr = optimizer.param_groups[0]["lr"]
                    logger.info(
                        "  epoch=%d step=%d loss=%.4f lr=%.2e",
                        epoch, global_step, avg, lr,
                    )

            epoch_time = time.time() - epoch_start
            train_loss = epoch_loss / len(train_loader)

            # Validation
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for batch in val_loader:
                    input_ids = batch["input_ids"].to(device)
                    attention_mask = batch["attention_mask"].to(device)
                    labels = batch["labels"].to(device)
                    outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                    val_loss += outputs.loss.item()
            val_loss /= max(len(val_loader), 1)

            logger.info(
                "Epoch %d/%d (%.0fs) — train_loss=%.4f val_loss=%.4f ppl=%.2f",
                epoch + 1, cfg.epochs, epoch_time,
                train_loss, val_loss, math.exp(min(val_loss, 20)),
            )

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                model.save_pretrained(output_dir / "model")
                tokenizer.save_pretrained(output_dir / "model")
                logger.info("  Saved best model (val_loss=%.4f)", val_loss)

        logger.info("Training complete. Best val_loss=%.4f", best_val_loss)

        return {
            "best_val_loss": best_val_loss,
            "final_train_loss": train_loss,
        }

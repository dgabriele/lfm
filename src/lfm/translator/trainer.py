"""Translator training: fine-tune a causal LM on IPA -> English pairs."""

from __future__ import annotations

import json
import logging
import math
from pathlib import Path

import numpy as np
import torch
import yaml
from torch import nn
from torch.utils.data import DataLoader

from lfm.generator.training_history import TrainingHistory
from lfm.translator.config import TranslatorConfig
from lfm.translator.dataset import IPATranslationDataset

logger = logging.getLogger(__name__)


class TranslatorTrainer:
    """Fine-tune a HuggingFace causal LM on IPA -> English translation.

    Supports:
    - Any ``AutoModelForCausalLM``-compatible model
    - Optional LoRA via ``peft`` (gated behind ``use_lora``)
    - AMP mixed precision
    - Gradient accumulation with cosine LR + warmup
    - TrainingHistory session logging

    Args:
        config: Translator training configuration.
    """

    def __init__(self, config: TranslatorConfig) -> None:
        self.config = config

    def train(self) -> dict[str, float]:
        """Run the full training pipeline.

        Returns:
            Dict of training and validation metrics.
        """
        from transformers import AutoModelForCausalLM, AutoTokenizer

        cfg = self.config
        torch.manual_seed(cfg.seed)
        device = torch.device(cfg.device)

        # Load model and tokenizer
        logger.info("Loading model: %s", cfg.model_name)
        tokenizer = AutoTokenizer.from_pretrained(cfg.model_name, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            cfg.model_name,
            torch_dtype=torch.float16,
            trust_remote_code=True,
        ).to(device)

        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            model.config.pad_token_id = model.config.eos_token_id

        # Add special tokens
        special_tokens = {"additional_special_tokens": ["<ipa>", "</ipa>", "<eng>", "</eng>"]}
        num_added = tokenizer.add_special_tokens(special_tokens)
        if num_added > 0:
            model.resize_token_embeddings(len(tokenizer))
            logger.info("Added %d special tokens", num_added)

        # Optional LoRA
        if cfg.use_lora:
            model = self._apply_lora(model, cfg)

        # Load pairs
        logger.info("Loading pairs from %s", cfg.pairs_path)
        pairs = self._load_pairs(cfg.pairs_path)
        logger.info("Loaded %d pairs", len(pairs))

        # Split data
        rng = np.random.default_rng(cfg.seed)
        n_val = max(1, int(len(pairs) * cfg.val_fraction))
        perm = rng.permutation(len(pairs))
        val_pairs = [pairs[i] for i in perm[:n_val]]
        train_pairs = [pairs[i] for i in perm[n_val:]]
        logger.info("Train: %d, Val: %d", len(train_pairs), len(val_pairs))

        # Datasets
        train_ds = IPATranslationDataset(
            [p[0] for p in train_pairs],
            [p[1] for p in train_pairs],
            tokenizer, cfg.max_len,
        )
        val_ds = IPATranslationDataset(
            [p[0] for p in val_pairs],
            [p[1] for p in val_pairs],
            tokenizer, cfg.max_len,
        )
        train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=cfg.batch_size)

        # Optimizer + scheduler
        optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=0.01)
        total_steps = len(train_loader) * cfg.epochs // cfg.gradient_accumulation_steps
        warmup_steps = int(total_steps * cfg.warmup_fraction)

        def lr_lambda(step: int) -> float:
            if step < warmup_steps:
                return step / max(warmup_steps, 1)
            progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
            return 0.5 * (1.0 + math.cos(math.pi * progress))

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
        scaler = torch.amp.GradScaler(enabled=cfg.use_amp)

        # Training history
        output_dir = Path(cfg.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        history = TrainingHistory(output_dir)
        history.start_session(start_epoch=0, config=cfg)

        # Save config snapshot
        config_path = output_dir / "config.yaml"
        with open(config_path, "w") as f:
            yaml.dump(cfg.model_dump(), f, default_flow_style=False)

        # Training loop
        results: dict[str, float] = {}
        best_val_loss = float("inf")
        accum = cfg.gradient_accumulation_steps

        for epoch in range(cfg.epochs):
            model.train()
            total_loss = 0.0
            n_steps = 0
            optimizer.zero_grad()

            for batch_idx, batch in enumerate(train_loader):
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["labels"].to(device)

                with torch.amp.autocast(device_type=device.type, enabled=cfg.use_amp):
                    outputs = model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        labels=labels,
                    )
                    loss = outputs.loss / accum

                scaler.scale(loss).backward()

                if (batch_idx + 1) % accum == 0 or (batch_idx + 1) == len(train_loader):
                    scaler.unscale_(optimizer)
                    nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()
                    scheduler.step()

                total_loss += outputs.loss.item()
                n_steps += 1

                if batch_idx % 50 == 0:
                    logger.info(
                        "  epoch=%d step=%d loss=%.4f lr=%.2e",
                        epoch, batch_idx, outputs.loss.item(),
                        scheduler.get_last_lr()[0],
                    )

            avg_train_loss = total_loss / max(n_steps, 1)

            # Validation
            model.eval()
            total_val_loss = 0.0
            n_val_steps = 0

            with torch.no_grad():
                for batch in val_loader:
                    input_ids = batch["input_ids"].to(device)
                    attention_mask = batch["attention_mask"].to(device)
                    labels = batch["labels"].to(device)

                    with torch.amp.autocast(device_type=device.type, enabled=cfg.use_amp):
                        outputs = model(
                            input_ids=input_ids,
                            attention_mask=attention_mask,
                            labels=labels,
                        )
                    total_val_loss += outputs.loss.item()
                    n_val_steps += 1

            avg_val_loss = total_val_loss / max(n_val_steps, 1)
            logger.info(
                "Epoch %d: train_loss=%.4f val_loss=%.4f",
                epoch, avg_train_loss, avg_val_loss,
            )

            results[f"epoch_{epoch}_train_loss"] = avg_train_loss
            results[f"epoch_{epoch}_val_loss"] = avg_val_loss

            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss

            history.update_epoch(epoch + 1, best_val_loss)

        results["final_val_loss"] = best_val_loss

        # Save model + tokenizer
        model_path = output_dir / "model"
        model.save_pretrained(model_path)
        tokenizer.save_pretrained(model_path)
        logger.info("Saved model to %s", model_path)

        # Save results
        with open(output_dir / "results.json", "w") as f:
            json.dump(results, f, indent=2)

        history.end_session(end_epoch=cfg.epochs, best_val_loss=best_val_loss)
        return results

    @staticmethod
    def _load_pairs(path: str) -> list[tuple[str, str]]:
        """Load pairs from JSONL file."""
        pairs = []
        with open(path, encoding="utf-8") as f:
            for line in f:
                record = json.loads(line)
                pairs.append((record["ipa"], record["english"]))
        return pairs

    @staticmethod
    def _apply_lora(model, cfg: TranslatorConfig):
        """Wrap model with LoRA adapters via peft."""
        try:
            from peft import LoraConfig, get_peft_model
        except ImportError as e:
            raise ImportError(
                "peft is required for LoRA training. "
                "Install with: poetry install --with translator"
            ) from e

        lora_config = LoraConfig(
            r=cfg.lora_r,
            lora_alpha=cfg.lora_alpha,
            target_modules=cfg.lora_target_modules,
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, lora_config)
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in model.parameters())
        logger.info(
            "LoRA: trainable=%d (%.2f%% of %d)",
            trainable, 100 * trainable / total, total,
        )
        return model

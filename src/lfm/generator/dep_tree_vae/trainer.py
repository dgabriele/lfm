"""Training loop for DepTreeVAE."""

from __future__ import annotations

import logging
import math
import signal
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
import yaml
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from lfm.generator.dep_tree_vae.config import DepTreeVAEConfig
from lfm.generator.dep_tree_vae.data import build_dataloaders
from lfm.generator.dep_tree_vae.model import DepTreeVAE

logger = logging.getLogger(__name__)


def _cleanup_gpu():
    """Release CUDA memory on exit."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def train_dep_tree_vae(cfg: DepTreeVAEConfig) -> None:
    """Full training loop for DepTreeVAE."""
    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
    out_dir = Path(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Signal handler: save checkpoint + free GPU on SIGTERM/SIGINT
    _shutdown_requested = False

    def _handle_signal(signum, frame):
        nonlocal _shutdown_requested
        sig_name = signal.Signals(signum).name
        logger.info("Received %s — saving checkpoint and shutting down...", sig_name)
        _shutdown_requested = True

    signal.signal(signal.SIGTERM, _handle_signal)
    signal.signal(signal.SIGINT, _handle_signal)
    import atexit
    atexit.register(_cleanup_gpu)

    # Save config snapshot
    with open(out_dir / "config.yaml", "w") as f:
        yaml.dump(cfg.model_dump(), f, default_flow_style=False)

    # Data
    train_loader, val_loader, sp, vocab_size = build_dataloaders(cfg)

    # Model
    model = DepTreeVAE(cfg, vocab_size).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(
        "DepTreeVAE: %d params (%.1fM), %d trainable (%.1fM), device=%s",
        n_params, n_params / 1e6, n_trainable, n_trainable / 1e6, device,
    )

    # Optimizer (only trainable params)
    param_groups = model.trainable_parameters()
    optimizer = AdamW(param_groups, lr=cfg.lr, weight_decay=0.01)

    total_steps = len(train_loader) * cfg.num_epochs
    scheduler = CosineAnnealingLR(optimizer, T_max=total_steps, eta_min=cfg.lr_min)
    scaler = torch.amp.GradScaler("cuda", enabled=cfg.use_amp)

    # Resume
    global_step = 0
    start_epoch = 0
    best_val_loss = float("inf")
    resume_path = out_dir / "resume.pt"
    if resume_path.exists():
        ckpt = torch.load(resume_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state"])
        optimizer.load_state_dict(ckpt["optimizer_state"])
        scheduler.load_state_dict(ckpt["scheduler_state"])
        scaler.load_state_dict(ckpt["scaler_state"])
        global_step = ckpt["global_step"]
        start_epoch = ckpt["epoch"]
        best_val_loss = ckpt.get("best_val_loss", float("inf"))
        logger.info(
            "Resumed from epoch=%d step=%d best_val=%.4f",
            start_epoch, global_step, best_val_loss,
        )

    accum = cfg.gradient_accumulation_steps

    for epoch in range(start_epoch, cfg.num_epochs):
        model.train()
        optimizer.zero_grad()

        epoch_loss = 0.0
        epoch_recon = 0.0
        epoch_skel = 0.0
        epoch_kl = 0.0
        epoch_disent = 0.0
        n_batches = 0
        batch_start = time.time()

        for i, batch in enumerate(train_loader):
            if _shutdown_requested:
                logger.info("Shutdown requested — saving checkpoint at step %d", global_step)
                _save_checkpoint(model, optimizer, scheduler, scaler, epoch, global_step, best_val_loss, out_dir)
                _cleanup_gpu()
                logger.info("Clean shutdown complete.")
                return

            batch = {k: v.to(device) for k, v in batch.items()}

            kl_weight = _kl_schedule(global_step, cfg)

            with torch.amp.autocast(device_type=device.type, enabled=cfg.use_amp):
                out = model(
                    tokens=batch["tokens"],
                    lengths=batch["lengths"],
                    role_ids=batch["role_ids"],
                    role_lengths=batch["role_lengths"],
                    kl_weight=kl_weight,
                )
                loss = out.total_loss / accum

            scaler.scale(loss).backward()

            if (i + 1) % accum == 0:
                scaler.unscale_(optimizer)
                gnorm = nn.utils.clip_grad_norm_(
                    [p for g in param_groups for p in g["params"]], 5.0,
                ).item()
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                scheduler.step()
                global_step += 1

                # Logging
                if global_step % cfg.log_every == 0:
                    elapsed = time.time() - batch_start
                    lr = scheduler.get_last_lr()[0]
                    z_std_mean = (0.5 * out.logvar).exp().mean().item()
                    logger.info(
                        "ep%d step=%d  recon=%.3f skel=%.3f kl=%.3f "
                        "disent=%.3f (s=%.3f c=%.3f a=%.3f h=%.3f)  "
                        "z_std=%.4f gnorm=%.2f lr=%.6f  [%.1fs]",
                        epoch, global_step,
                        out.recon_loss.item(),
                        out.skeleton_loss.item(),
                        out.kl_loss.item(),
                        out.disentangle["total"].item(),
                        out.disentangle["struct_loss"].item(),
                        out.disentangle["content_loss"].item(),
                        out.disentangle["adversarial_loss"].item(),
                        out.disentangle["hsic_loss"].item(),
                        z_std_mean, gnorm, lr, elapsed,
                    )
                    batch_start = time.time()

                # Checkpoint
                if global_step % cfg.checkpoint_every_steps == 0:
                    _save_checkpoint(
                        model, optimizer, scheduler, scaler,
                        epoch, global_step, best_val_loss, out_dir,
                    )

            epoch_loss += out.total_loss.item()
            epoch_recon += out.recon_loss.item()
            epoch_skel += out.skeleton_loss.item()
            epoch_kl += out.kl_loss.item()
            epoch_disent += out.disentangle["total"].item()
            n_batches += 1

        # Epoch summary
        logger.info(
            "ep%d done — avg recon=%.4f skel=%.4f kl=%.4f disent=%.4f",
            epoch,
            epoch_recon / max(n_batches, 1),
            epoch_skel / max(n_batches, 1),
            epoch_kl / max(n_batches, 1),
            epoch_disent / max(n_batches, 1),
        )

        # Validation
        val_loss = _validate(model, val_loader, device, cfg, global_step)
        logger.info("ep%d val_loss=%.4f (best=%.4f)", epoch, val_loss, best_val_loss)

        _save_checkpoint(
            model, optimizer, scheduler, scaler,
            epoch + 1, global_step, best_val_loss, out_dir,
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), out_dir / "best.pt")
            logger.info("  New best val_loss=%.4f", val_loss)

    logger.info("Training complete. best_val_loss=%.4f", best_val_loss)


@torch.no_grad()
def _validate(
    model: DepTreeVAE,
    val_loader: torch.utils.data.DataLoader,
    device: torch.device,
    cfg: DepTreeVAEConfig,
    global_step: int,
) -> float:
    model.eval()
    total_loss = 0.0
    total_recon = 0.0
    total_skel = 0.0
    total_kl = 0.0
    total_disent = 0.0
    total_hsic = 0.0
    total_adv = 0.0
    n = 0
    for batch in val_loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        out = model(
            tokens=batch["tokens"],
            lengths=batch["lengths"],
            role_ids=batch["role_ids"],
            role_lengths=batch["role_lengths"],
            kl_weight=_kl_schedule(global_step, cfg),
        )
        total_loss += out.total_loss.item()
        total_recon += out.recon_loss.item()
        total_skel += out.skeleton_loss.item()
        total_kl += out.kl_loss.item()
        total_disent += out.disentangle["total"].item()
        total_hsic += out.disentangle["hsic_loss"].item()
        total_adv += out.disentangle["adversarial_loss"].item()
        n += 1
    model.train()
    nm = max(n, 1)
    logger.info(
        "  val breakdown: recon=%.4f skel=%.4f kl=%.4f "
        "disent=%.4f (a=%.4f h=%.4f)",
        total_recon / nm, total_skel / nm, total_kl / nm,
        total_disent / nm, total_adv / nm, total_hsic / nm,
    )
    return total_loss / nm


def _kl_schedule(step: int, cfg: DepTreeVAEConfig) -> float:
    if cfg.kl_weight <= 0:
        return 0.0
    warmup = max(cfg.kl_warmup_steps, 1)
    return cfg.kl_weight * min(step / warmup, 1.0)


def _save_checkpoint(
    model: DepTreeVAE,
    optimizer: torch.optim.Optimizer,
    scheduler: object,
    scaler: GradScaler,
    epoch: int,
    global_step: int,
    best_val_loss: float,
    out_dir: Path,
) -> None:
    ckpt = {
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict(),
        "scaler_state": scaler.state_dict(),
        "epoch": epoch,
        "global_step": global_step,
        "best_val_loss": best_val_loss,
    }
    torch.save(ckpt, out_dir / "resume.pt")
    logger.info("Checkpoint at step %d (epoch %d)", global_step, epoch)

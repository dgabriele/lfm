"""Training loop for DepTree Diffusion VAE.

Reuses patterns from the autoregressive DepTreeVAE trainer:
signal handling, AMP, checkpointing, logging, prior diagnostic.
"""

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

from lfm.generator.dep_tree_diffusion.config import DepTreeDiffusionConfig
from lfm.generator.dep_tree_diffusion.model import DepTreeDiffusionVAE

logger = logging.getLogger(__name__)


def _cleanup_gpu():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def train_dep_tree_diffusion(cfg: DepTreeDiffusionConfig) -> None:
    """Main training entry point."""
    device = torch.device(cfg.device)
    out_dir = Path(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

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

    with open(out_dir / "config.yaml", "w") as f:
        yaml.dump(cfg.model_dump(), f, default_flow_style=False)

    # Data — reuse the dep tree dataset + collation
    from lfm.generator.dep_tree_vae.data import build_dataloaders
    train_loader, val_loader, sp, vocab_size = build_dataloaders(cfg)

    # Model
    model = DepTreeDiffusionVAE(cfg, vocab_size).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    logger.info(
        "DepTreeDiffusionVAE: %d params (%.1fM), device=%s",
        n_params, n_params / 1e6, device,
    )

    param_groups = model.trainable_parameters()
    optimizer = AdamW(
        [{"params": g["params"], "lr": cfg.lr} for g in param_groups],
        weight_decay=0.01,
    )
    total_steps = cfg.num_epochs * (len(train_loader) // cfg.gradient_accumulation_steps)
    scheduler = CosineAnnealingLR(optimizer, T_max=total_steps, eta_min=cfg.lr_min)

    try:
        from torch.amp import GradScaler
    except ImportError:
        from torch.cuda.amp import GradScaler
    scaler = GradScaler(enabled=cfg.use_amp)

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
        start_epoch = ckpt.get("epoch", 0)
        global_step = ckpt.get("global_step", 0)
        best_val_loss = ckpt.get("best_val_loss", float("inf"))
        logger.info("Resumed from epoch=%d step=%d best_val=%.4f", start_epoch, global_step, best_val_loss)

    accum = cfg.gradient_accumulation_steps

    for epoch in range(start_epoch, cfg.num_epochs):
        model.train()
        epoch_loss = 0.0
        n_batches = 0
        batch_start = time.time()

        # Word dropout annealing
        if cfg.word_dropout > 0:
            frac = min(epoch / max(cfg.word_dropout_anneal_epochs, 1), 1.0)
            model._word_dropout_p = cfg.word_dropout + frac * (cfg.word_dropout_min - cfg.word_dropout)

        for i, batch in enumerate(train_loader):
            if _shutdown_requested:
                logger.info("Shutdown requested — saving checkpoint at step %d", global_step)
                _save_checkpoint(model, optimizer, scheduler, scaler, epoch, global_step, best_val_loss, out_dir)
                logger.info("Clean shutdown complete.")
                return

            batch = {k: v.to(device) for k, v in batch.items()}
            kl_weight = _kl_schedule(global_step, cfg)

            # Need dep_depths — for now, use uniform depth=1 as placeholder
            # TODO: add real dep tree depths to the dataset
            content_len = batch["tokens"].size(1)
            dep_depths = torch.ones(batch["tokens"].size(0), content_len, dtype=torch.long, device=device)

            with torch.amp.autocast(device_type=device.type, enabled=cfg.use_amp):
                out = model(
                    tokens=batch["tokens"],
                    lengths=batch["lengths"],
                    role_ids=batch["role_ids"],
                    role_lengths=batch["role_lengths"],
                    dep_depths=dep_depths,
                    kl_weight=kl_weight,
                )

                z_var_loss = torch.tensor(0.0, device=device)
                if cfg.z_var_weight > 0:
                    z = torch.cat([out.z_struct, out.z_content], dim=-1)
                    z_var_loss = cfg.z_var_weight * (z.var(dim=0) - cfg.z_var_target).pow(2).mean()

                loss = (out.total_loss + z_var_loss) / accum

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

                if global_step % cfg.log_every == 0:
                    elapsed = time.time() - batch_start
                    z_std = (0.5 * out.logvar).exp().mean().item()
                    logger.info(
                        "ep%d step=%d  recon=%.3f skel=%.3f kl=%.3f "
                        "disent=%.3f z_std=%.4f gnorm=%.2f lr=%.6f  [%.1fs]",
                        epoch, global_step,
                        out.recon_loss.item(), out.skeleton_loss.item(),
                        out.kl_loss.item(), out.disentangle["total"].item(),
                        z_std, gnorm, scheduler.get_last_lr()[0], elapsed,
                    )
                    batch_start = time.time()

                if global_step % cfg.checkpoint_every_steps == 0:
                    _save_checkpoint(model, optimizer, scheduler, scaler, epoch, global_step, best_val_loss, out_dir)
                    _prior_diagnostic(model, sp, device, cfg)

            epoch_loss += out.total_loss.item()
            n_batches += 1

        logger.info(
            "ep%d done — avg_loss=%.4f",
            epoch, epoch_loss / max(n_batches, 1),
        )

        _save_checkpoint(model, optimizer, scheduler, scaler, epoch + 1, global_step, best_val_loss, out_dir)

    logger.info("Training complete. best_val_loss=%.4f", best_val_loss)


def _kl_schedule(step: int, cfg: DepTreeDiffusionConfig) -> float:
    if cfg.kl_weight <= 0:
        return 0.0
    warmup = max(cfg.kl_warmup_steps, 1)
    return cfg.kl_weight * min(step / warmup, 1.0)


@torch.no_grad()
def _prior_diagnostic(
    model: DepTreeDiffusionVAE, sp, device: torch.device,
    cfg: DepTreeDiffusionConfig, n_samples: int = 32,
) -> None:
    """Sample z ~ N(0,1), generate via diffusion, log quality metrics."""
    from lfm.generator.dep_tree_vae.config import DEP_RELATIONS
    from lfm.generator.dep_tree_vae.skeleton import SKEL_BOS, SKEL_EOS

    model.eval()
    z = torch.randn(n_samples, cfg.latent.total_dim, device=device)
    z_struct, z_content = model.latent.split(z)

    skel_tokens = model.skeleton_decoder(z_struct)[0]

    surfaces = []
    lengths = []

    for i in range(n_samples):
        roles = []
        for t in skel_tokens[i]:
            v = t.item()
            if v == SKEL_BOS:
                continue
            if v == SKEL_EOS:
                break
            if v < len(DEP_RELATIONS):
                roles.append(v)
        if not roles:
            roles = [DEP_RELATIONS.index("root")]

        num_roles = len(roles)
        seq_len = num_roles * cfg.diffusion.max_tokens_per_role

        role_ids = torch.tensor(roles, device=device).unsqueeze(0)
        role_mask = torch.ones(1, num_roles, dtype=torch.bool, device=device)
        memory = model.phrase_projector(z_content[i:i+1], role_ids, role_mask)

        # Expand role_ids to per-token
        per_tok_roles = role_ids.repeat_interleave(cfg.diffusion.max_tokens_per_role, dim=1)
        # Depth = role index (simple proxy for tree depth)
        per_tok_depths = torch.arange(num_roles, device=device).repeat_interleave(
            cfg.diffusion.max_tokens_per_role,
        ).unsqueeze(0)

        tok_ids = model.diffusion_decoder.sample(
            seq_len, per_tok_roles, per_tok_depths, memory,
            num_steps=cfg.diffusion.num_diffusion_steps,
            depth_scale=cfg.diffusion.depth_scale,
        )

        ids = tok_ids[0].tolist()
        ids = [t for t in ids if 0 < t < sp.GetPieceSize()]
        surface = sp.DecodeIds(ids)
        surfaces.append(surface)
        lengths.append(len(ids))

    avg_len = sum(lengths) / len(lengths)
    len_std = (sum((l - avg_len)**2 for l in lengths) / len(lengths)) ** 0.5

    logger.info(
        "PRIOR DIAGNOSTIC (diffusion): avg_len=%.1f±%.1f (n=%d)",
        avg_len, len_std, n_samples,
    )
    for j, s in enumerate(surfaces[:4]):
        logger.info("  prior[%d]: %s", j, s[:120])

    model.train()


def _save_checkpoint(model, optimizer, scheduler, scaler, epoch, global_step, best_val_loss, out_dir):
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

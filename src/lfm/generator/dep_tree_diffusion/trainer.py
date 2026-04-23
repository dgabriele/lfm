"""Training loop for DepTree Diffusion VAE.

Reuses patterns from the autoregressive DepTreeVAE trainer:
signal handling, AMP, checkpointing, logging, prior diagnostic.
"""

from __future__ import annotations

import logging
import math
import os
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

    # Data — extended pipeline with per-token tree depths
    from lfm.generator.dep_tree_diffusion.data import build_diffusion_dataloaders
    train_loader, val_loader, sp, vocab_size = build_diffusion_dataloaders(cfg)

    # Completeness scorer (frozen, for auxiliary loss)
    completeness_scorer = None
    if cfg.completeness_scorer_path:
        from lfm.generator.completeness_scorer.model import CompletenessScorer, CompletenessConfig
        scorer_ckpt = torch.load(cfg.completeness_scorer_path, map_location=device, weights_only=False)
        scorer_cfg = CompletenessConfig(**scorer_ckpt["config"])
        completeness_scorer = CompletenessScorer(scorer_cfg).to(device)
        completeness_scorer.load_state_dict(scorer_ckpt["model_state"])
        completeness_scorer.eval()
        for p in completeness_scorer.parameters():
            p.requires_grad = False
        logger.info("Loaded frozen completeness scorer (val_acc=%.1f%%)", scorer_ckpt.get("val_acc", 0) * 100)

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
        missing, unexpected = model.load_state_dict(ckpt["model_state"], strict=False)
        if missing:
            logger.info("New parameters (randomly initialized): %s", missing)
        try:
            optimizer.load_state_dict(ckpt["optimizer_state"])
        except (ValueError, RuntimeError) as e:
            logger.warning("Optimizer state incompatible (new params?), resetting: %s", e)
        # Update optimizer LR to match config (may have changed)
        for pg in optimizer.param_groups:
            pg["lr"] = cfg.lr
        scheduler = CosineAnnealingLR(optimizer, T_max=total_steps, eta_min=cfg.lr_min)
        try:
            scaler.load_state_dict(ckpt["scaler_state"])
        except RuntimeError:
            pass
        start_epoch = ckpt.get("epoch", 0)
        global_step = ckpt.get("global_step", 0)
        best_val_loss = ckpt.get("best_val_loss", float("inf"))
        logger.info("Resumed from epoch=%d step=%d best_val=%.4f", start_epoch, global_step, best_val_loss)

    # torch.compile — enable via env var COMPILE_DECODER=1 on images with gcc
    if os.environ.get("COMPILE_DECODER") == "1" and hasattr(torch, "compile"):
        try:
            model.diffusion_decoder = torch.compile(model.diffusion_decoder)
            logger.info("torch.compile applied to diffusion decoder")
        except Exception as e:
            logger.warning("torch.compile failed: %s", e)

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

            with torch.amp.autocast(device_type=device.type, enabled=cfg.use_amp):
                out = model(
                    tokens=batch["tokens"],
                    lengths=batch["lengths"],
                    depths=batch["depths"],
                    role_ids=batch["role_ids"],
                    role_lengths=batch["role_lengths"],
                    kl_weight=kl_weight,
                    role_token_counts=batch["role_token_counts"],
                    global_step=global_step,
                )

                z_var_loss = torch.tensor(0.0, device=device)
                if cfg.z_var_weight > 0:
                    z = torch.cat([out.z_struct, out.z_content], dim=-1)
                    z_var_loss = cfg.z_var_weight * (z.var(dim=0) - cfg.z_var_target).pow(2).mean()

                # Completeness loss: frozen scorer on decoder's soft-token output.
                # Rewards structurally complete thoughts, penalizes word salad/loops.
                completeness_loss = torch.tensor(0.0, device=device)
                if completeness_scorer is not None and cfg.completeness_weight > 0:
                    # Get decoder logits from a low-noise forward pass (t≈0.1)
                    # so the output is close to what generation produces.
                    with torch.amp.autocast(device_type=device.type, enabled=False):
                        t_low = torch.full((batch["tokens"].size(0),), 0.1, device=device)
                        depths_b = batch["depths"]
                        x0 = model.diffusion_decoder.token_embedding(
                            batch["tokens"].clamp(max=model.diffusion_decoder.token_embedding.num_embeddings - 1)
                        )
                        x_t, _ = model.diffusion_decoder.add_noise(x0, t_low.unsqueeze(1).expand_as(depths_b))
                        per_token_roles = model._extract_per_token_roles(batch["tokens"], batch["lengths"])
                        z_mem = model._z_to_memory(torch.cat([out.z_struct, out.z_content], dim=-1))
                        padding = torch.arange(batch["tokens"].size(1), device=device).unsqueeze(0) >= batch["lengths"].unsqueeze(1)
                        x0_pred = model.diffusion_decoder(
                            x_t, t_low.unsqueeze(1).expand_as(depths_b),
                            per_token_roles, depths_b, z_mem, padding,
                        )
                        logits = model.diffusion_decoder.output_head(x0_pred)
                        # Truncate to scorer's vocab size (SPM tokens only)
                        scorer_vocab = completeness_scorer.cfg.vocab_size
                        scores = completeness_scorer.score_soft(logits[:, :, :scorer_vocab].float(), batch["lengths"])
                        # Maximize score (complete thoughts) → minimize -score
                        completeness_loss = -cfg.completeness_weight * scores.mean()

                loss = (out.total_loss + z_var_loss + completeness_loss) / accum

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
                    topo_str = f" topo={out.topo_rho.item():.3f}" if out.topo_rho.item() != 0 else ""
                    logger.info(
                        "ep%d step=%d  recon=%.3f kl=%.3f%s "
                        "z_std=%.4f gnorm=%.2f lr=%.6f  [%.1fs]",
                        epoch, global_step,
                        out.recon_loss.item(), out.kl_loss.item(),
                        topo_str, z_std, gnorm, scheduler.get_last_lr()[0], elapsed,
                    )
                    batch_start = time.time()

                if global_step % cfg.checkpoint_every_steps == 0:
                    _save_checkpoint(model, optimizer, scheduler, scaler, epoch, global_step, best_val_loss, out_dir)
                    torch.cuda.empty_cache()
                    try:
                        _prior_diagnostic(model, sp, device, cfg)
                        _downstream_diagnostic(model, sp, device, cfg)
                    except RuntimeError as e:
                        if "out of memory" in str(e):
                            logger.warning("OOM during diagnostics — skipping (step=%d)", global_step)
                            torch.cuda.empty_cache()
                        else:
                            raise

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


_diag_loader = None


@torch.no_grad()
def _prior_diagnostic(
    model: DepTreeDiffusionVAE, sp, device: torch.device,
    cfg: DepTreeDiffusionConfig, n_samples: int = 16,
) -> None:
    """Sample z ~ N(0,1) but use real skeletons/depths from training data.

    The diffusion decoder generates content conditioned on real dependency
    structures. This tests whether prior z produces coherent content for
    realistic sentence frames.
    """
    global _diag_loader
    from lfm.generator.dep_tree_diffusion.data import (
        DiffusionDepTreeDataset, collate_diffusion,
    )
    from pathlib import Path as _P

    model.eval()

    # Lazy-load a small diagnostic dataloader
    if _diag_loader is None:
        cache_dir = _P(cfg.dataset_path) / "diffusion_cache"
        ds = DiffusionDepTreeDataset(cache_dir)
        _diag_loader = torch.utils.data.DataLoader(
            ds, batch_size=n_samples, shuffle=True,
            collate_fn=collate_diffusion, num_workers=0,
        )

    # Get real skeletons/depths from training data
    batch = next(iter(_diag_loader))
    real_tokens = batch["tokens"].to(device)
    real_depths = batch["depths"].to(device)
    real_lengths = batch["lengths"].to(device)
    b = min(real_tokens.size(0), n_samples)

    # Extract per-token roles from real data
    per_token_roles = model._extract_per_token_roles(real_tokens[:b], real_lengths[:b])

    # Sample z from prior, project to memory
    z = torch.randn(b, cfg.latent.total_dim, device=device)
    z_memory = model._z_to_memory(z)

    # Generate via diffusion using real structure + prior z
    seq_len = real_tokens.size(1)
    tok_ids = model.diffusion_decoder.sample(
        seq_len, per_token_roles[:b], real_depths[:b], z_memory,
        num_steps=cfg.diffusion.num_diffusion_steps,
        depth_scale=cfg.diffusion.depth_scale,
        min_noise=cfg.diffusion.min_noise,
        role_offset=model._role_offset,
        invert_depth_noise=cfg.diffusion.invert_depth_noise,
        ref_tokens=real_tokens[:b],
    )

    surfaces = []
    lengths = []
    spm_size = sp.get_piece_size()
    for i in range(b):
        n_tok = real_lengths[i].item()
        ids = tok_ids[i, :n_tok].tolist()
        ids = [t for t in ids if 0 < t < spm_size]
        surfaces.append(sp.DecodeIds(ids))
        lengths.append(len(ids))

    avg_len = sum(lengths) / max(len(lengths), 1)
    len_std = (sum((l - avg_len)**2 for l in lengths) / max(len(lengths), 1)) ** 0.5

    logger.info(
        "PRIOR DIAGNOSTIC (diffusion): avg_len=%.1f±%.1f (n=%d)",
        avg_len, len_std, b,
    )
    for j, s in enumerate(surfaces[:4]):
        logger.info("  prior[%d]: %s", j, s[:120])

    model.train()


@torch.no_grad()
def _downstream_diagnostic(
    model: DepTreeDiffusionVAE, sp, device: torch.device,
    cfg: DepTreeDiffusionConfig, n_pairs: int = 8,
) -> None:
    """Test downstream viability: interpolation smoothness + discrimination.

    1. Encode pairs of real sentences, interpolate z, decode at midpoint.
       Measure: do interpolated outputs differ from endpoints?
    2. Encode a batch, decode each, measure: do different z produce
       distinguishably different output? (pairwise output diversity)
    3. Log topology ρ: correlation between z-distances and output-distances.
    """
    global _diag_loader
    from lfm.generator.dep_tree_diffusion.data import (
        DiffusionDepTreeDataset, collate_diffusion,
    )
    from pathlib import Path as _P
    import numpy as np

    model.eval()

    if _diag_loader is None:
        cache_dir = _P(cfg.dataset_path) / "diffusion_cache"
        ds = DiffusionDepTreeDataset(cache_dir)
        _diag_loader = torch.utils.data.DataLoader(
            ds, batch_size=n_pairs * 2, shuffle=True,
            collate_fn=collate_diffusion, num_workers=0,
        )

    batch = next(iter(_diag_loader))
    tokens = batch["tokens"].to(device)
    lengths = batch["lengths"].to(device)
    depths = batch["depths"].to(device)
    b = tokens.size(0)

    # Encode all
    mu, _ = model.encoder(tokens, lengths)
    per_token_roles = model._extract_per_token_roles(tokens, lengths)

    # Decode each z
    def _decode_z(z, ref_idx=0):
        z_mem = model._z_to_memory(z.unsqueeze(0))
        tok = model.diffusion_decoder.sample(
            tokens[ref_idx:ref_idx+1].size(1),
            per_token_roles[ref_idx:ref_idx+1],
            depths[ref_idx:ref_idx+1],
            z_mem,
            num_steps=cfg.diffusion.num_diffusion_steps,
            depth_scale=cfg.diffusion.depth_scale,
            min_noise=cfg.diffusion.min_noise,
            role_offset=model._role_offset,
            ref_tokens=tokens[ref_idx:ref_idx+1],
        )
        ids = [int(t) for t in tok[0].tolist() if 0 < t < sp.GetPieceSize()]
        return sp.DecodeIds(ids)

    from lfm.translator.romanize import respell
    struct_dim = cfg.latent.struct_dim

    logger.info("DOWNSTREAM DIAGNOSTIC:")

    # 1a. Posterior interpolation — between encoded real sentences
    logger.info("  POSTERIOR INTERPOLATION (encoded data centroids):")
    for i in range(min(2, n_pairs)):
        a_idx, b_idx = i * 2, i * 2 + 1
        z_a, z_b = mu[a_idx], mu[b_idx]

        z_a_struct, z_a_content = z_a[:struct_dim], z_a[struct_dim:]
        z_b_struct, z_b_content = z_b[:struct_dim], z_b[struct_dim:]

        # Structure interpolation (content held constant from A)
        z_start = z_a
        z_mid = torch.cat([0.5 * z_a_struct + 0.5 * z_b_struct, z_a_content])
        z_end = torch.cat([z_b_struct, z_a_content])

        logger.info("  struct_interp[%d] A:   %s", i, respell(_decode_z(z_start, a_idx)))
        logger.info("  struct_interp[%d] mid: %s", i, respell(_decode_z(z_mid, a_idx)))
        logger.info("  struct_interp[%d] B:   %s", i, respell(_decode_z(z_end, a_idx)))

        # Content interpolation (structure held constant from A)
        z_start = z_a
        z_mid = torch.cat([z_a_struct, 0.5 * z_a_content + 0.5 * z_b_content])
        z_end = torch.cat([z_a_struct, z_b_content])

        logger.info("  content_interp[%d] A:   %s", i, respell(_decode_z(z_start, a_idx)))
        logger.info("  content_interp[%d] mid: %s", i, respell(_decode_z(z_mid, a_idx)))
        logger.info("  content_interp[%d] B:   %s", i, respell(_decode_z(z_end, a_idx)))

    # 1b. Prior-region interpolation — z sampled from N(0, 0.8²)
    # Tests what game agents would see: z from the prior, not encoder posteriors.
    logger.info("  PRIOR-REGION INTERPOLATION (z ~ N(0, 0.64)):")
    z_dim = cfg.latent.total_dim
    for i in range(2):
        z_a_prior = torch.randn(z_dim, device=device) * 0.8
        z_b_prior = torch.randn(z_dim, device=device) * 0.8
        z_mid_prior = 0.5 * z_a_prior + 0.5 * z_b_prior
        ref = 0
        logger.info("  prior_interp[%d] A:   %s", i, respell(_decode_z(z_a_prior, ref)))
        logger.info("  prior_interp[%d] mid: %s", i, respell(_decode_z(z_mid_prior, ref)))
        logger.info("  prior_interp[%d] B:   %s", i, respell(_decode_z(z_b_prior, ref)))

    # 2. Discrimination: decode N different z values, count unique outputs
    decoded_set = set()
    for i in range(min(b, 16)):
        text = _decode_z(mu[i], i)
        decoded_set.add(text)
    unique_ratio = len(decoded_set) / min(b, 16)
    logger.info("  discrimination: %d/%d unique decoded (%.0f%%)",
                len(decoded_set), min(b, 16), unique_ratio * 100)

    # 3. Topology: z-distance vs output-distance correlation
    # Use mean logits as output representation (faster than full decode)
    z_all = mu[:min(b, 16)]
    decoded_reprs = []
    for i in range(z_all.size(0)):
        z_mem = model._z_to_memory(z_all[i:i+1])
        t_low = torch.full((1, tokens.size(1)), 0.1, device=device)
        x0 = model.diffusion_decoder.token_embedding(
            tokens[i:i+1].clamp(max=model.diffusion_decoder.token_embedding.num_embeddings - 1)
        )
        x_t, _ = model.diffusion_decoder.add_noise(x0, t_low)
        pad = torch.arange(tokens.size(1), device=device).unsqueeze(0) >= lengths[i:i+1].unsqueeze(1)
        x0_p = model.diffusion_decoder(
            x_t, t_low, per_token_roles[i:i+1], depths[i:i+1], z_mem, pad,
        )
        valid = (~pad).unsqueeze(-1).float()
        pooled = (x0_p * valid).sum(dim=1) / valid.sum(dim=1).clamp(min=1)
        decoded_reprs.append(pooled)

    decoded_reprs = torch.cat(decoded_reprs, dim=0)
    idx = torch.triu_indices(z_all.size(0), z_all.size(0), offset=1, device=device)
    z_d = torch.cdist(z_all, z_all)[idx[0], idx[1]]
    o_d = torch.cdist(decoded_reprs, decoded_reprs)[idx[0], idx[1]]
    if z_d.numel() > 2:
        zc = z_d - z_d.mean()
        oc = o_d - o_d.mean()
        rho = (zc * oc).sum() / (zc.norm() * oc.norm()).clamp(min=1e-8)
        logger.info("  topology ρ=%.4f (z-dist vs output-dist correlation)", rho.item())
    else:
        logger.info("  topology: insufficient samples")

    model.train()


def _save_checkpoint(model, optimizer, scheduler, scaler, epoch, global_step, best_val_loss, out_dir):
    # Strip _orig_mod. prefix from torch.compile wrapped modules
    state = model.state_dict()
    state = {k.replace("_orig_mod.", ""): v for k, v in state.items()}
    ckpt = {
        "model_state": state,
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict(),
        "scaler_state": scaler.state_dict(),
        "epoch": epoch,
        "global_step": global_step,
        "best_val_loss": best_val_loss,
    }
    torch.save(ckpt, out_dir / "resume.pt")
    logger.info("Checkpoint at step %d (epoch %d)", global_step, epoch)

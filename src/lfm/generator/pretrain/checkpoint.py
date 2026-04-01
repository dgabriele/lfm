"""Checkpoint save/load and file hashing for VAE pretraining."""

from __future__ import annotations

import hashlib
import logging
import math
from pathlib import Path

import torch
from torch import nn

from .config import VAEPretrainConfig

logger = logging.getLogger(__name__)


def _file_hash(path: str | Path) -> str:
    """SHA-256 of a file for consistency checks."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


def save_best_checkpoint(
    cfg: VAEPretrainConfig,
    modules: dict[str, nn.Module],
    *,
    vocab_size: int,
    train_loss: float,
    val_loss: float,
    z_running_mean: torch.Tensor,
    z_running_std: torch.Tensor,
    spm_path: str,
) -> None:
    """Save the decoder-only checkpoint (for inference)."""
    output_path = Path(cfg.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    ckpt = {
        "latent_dim": cfg.latent_dim,
        "vocab_size": vocab_size,
        "decoder_hidden_dim": cfg.decoder_hidden_dim,
        "decoder_num_layers": cfg.decoder_num_layers,
        "decoder_num_heads": cfg.decoder_num_heads,
        "max_seq_len": cfg.max_seq_len,
        "num_memory_tokens": getattr(cfg, "num_memory_tokens", 1),
        "encoder_num_layers": getattr(cfg, "encoder_num_layers", 2),
        "attention_head_windows": list(cfg.attention_head_windows),
        "attention_global_every": cfg.attention_global_every,
        "use_rope": getattr(cfg, "use_rope", True),
        "share_decoder_layers": getattr(cfg, "share_decoder_layers", True),
        "encoder_pooling": getattr(cfg, "encoder_pooling", "mean"),
        "latent_to_decoder": modules["latent_to_decoder"].state_dict(),
        "token_embedding": modules["dec_token_embedding"].state_dict(),
        "pos_embedding": modules["dec_pos_embedding"].state_dict(),
        "decoder": modules["decoder"].state_dict(),
        "output_head": modules["output_head"].state_dict(),
        # Latent calibration statistics — used at agent
        # time to keep projected z in-distribution.
        "z_mean": z_running_mean.cpu(),
        "z_std": z_running_std.cpu(),
        "train_loss": train_loss,
        "val_loss": val_loss,
        "spm_hash": _file_hash(spm_path),
        **({"residual_vq": modules["_residual_vq"].state_dict(),
            "use_vq": True,
            "vq_num_levels": cfg.vq_num_levels,
            "vq_codebook_size": cfg.vq_codebook_size}
           if modules.get("_residual_vq") is not None else {}),
    }
    torch.save(ckpt, output_path)
    logger.info("Saved best decoder checkpoint to %s", output_path)


def save_resume_checkpoint(
    output_dir: str,
    *,
    epoch: int,
    global_step: int,
    best_val_loss: float,
    spm_path: str,
    z_running_mean: torch.Tensor,
    z_running_std: torch.Tensor,
    modules: dict[str, nn.Module],
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    scaler: torch.amp.GradScaler,
    contrastive_proj: nn.Module | None = None,
    cfg: object | None = None,
) -> None:
    """Save full training state for resume (every epoch)."""
    resume_path = Path(output_dir) / "vae_resume.pt"
    _ckpt = {
        "epoch": epoch,
        "global_step": global_step,
        "best_val_loss": best_val_loss,
        "spm_hash": _file_hash(spm_path),
        "z_mean": z_running_mean.cpu(),
        "z_std": z_running_std.cpu(),
        "modules": {
            k: m.state_dict()
            for k, m in modules.items()
            if isinstance(m, nn.Module)
        },
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "scaler": scaler.state_dict(),
    }
    if contrastive_proj is not None:
        _ckpt["contrastive_proj"] = contrastive_proj.state_dict()
    if modules.get("_residual_vq") is not None:
        _ckpt["residual_vq"] = modules["_residual_vq"].state_dict()
    # Architecture metadata for checkpoint consumers (viz, agent games)
    if cfg is not None:
        _ckpt["num_memory_tokens"] = getattr(cfg, "num_memory_tokens", 1)
        _ckpt["encoder_num_layers"] = getattr(cfg, "encoder_num_layers", 2)
        _ckpt["attention_head_windows"] = list(getattr(cfg, "attention_head_windows", [3,3,7,7,15,15,0,0]))
        _ckpt["attention_global_every"] = getattr(cfg, "attention_global_every", 7)
        _ckpt["use_rope"] = getattr(cfg, "use_rope", True)
        _ckpt["share_decoder_layers"] = getattr(cfg, "share_decoder_layers", True)
        _ckpt["encoder_pooling"] = getattr(cfg, "encoder_pooling", "mean")
        _ckpt["decoder_hidden_dim"] = getattr(cfg, "decoder_hidden_dim", 512)
        _ckpt["latent_dim"] = getattr(cfg, "latent_dim", 256)
    torch.save(_ckpt, resume_path)


def load_resume_checkpoint(
    resume_path: Path,
    *,
    device: torch.device,
    modules: dict[str, nn.Module],
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    current_spm_hash: str,
    contrastive_proj: nn.Module | None = None,
) -> tuple[int, int, float]:
    """Load checkpoint for training resume.

    Returns:
        Tuple of ``(start_epoch, global_step, best_val_loss)``.
    """
    logger.info("Resuming from %s", resume_path)
    ckpt = torch.load(resume_path, map_location=device, weights_only=False)
    # Verify SPM consistency
    ckpt_spm_hash = ckpt.get("spm_hash")
    if ckpt_spm_hash and ckpt_spm_hash != current_spm_hash:
        raise RuntimeError(
            f"SPM model mismatch: checkpoint was trained with "
            f"spm_hash={ckpt_spm_hash} but current spm.model has "
            f"hash={current_spm_hash}. Delete vae_resume.pt to "
            f"start fresh, or restore the matching spm.model."
        )
    for k, m in modules.items():
        if isinstance(m, nn.Module) and k in ckpt["modules"]:
            m.load_state_dict(ckpt["modules"][k])
    if contrastive_proj is not None and "contrastive_proj" in ckpt:
        contrastive_proj.load_state_dict(ckpt["contrastive_proj"])
        logger.info("Restored contrastive projection from checkpoint")
    # Check for gradient explosion in saved optimizer state.
    # If the last gnorm was extreme, skip optimizer restore
    # and start with fresh Adam momentum (model weights are fine).
    _saved_gnorm_ok = True
    if "last_grad_norm" in ckpt and not math.isfinite(ckpt["last_grad_norm"]):
        _saved_gnorm_ok = False
    # Also detect via NaN in optimizer state tensors
    if _saved_gnorm_ok:
        for group in ckpt["optimizer"]["state"].values():
            for v in group.values():
                if isinstance(v, torch.Tensor) and torch.isnan(v).any():
                    _saved_gnorm_ok = False
                    break
            if not _saved_gnorm_ok:
                break

    if _saved_gnorm_ok:
        try:
            optimizer.load_state_dict(ckpt["optimizer"])
        except (ValueError, RuntimeError) as e:
            logger.warning(
                "Optimizer state mismatch (likely new parameters added): %s "
                "— using fresh optimizer (model weights preserved)", e,
            )
    else:
        logger.warning(
            "Contaminated optimizer state detected — "
            "using fresh optimizer (model weights preserved)"
        )
    # Skip restoring scaler state — use fresh low-scale init
    # to avoid fp16 overflow from high saved scales.
    # scaler.load_state_dict(ckpt["scaler"])
    if "scheduler" in ckpt:
        try:
            scheduler.load_state_dict(ckpt["scheduler"])
        except Exception:
            logger.warning("Could not restore scheduler state — reinitializing")
    if modules.get("_residual_vq") is not None and "residual_vq" in ckpt:
        modules["_residual_vq"].load_state_dict(ckpt["residual_vq"])
        logger.info("Restored ResidualVQ codebooks from checkpoint")
    start_epoch = ckpt["epoch"]
    global_step = ckpt["global_step"]
    best_val_loss = ckpt["best_val_loss"]
    logger.info(
        "Resumed at epoch %d, step %d, best_val=%.4f",
        start_epoch, global_step, best_val_loss,
    )
    return start_epoch, global_step, best_val_loss

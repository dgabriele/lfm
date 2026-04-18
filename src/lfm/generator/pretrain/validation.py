"""Validation loop and epoch summary logging for VAE pretraining."""

from __future__ import annotations

import logging
import math

import torch
import torch.nn.functional as F  # noqa: N812
from torch import Tensor, nn

from .config import VAEPretrainConfig
from .forward import _vae_forward

logger = logging.getLogger(__name__)


class ValidationResult:
    """Container for validation pass results."""

    __slots__ = (
        "val_ce", "val_kl", "val_loss", "all_kl_per_dim",
        "val_bucket_ce_sum", "val_bucket_ce_count", "val_count",
    )

    def __init__(
        self,
        val_ce: float,
        val_kl: float,
        val_loss: float,
        all_kl_per_dim: list[Tensor],
        val_bucket_ce_sum: list[float],
        val_bucket_ce_count: list[int],
        val_count: int,
    ) -> None:
        self.val_ce = val_ce
        self.val_kl = val_kl
        self.val_loss = val_loss
        self.all_kl_per_dim = all_kl_per_dim
        self.val_bucket_ce_sum = val_bucket_ce_sum
        self.val_bucket_ce_count = val_bucket_ce_count
        self.val_count = val_count


def run_validation(
    *,
    cfg: VAEPretrainConfig,
    modules: dict[str, nn.Module],
    val_loader: object,
    device: torch.device,
    full_vocab: int,
    bos_id: int,
    do_kl: bool,
) -> ValidationResult:
    """Run the full validation pass over val_loader."""
    for m in modules.values():
        if isinstance(m, nn.Module):
            m.eval()

    val_ce_sum = 0.0
    val_kl_sum = 0.0
    val_count = 0
    all_kl_per_dim: list[Tensor] = []
    _val_bucket_ce_sum = [0.0, 0.0, 0.0]
    _val_bucket_ce_count = [0, 0, 0]

    with torch.no_grad():
        for batch_tokens, batch_lengths in val_loader:
            batch_tokens = batch_tokens.to(device)
            batch_lengths = torch.as_tensor(
                batch_lengths, device=device
            )
            b = batch_tokens.size(0)

            with torch.amp.autocast(
                device_type=device.type, enabled=cfg.use_amp
            ):
                ce_loss, kl_loss, kl_per_dim, _, dec_hidden_val, _, _, _, _, _ = _vae_forward(
                    batch_tokens,
                    batch_lengths,
                    bos_id=bos_id,
                    full_vocab=full_vocab,
                    kl_free_bits=0.0,
                    compute_kl=do_kl,
                    **modules,
                )

            val_ce_sum += ce_loss.item() * b
            val_kl_sum += kl_loss.item() * b
            val_count += b
            all_kl_per_dim.append(kl_per_dim.detach().cpu())

            # Bucketed val CE
            _v_logits = modules["output_head"](dec_hidden_val)
            _v_src_mask = (
                torch.arange(batch_tokens.size(1), device=device).unsqueeze(0)
                < batch_lengths.unsqueeze(1)
            )
            _v_per_tok = F.cross_entropy(
                _v_logits.reshape(-1, full_vocab),
                batch_tokens.reshape(-1),
                reduction="none",
            ).reshape(b, -1)
            _v_per_sample = (
                (_v_per_tok * _v_src_mask.float()).sum(dim=1)
                / batch_lengths.float().clamp(min=1)
            )
            for _vi in range(b):
                _vlen = batch_lengths[_vi].item()
                _vbkt = 0 if _vlen < 20 else (1 if _vlen <= 50 else 2)
                _val_bucket_ce_sum[_vbkt] += _v_per_sample[_vi].item()
                _val_bucket_ce_count[_vbkt] += 1

    val_ce = val_ce_sum / max(val_count, 1)
    val_kl = val_kl_sum / max(val_count, 1)
    val_loss = val_ce + cfg.kl_weight * val_kl

    return ValidationResult(
        val_ce=val_ce,
        val_kl=val_kl,
        val_loss=val_loss,
        all_kl_per_dim=all_kl_per_dim,
        val_bucket_ce_sum=_val_bucket_ce_sum,
        val_bucket_ce_count=_val_bucket_ce_count,
        val_count=val_count,
    )


def log_epoch_summary(
    *,
    epoch: int,
    cfg: VAEPretrainConfig,
    epoch_time: float,
    train_ce: float,
    train_kl: float,
    train_zvar: float,
    train_dip: float,
    train_cl: float,
    train_klb: float,
    train_bow: float,
    train_vq: float,
    train_loss: float,
    train_acc: float,
    val: ValidationResult,
    z_running_std: Tensor,
    ss_p: float,
    wd_p: float,
    do_kl: bool,
    use_contrastive: bool,
    modules: dict[str, nn.Module],
    scheduler: object,
    bucket_ce_sum: list[float],
    bucket_ce_count: list[int],
    train_count: int,
) -> None:
    """Log the epoch summary and bucketed CE statistics."""
    _BUCKET_NAMES = ["short(<20)", "med(20-50)", "long(>50)"]

    # Build epoch summary with only active components
    epoch_parts = [f"Epoch {epoch + 1}/{cfg.num_epochs} ({epoch_time:.0f}s)"]
    epoch_parts.append(f"train: CE={train_ce:.4f}")
    if do_kl:
        epoch_parts.append(f"KL={train_kl:.4f}")
    if cfg.z_var_weight > 0:
        epoch_parts.append(f"zvar={train_zvar:.4f}")
    if cfg.dip_weight > 0:
        epoch_parts.append(f"dip={train_dip:.6f}")
    if use_contrastive:
        epoch_parts.append(f"CL={train_cl:.4f}")
    if cfg.kl_beta > 0:
        epoch_parts.append(f"KLβ={train_klb:.4f}")
    if cfg.bow_weight > 0:
        epoch_parts.append(f"BoW={train_bow:.4f}")
    if cfg.use_vq:
        epoch_parts.append(f"VQ={train_vq:.4f}")
        rvq = modules["_residual_vq"]
        util = rvq.utilization
        util_str = "/".join(f"{u:.0%}" for u in util)
        epoch_parts.append(f"cb_util={util_str}")
        # Reset dead codes by splitting high-usage codes
        resets = rvq.reset_dead_codes(threshold=1.0, epsilon=0.01)
        if any(r > 0 for r in resets):
            reset_str = "/".join(str(r) for r in resets)
            logger.info("  Reset %s dead codes (by splitting)", reset_str)
        rvq.reset_usage()
    epoch_parts.append(f"acc={train_acc:.1%}")
    epoch_parts.append(f"total={train_loss:.4f}")
    epoch_parts.append(f"| val: CE={val.val_ce:.4f}")
    if do_kl:
        epoch_parts.append(f"KL={val.val_kl:.4f}")
    epoch_parts.append(f"total={val.val_loss:.4f}")
    # Log z distribution health + LR
    current_lr = scheduler.get_last_lr()[0]
    epoch_parts.append(
        f"| z_std={z_running_std.mean():.4f}"
        f" z_active(>{z_running_std.mean().item() * 0.5:.4f})="
        f"{int((z_running_std > z_running_std.mean() * 0.5).sum())}/{cfg.latent_dim}"
        f" lr={current_lr:.6f}"
    )
    if ss_p > 0:
        epoch_parts.append(f"ss={ss_p:.2f}")
    if wd_p > 0:
        epoch_parts.append(f"wd={wd_p:.2f}")
    if do_kl and val.all_kl_per_dim:
        kl_cat = torch.cat(val.all_kl_per_dim, dim=0)
        mean_kl_per_dim = kl_cat.mean(dim=0)
        active_dims = int((mean_kl_per_dim > 0.1).sum().item())
        epoch_parts.append(f"| active={active_dims}/{cfg.latent_dim}")

    logger.info("  ".join(epoch_parts))

    # Epoch-level bucketed CE by sequence length
    _epoch_bkt_parts = []
    for _bi in range(3):
        if bucket_ce_count[_bi] > 0:
            _bkt_ce = bucket_ce_sum[_bi] / bucket_ce_count[_bi]
            _bkt_pct = bucket_ce_count[_bi] / max(train_count, 1) * 100
            _epoch_bkt_parts.append(
                f"{_BUCKET_NAMES[_bi]}={_bkt_ce:.3f} (n={bucket_ce_count[_bi]}, {_bkt_pct:.0f}%)"
            )
    if _epoch_bkt_parts:
        logger.info("  train CE by len: %s", " | ".join(_epoch_bkt_parts))
    _val_bkt_parts = []
    for _bi in range(3):
        if val.val_bucket_ce_count[_bi] > 0:
            _vbkt_ce = val.val_bucket_ce_sum[_bi] / val.val_bucket_ce_count[_bi]
            _vbkt_pct = val.val_bucket_ce_count[_bi] / max(val.val_count, 1) * 100
            _val_bkt_parts.append(
                f"{_BUCKET_NAMES[_bi]}={_vbkt_ce:.3f} (n={val.val_bucket_ce_count[_bi]}, {_vbkt_pct:.0f}%)"
            )
    if _val_bkt_parts:
        logger.info("  val CE by len:   %s", " | ".join(_val_bkt_parts))


def run_contrastive_alignment_diagnostic(
    *,
    epoch: int,
    cfg: VAEPretrainConfig,
    modules: dict[str, nn.Module],
    train_loader: object,
    device: torch.device,
    full_vocab: int,
    bos_id: int,
    corpus_embeddings: Tensor,
    use_contrastive: bool,
) -> None:
    """Measure Spearman correlation between z-cosine and embedding-cosine."""
    with torch.no_grad():
        # Collect z vectors from a subset of training data
        _diag_z: list[Tensor] = []
        _diag_idx: list[int] = []
        _diag_count = 0
        for _db in train_loader:
            if use_contrastive:
                _dt, _dl, _di = _db
            else:
                _dt, _dl = _db
                _di = None
            _dt = _dt.to(device)
            _dl = torch.as_tensor(_dl, device=device)
            with torch.amp.autocast(
                device_type=device.type, enabled=cfg.use_amp,
            ):
                _, _, _, _zb, _, _, _, _, _ = _vae_forward(
                    _dt, _dl, bos_id=bos_id, full_vocab=full_vocab,
                    kl_free_bits=0.0, compute_kl=False, **modules,
                )
            _diag_z.append(_zb.cpu())
            if _di is not None:
                _diag_idx.extend(_di.tolist())
            _diag_count += _zb.size(0)
            if _diag_count >= 2000:
                break

        _dz = torch.cat(_diag_z, dim=0)[:2000]
        _di_list = _diag_idx[:2000]
        _de = corpus_embeddings[_di_list]

        _zn = F.normalize(_dz.float(), dim=-1)
        _en = F.normalize(_de.float(), dim=-1)

        # Sample 2000 random pairs
        _rng = torch.Generator().manual_seed(epoch)
        _ia = torch.randint(0, len(_dz), (2000,), generator=_rng)
        _ib = torch.randint(0, len(_dz), (2000,), generator=_rng)
        _z_sims = (_zn[_ia] * _zn[_ib]).sum(dim=-1)
        _e_sims = (_en[_ia] * _en[_ib]).sum(dim=-1)

        # Spearman correlation
        from scipy.stats import spearmanr as _spearmanr

        _corr, _ = _spearmanr(_z_sims.numpy(), _e_sims.numpy())
        logger.info(
            "  z-embed alignment: r=%.3f (v1 baseline=0.245)",
            _corr,
        )

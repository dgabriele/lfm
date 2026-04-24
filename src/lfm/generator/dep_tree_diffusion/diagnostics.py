"""Checkpoint diagnostic digest for DepTree Diffusion VAE.

Provides a comprehensive multi-perspective analysis at each checkpoint:
reconstruction quality, latent space health, topology, information theory,
linguistic coherence, and prior generation quality.

Designed to be called from the trainer or any external consumer with
a loaded model, sentencepiece processor, and config.
"""

from __future__ import annotations

import logging
from collections import Counter
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import DataLoader

from lfm.generator.dep_tree_diffusion.config import DepTreeDiffusionConfig
from lfm.generator.dep_tree_diffusion.model import DepTreeDiffusionVAE
from lfm.generator.dep_tree_vae.config import DEP_RELATIONS

logger = logging.getLogger(__name__)

_diag_loader: DataLoader | None = None

# Fixed t for single-step evaluation — the model's output_head is
# well-calibrated here (99%+ accuracy). Lower values produce garbage.
_EVAL_T = 0.5


def _get_loader(cfg: DepTreeDiffusionConfig, batch_size: int) -> DataLoader:
    global _diag_loader
    if _diag_loader is None or _diag_loader.batch_size != batch_size:
        from lfm.generator.dep_tree_diffusion.data import (
            DiffusionDepTreeDataset,
            collate_diffusion,
        )

        cache_dir = Path(cfg.dataset_path) / "diffusion_cache"
        ds = DiffusionDepTreeDataset(cache_dir)
        _diag_loader = DataLoader(
            ds, batch_size=batch_size, shuffle=True,
            collate_fn=collate_diffusion, num_workers=0,
        )
    return _diag_loader


def _pearson(a: Tensor, b: Tensor) -> Tensor:
    ac = a - a.mean()
    bc = b - b.mean()
    return (ac * bc).sum() / (ac.norm() * bc.norm()).clamp(min=1e-8)


def _decode_batch(
    model: DepTreeDiffusionVAE,
    z: Tensor,
    tokens: Tensor,
    lengths: Tensor,
    per_token_roles: Tensor,
    depths: Tensor,
    cfg: DepTreeDiffusionConfig,
    sp,
) -> tuple[list[str], list[list[int]]]:
    """Full diffusion decode for a batch of z vectors. Returns texts and token id lists."""
    s = tokens.size(1)
    n = z.size(0)
    spm_size = sp.get_piece_size()

    tok_ids = model.diffusion_decoder.sample(
        s, per_token_roles[:n], depths[:n],
        model._z_to_memory(z),
        num_steps=cfg.diffusion.num_diffusion_steps,
        depth_scale=cfg.diffusion.depth_scale,
        min_noise=cfg.diffusion.min_noise,
        role_offset=model._role_offset,
        ref_tokens=tokens[:n],
        word_pos_noise_scale=cfg.diffusion.word_pos_noise_scale,
    )

    texts = []
    token_lists = []
    for i in range(n):
        n_tok = lengths[i].item()
        ids = tok_ids[i, :n_tok].tolist()
        content_ids = [t for t in ids if 0 < t < spm_size]
        token_lists.append(content_ids)
        texts.append(sp.DecodeIds(content_ids))
    return texts, token_lists


def _repetition_stats(token_lists: list[list[int]]) -> tuple[float, int]:
    """Consecutive-repetition rate and total consecutive pairs."""
    reps = 0
    total = 0
    for ids in token_lists:
        for j in range(1, len(ids)):
            total += 1
            if ids[j] == ids[j - 1]:
                reps += 1
    return reps / max(total, 1), total


@torch.no_grad()
def checkpoint_digest(
    model: DepTreeDiffusionVAE,
    sp,
    device: torch.device,
    cfg: DepTreeDiffusionConfig,
    n_samples: int = 256,
    n_decode: int = 32,
    n_topo: int = 64,
) -> dict:
    """Run comprehensive diagnostic digest and log results.

    Single-step metrics (reconstruction, topology, entropy) use a fixed
    t_global=0.5 where the model's output_head is calibrated. Full-decode
    metrics use the complete iterative sampling pipeline and reflect
    actual downstream generation quality.

    Returns:
        Dict of all measured scalars (for programmatic consumption).
    """
    from lfm.translator.romanize import respell

    model.eval()
    loader = _get_loader(cfg, n_samples)
    batch = next(iter(loader))
    tokens = batch["tokens"].to(device)
    depths = batch["depths"].to(device)
    lengths = batch["lengths"].to(device)
    b, s = tokens.shape
    spm_size = sp.get_piece_size()

    metrics: dict[str, float] = {}

    logger.info("=" * 70)
    logger.info("CHECKPOINT DIGEST (n=%d)", b)
    logger.info("=" * 70)

    # ── Encode ───────────────────────────────────────────────────
    mu, logvar = model.encoder(tokens, lengths)
    z_struct, z_content, z = model.latent(mu, logvar)
    per_token_roles = model._extract_per_token_roles(tokens, lengths)
    z_memory = model._z_to_memory(z)

    is_role = tokens >= model._role_offset

    # Single-step forward at fixed t=0.5 for calibrated metrics
    x0_pred, logits, padding = model.low_noise_forward(
        tokens, lengths, depths, per_token_roles, z_memory,
        t_global=_EVAL_T,
    )
    is_content = ~padding & ~is_role
    preds = logits.argmax(dim=-1)

    # ── RECONSTRUCTION (single-step at t=0.5) ───────────────────
    content_correct = (preds == tokens) & is_content
    depth_bins = [0, 1, 2, 3, 4]

    depth_accs = []
    for d in depth_bins:
        mask = is_content & ((depths == d) if d < 4 else (depths >= d))
        if mask.any():
            acc = content_correct[mask].float().mean().item()
            label = str(d) if d < 4 else "4+"
            depth_accs.append((label, acc, mask.sum().item()))
            metrics[f"acc_d{label}"] = acc

    total_acc = content_correct[is_content].float().mean().item() if is_content.any() else 0.0
    ce = F.cross_entropy(
        logits[is_content], tokens[is_content].clamp(max=logits.size(-1) - 1),
    ).item() if is_content.any() else 0.0
    metrics["acc_total"] = total_acc
    metrics["ce_total"] = ce

    logger.info("── Reconstruction (t=%.1f) ──", _EVAL_T)
    logger.info("  overall: acc=%.1f%% CE=%.4f", total_acc * 100, ce)
    logger.info("  by depth: %s", "  ".join(f"d{d}={a:.0%}({n})" for d, a, n in depth_accs))

    # Per-role-type accuracy
    role_acc_map: dict[int, tuple[float, int]] = {}
    for ri in range(len(DEP_RELATIONS)):
        mask = is_content & (per_token_roles == ri)
        if mask.any():
            role_acc_map[ri] = (content_correct[mask].float().mean().item(), mask.sum().item())

    sorted_roles = sorted(role_acc_map.items(), key=lambda x: -x[1][1])[:10]
    logger.info("  by role (top-10): %s",
                "  ".join(f"{DEP_RELATIONS[ri]}={a:.0%}({n})" for ri, (a, n) in sorted_roles))

    worst = sorted(
        ((ri, a, n) for ri, (a, n) in role_acc_map.items() if n > 20),
        key=lambda x: x[1],
    )[:3]
    if worst:
        logger.info("  worst-3 roles: %s",
                    "  ".join(f"{DEP_RELATIONS[ri]}={a:.0%}({n})" for ri, a, n in worst))

    # ── LATENT SPACE HEALTH ──────────────────────────────────────
    z_std = (0.5 * logvar).exp()
    metrics["z_std_mean"] = z_std.mean().item()
    metrics["z_std_min"] = z_std.min().item()
    metrics["z_std_max"] = z_std.max().item()

    kl_per_dim = 0.5 * (mu.pow(2) + logvar.exp() - 1 - logvar)
    active_units = (kl_per_dim.mean(dim=0) > 0.01).sum().item()
    total_dims = kl_per_dim.size(1)
    kl_total = kl_per_dim.sum(dim=-1).mean().item()
    kl_struct = kl_per_dim[:, :cfg.latent.struct_dim].sum(dim=-1).mean().item()
    kl_content = kl_per_dim[:, cfg.latent.struct_dim:].sum(dim=-1).mean().item()
    metrics["kl_total"] = kl_total
    metrics["kl_struct"] = kl_struct
    metrics["kl_content"] = kl_content
    metrics["active_units"] = active_units
    metrics["active_units_pct"] = active_units / total_dims * 100

    z_var_per_dim = z.var(dim=0)
    undercovered = (z_var_per_dim < 0.01).sum().item()
    overcovered = (z_var_per_dim > 2.0).sum().item()
    metrics["undercovered_dims"] = undercovered
    metrics["overcovered_dims"] = overcovered

    logger.info("── Latent Space ──")
    logger.info("  z_std: mean=%.4f ±%.4f [%.4f, %.4f]",
                metrics["z_std_mean"], z_std.std().item(), metrics["z_std_min"], metrics["z_std_max"])
    logger.info("  KL: total=%.2f (struct=%.2f content=%.2f)", kl_total, kl_struct, kl_content)
    logger.info("  active units: %d/%d (%.0f%%)  undercovered=%d overcovered=%d",
                active_units, total_dims, metrics["active_units_pct"], undercovered, overcovered)

    # ── TOPOLOGY (t=0.5) ────────────────────────────────────────
    valid_f = (~padding).unsqueeze(-1).float()
    decoded_repr = (x0_pred * valid_f).sum(dim=1) / valid_f.sum(dim=1).clamp(min=1)
    nt = min(b, n_topo)
    idx_t = torch.triu_indices(nt, nt, offset=1, device=device)
    z_d = torch.cdist(z[:nt], z[:nt])[idx_t[0], idx_t[1]]
    o_d = torch.cdist(decoded_repr[:nt], decoded_repr[:nt])[idx_t[0], idx_t[1]]

    topo_rho = _pearson(z_d, o_d).item() if z_d.numel() > 2 else 0.0
    metrics["topo_rho"] = topo_rho

    zs_d = torch.cdist(z[:nt, :cfg.latent.struct_dim], z[:nt, :cfg.latent.struct_dim])[idx_t[0], idx_t[1]]
    zc_d = torch.cdist(z[:nt, cfg.latent.struct_dim:], z[:nt, cfg.latent.struct_dim:])[idx_t[0], idx_t[1]]
    rho_struct = _pearson(zs_d, o_d).item() if z_d.numel() > 2 else 0.0
    rho_content = _pearson(zc_d, o_d).item() if z_d.numel() > 2 else 0.0
    metrics["topo_rho_struct"] = rho_struct
    metrics["topo_rho_content"] = rho_content

    logger.info("── Topology (t=%.1f, n=%d) ──", _EVAL_T, nt)
    logger.info("  ρ full=%.4f  struct=%.4f  content=%.4f", topo_rho, rho_struct, rho_content)

    # ── ENTROPY (t=0.5) ─────────────────────────────────────────
    log_probs = torch.log_softmax(logits, dim=-1)
    probs = log_probs.exp()
    per_pos_entropy = -(probs * log_probs).sum(dim=-1)
    content_entropy = per_pos_entropy[is_content]

    metrics["entropy_mean"] = content_entropy.mean().item()
    metrics["entropy_std"] = content_entropy.std().item()
    metrics["entropy_min"] = content_entropy.min().item()
    metrics["entropy_p10"] = content_entropy.quantile(0.1).item()
    metrics["entropy_p50"] = content_entropy.quantile(0.5).item()

    below_floor = (content_entropy < cfg.entropy_floor).float().mean().item()
    metrics["below_entropy_floor_pct"] = below_floor * 100

    eff_vocab = (probs[is_content] > 0.01).float().sum(dim=-1).mean().item()
    top1_prob = probs[is_content].max(dim=-1).values.mean().item()
    metrics["eff_vocab"] = eff_vocab
    metrics["top1_confidence"] = top1_prob * 100

    logger.info("── Entropy (t=%.1f) ──", _EVAL_T)
    logger.info("  entropy: mean=%.2f ±%.2f  min=%.2f p10=%.2f p50=%.2f",
                metrics["entropy_mean"], metrics["entropy_std"],
                metrics["entropy_min"], metrics["entropy_p10"], metrics["entropy_p50"])
    logger.info("  below floor (%.1f): %.1f%%  eff_vocab=%.1f  top1_conf=%.1f%%",
                cfg.entropy_floor, metrics["below_entropy_floor_pct"], eff_vocab, top1_prob * 100)

    depth_ents = []
    for d in depth_bins:
        mask = is_content & ((depths == d) if d < 4 else (depths >= d))
        if mask.any():
            label = str(d) if d < 4 else "4+"
            ent = per_pos_entropy[mask].mean().item()
            depth_ents.append((label, ent))
            metrics[f"entropy_d{label}"] = ent
    logger.info("  entropy by depth: %s", "  ".join(f"d{d}={e:.2f}" for d, e in depth_ents))

    # ── FULL DECODE: POSTERIOR ───────────────────────────────────
    nd = min(b, n_decode)
    decoded_texts, decoded_ids = _decode_batch(
        model, z[:nd], tokens, lengths, per_token_roles, depths, cfg, sp,
    )

    unique_texts = len(set(decoded_texts))
    rep_rate, _ = _repetition_stats(decoded_ids)
    metrics["posterior_diversity_pct"] = unique_texts / nd * 100
    metrics["posterior_rep_rate_pct"] = rep_rate * 100

    all_ids = [t for ids in decoded_ids for t in ids]
    freq = Counter(all_ids)
    unique_tokens = len(freq)
    total_gen = len(all_ids)
    top10_mass = sum(c for _, c in freq.most_common(10)) / max(total_gen, 1)
    metrics["posterior_unique_tokens"] = unique_tokens
    metrics["posterior_top10_mass_pct"] = top10_mass * 100

    logger.info("── Full Decode: Posterior (n=%d) ──", nd)
    logger.info("  diversity: %d/%d unique (%.0f%%)  rep_rate=%.1f%%  vocab=%d  top-10 mass=%.0f%%",
                unique_texts, nd, metrics["posterior_diversity_pct"],
                rep_rate * 100, unique_tokens, top10_mass * 100)

    logger.info("  ── GT → Generated ──")
    for i in range(min(4, nd)):
        n_tok = lengths[i].item()
        gt_ids = [t for t in tokens[i, :n_tok].tolist() if 0 < t < spm_size]
        logger.info("  [%d] GT:  %s", i, respell(sp.DecodeIds(gt_ids)))
        logger.info("  [%d] Gen: %s", i, respell(decoded_texts[i]))

    # ── FULL DECODE: PRIOR ──────────────────────────────────────
    z_prior = torch.randn(nd, cfg.latent.total_dim, device=device)
    prior_texts, prior_ids = _decode_batch(
        model, z_prior, tokens, lengths, per_token_roles, depths, cfg, sp,
    )

    prior_unique = len(set(prior_texts))
    prior_rep, _ = _repetition_stats(prior_ids)
    prior_vocab = len({t for ids in prior_ids for t in ids})
    metrics["prior_diversity_pct"] = prior_unique / nd * 100
    metrics["prior_rep_rate_pct"] = prior_rep * 100
    metrics["prior_unique_tokens"] = prior_vocab

    logger.info("── Full Decode: Prior (n=%d) ──", nd)
    logger.info("  diversity: %d/%d unique (%.0f%%)  rep_rate=%.1f%%  vocab=%d",
                prior_unique, nd, metrics["prior_diversity_pct"], prior_rep * 100, prior_vocab)
    for i in range(min(4, nd)):
        logger.info("  prior[%d]: %s", i, respell(prior_texts[i]))

    logger.info("=" * 70)
    model.train()
    return metrics

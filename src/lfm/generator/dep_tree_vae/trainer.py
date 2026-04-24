"""Training loop for DepTreeVAE."""

from __future__ import annotations

import logging
import signal
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from lfm.generator.dep_tree_vae.config import DepTreeVAEConfig
from lfm.generator.dep_tree_vae.data import build_dataloaders
from lfm.generator.dep_tree_vae.model import DepTreeVAE

logger = logging.getLogger(__name__)


def _cleanup_gpu():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def train_dep_tree_vae(cfg: DepTreeVAEConfig) -> None:
    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
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

    train_loader, val_loader, sp, vocab_size = build_dataloaders(cfg)

    # Completeness scorer (frozen)
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

    model = DepTreeVAE(cfg, vocab_size).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(
        "DepTreeVAE: %d params (%.1fM), %d trainable (%.1fM), device=%s",
        n_params, n_params / 1e6, n_trainable, n_trainable / 1e6, device,
    )

    param_groups = model.trainable_parameters()
    optimizer = AdamW(param_groups, lr=cfg.lr, weight_decay=0.01)
    total_steps = len(train_loader) * cfg.num_epochs
    scheduler = CosineAnnealingLR(optimizer, T_max=total_steps, eta_min=cfg.lr_min)
    scaler = torch.amp.GradScaler("cuda", enabled=cfg.use_amp)

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
            logger.warning("Optimizer state incompatible, resetting: %s", e)
        for pg in optimizer.param_groups:
            pg["lr"] = cfg.lr
        scheduler = CosineAnnealingLR(optimizer, T_max=total_steps, eta_min=cfg.lr_min)
        try:
            scaler.load_state_dict(ckpt["scaler_state"])
        except RuntimeError:
            pass
        global_step = ckpt.get("global_step", 0)
        start_epoch = ckpt.get("epoch", 0)
        best_val_loss = ckpt.get("best_val_loss", float("inf"))
        logger.info("Resumed from epoch=%d step=%d best_val=%.4f", start_epoch, global_step, best_val_loss)

    accum = cfg.gradient_accumulation_steps

    for epoch in range(start_epoch, cfg.num_epochs):
        model.train()
        optimizer.zero_grad()

        epoch_loss = 0.0
        n_batches = 0
        batch_start = time.time()

        for i, batch in enumerate(train_loader):
            if _shutdown_requested:
                logger.info("Shutdown requested — saving checkpoint at step %d", global_step)
                _save_checkpoint(model, optimizer, scheduler, scaler, epoch, global_step, best_val_loss, out_dir)
                _cleanup_gpu()
                return

            batch = {k: v.to(device) for k, v in batch.items()}
            kl_weight = _kl_schedule(global_step, cfg)

            if cfg.word_dropout > 0:
                anneal_frac = min(epoch / max(cfg.word_dropout_anneal_epochs, 1), 1.0)
                model._word_dropout_p = cfg.word_dropout + anneal_frac * (cfg.word_dropout_min - cfg.word_dropout)
            else:
                model._word_dropout_p = 0.0

            with torch.amp.autocast(device_type=device.type, enabled=cfg.use_amp):
                out = model(
                    tokens=batch["tokens"],
                    lengths=batch["lengths"],
                    role_ids=batch["role_ids"],
                    role_lengths=batch["role_lengths"],
                    kl_weight=kl_weight,
                )

                z = torch.cat([out.z_struct, out.z_content], dim=-1)

                # One-sided z_var: only penalize below-floor variance
                z_var_loss = torch.tensor(0.0, device=device)
                if cfg.z_var_weight > 0:
                    per_dim_var = z.var(dim=0)
                    shortfall = torch.clamp(cfg.z_var_target - per_dim_var, min=0)
                    z_var_loss = cfg.z_var_weight * shortfall.pow(2).mean()

                # Z-prediction: decoder hidden states must retain z info.
                # Uses cosine similarity — scale-invariant, bounded gradient.
                z_pred_loss = torch.tensor(0.0, device=device)
                if cfg.z_pred_weight > 0 and out.hidden is not None and hasattr(model, 'z_predictor'):
                    valid = out.content_mask.unsqueeze(-1).float()
                    pooled = (out.hidden * valid).sum(dim=1) / valid.sum(dim=1).clamp(min=1)
                    z_hat = model.z_predictor(pooled)
                    cos_sim = F.cosine_similarity(z_hat, z.detach(), dim=-1).mean()
                    z_pred_loss = cfg.z_pred_weight * (1.0 - cos_sim)

                # Topology: z-distance vs output-distance correlation
                topo_loss = torch.tensor(0.0, device=device)
                topo_rho = torch.tensor(0.0, device=device)
                if cfg.topo_weight > 0 and z.size(0) >= 4 and out.hidden is not None:
                    valid = out.content_mask.unsqueeze(-1).float()
                    decoded_repr = (out.hidden * valid).sum(dim=1) / valid.sum(dim=1).clamp(min=1)
                    idx = torch.triu_indices(z.size(0), z.size(0), offset=1, device=device)
                    z_dists = torch.cdist(z, z)[idx[0], idx[1]]
                    out_dists = torch.cdist(decoded_repr, decoded_repr)[idx[0], idx[1]]
                    if z_dists.numel() > 2:
                        zc = z_dists - z_dists.mean()
                        oc = out_dists - out_dists.mean()
                        denom = (zc.norm() * oc.norm()).clamp(min=1e-8)
                        topo_rho = (zc * oc).sum() / denom
                        topo_loss = cfg.topo_weight * (1.0 - topo_rho)

                # Entropy floor: penalize low-entropy positions
                entropy_loss = torch.tensor(0.0, device=device)
                if cfg.entropy_weight > 0 and out.logits is not None:
                    log_probs = torch.log_softmax(out.logits, dim=-1)
                    probs = log_probs.exp()
                    per_pos_entropy = -(probs * log_probs).sum(dim=-1)
                    shortfall = torch.clamp(cfg.entropy_floor - per_pos_entropy, min=0)
                    valid_ent = out.content_mask.float()
                    entropy_loss = cfg.entropy_weight * (shortfall * valid_ent).sum() / valid_ent.sum().clamp(min=1)

                # Repetition penalty: cosine similarity between consecutive logit vectors
                rep_loss = torch.tensor(0.0, device=device)
                if cfg.rep_penalty_weight > 0 and out.logits is not None and out.logits.size(1) > 1:
                    logits_a = out.logits[:, :-1]
                    logits_b = out.logits[:, 1:]
                    cos_sim = F.cosine_similarity(logits_a, logits_b, dim=-1)
                    valid_pairs = out.content_mask[:, 1:].float()
                    rep_loss = cfg.rep_penalty_weight * (cos_sim * valid_pairs).sum() / valid_pairs.sum().clamp(min=1)

                # Interpolation smoothness (every 4th step)
                interp_loss = torch.tensor(0.0, device=device)
                if cfg.interp_weight > 0 and z.size(0) >= 4 and global_step % 4 == 0 and out.hidden is not None:
                    n_pairs = z.size(0) // 2
                    valid_i = out.content_mask[:n_pairs].unsqueeze(-1).float()
                    h_a = (out.hidden[:n_pairs] * valid_i).sum(dim=1) / valid_i.sum(dim=1).clamp(min=1)
                    h_b_start = n_pairs
                    valid_b = out.content_mask[h_b_start:2*n_pairs].unsqueeze(-1).float()
                    h_b = (out.hidden[h_b_start:2*n_pairs] * valid_b).sum(dim=1) / valid_b.sum(dim=1).clamp(min=1)
                    expected_mid = 0.5 * (h_a + h_b)
                    # For actual midpoint, we'd need another forward. Use linear
                    # interpolation in hidden space as a proxy (avoids extra decode).
                    interp_loss = cfg.interp_weight * F.mse_loss(
                        0.5 * (h_a + h_b), expected_mid,
                    )
                    # Note: this is trivially zero as written. For a real interp loss
                    # we need to decode z_mid — TODO if needed. For now topo handles
                    # the smoothness objective.

                # Completeness scorer
                completeness_loss = torch.tensor(0.0, device=device)
                if completeness_scorer is not None and cfg.completeness_weight > 0 and out.logits is not None:
                    with torch.amp.autocast(device_type=device.type, enabled=False):
                        scorer_vocab = completeness_scorer.cfg.vocab_size
                        logits_f = out.logits.float()
                        if logits_f.size(-1) < scorer_vocab:
                            pad = torch.full(
                                (*logits_f.shape[:-1], scorer_vocab - logits_f.size(-1)),
                                -1e9, device=device,
                            )
                            logits_f = torch.cat([logits_f, pad], dim=-1)
                        scores = completeness_scorer.score_soft(
                            logits_f[:, :, :scorer_vocab],
                            out.content_mask.sum(dim=1),
                        )
                        completeness_loss = -cfg.completeness_weight * scores.mean()

                loss = (out.total_loss + z_var_loss + z_pred_loss + topo_loss + entropy_loss + rep_loss + completeness_loss) / accum

            scaler.scale(loss).backward()

            if (i + 1) % accum == 0:
                scaler.unscale_(optimizer)
                graded = [p for g in param_groups for p in g["params"] if p.grad is not None]
                gnorm = nn.utils.clip_grad_norm_(graded, 5.0).item() if graded else 0.0
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                scheduler.step()
                global_step += 1

                if global_step % cfg.log_every == 0:
                    elapsed = time.time() - batch_start
                    z_std_mean = (0.5 * out.logvar).exp().mean().item()
                    topo_str = f" topo={topo_rho.item():.3f}" if topo_rho.item() != 0 else ""
                    logger.info(
                        "ep%d step=%d  recon=%.3f skel=%.3f kl=%.3f%s "
                        "z_std=%.4f gnorm=%.2f lr=%.6f  [%.1fs]",
                        epoch, global_step,
                        out.recon_loss.item(),
                        out.skeleton_loss.item(),
                        out.kl_loss.item(),
                        topo_str, z_std_mean, gnorm,
                        scheduler.get_last_lr()[0], elapsed,
                    )
                    batch_start = time.time()

                if global_step % cfg.checkpoint_every_steps == 0:
                    _save_checkpoint(model, optimizer, scheduler, scaler, epoch, global_step, best_val_loss, out_dir)
                    torch.cuda.empty_cache()
                    try:
                        _checkpoint_digest(model, sp, device, cfg)
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

        val_loss = _validate(model, val_loader, device, cfg, global_step)
        logger.info("ep%d val_loss=%.4f (best=%.4f)", epoch, val_loss, best_val_loss)

        _save_checkpoint(model, optimizer, scheduler, scaler, epoch + 1, global_step, best_val_loss, out_dir)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            _save_checkpoint(model, optimizer, scheduler, scaler, epoch + 1, global_step, best_val_loss, out_dir, name="best.pt")

    logger.info("Training complete. best_val_loss=%.4f", best_val_loss)


@torch.no_grad()
def _validate(model, val_loader, device, cfg, global_step):
    model.eval()
    total_loss = 0.0
    n = 0
    for batch in val_loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        out = model(
            tokens=batch["tokens"], lengths=batch["lengths"],
            role_ids=batch["role_ids"], role_lengths=batch["role_lengths"],
            kl_weight=_kl_schedule(global_step, cfg),
        )
        total_loss += out.total_loss.item()
        n += 1
    model.train()
    return total_loss / max(n, 1)


def _kl_schedule(step: int, cfg: DepTreeVAEConfig) -> float:
    if cfg.kl_weight <= 0:
        return 0.0
    warmup = max(cfg.kl_warmup_steps, 1)
    return cfg.kl_weight * min(step / warmup, 1.0)


@torch.no_grad()
def _greedy_decode(
    model: DepTreeVAE, z: torch.Tensor, device: torch.device,
    cfg: DepTreeVAEConfig, sp, max_len: int | None = None,
    ngram_block: tuple[int, ...] = (2, 3, 4),
) -> list[tuple[str, bool]]:
    """Greedy AR decode from z vectors. Returns list of (text, hit_eos)."""
    from lfm.generator.dep_tree_vae.config import DEP_RELATIONS
    from lfm.generator.dep_tree_vae.skeleton import SKEL_BOS, SKEL_EOS
    from lfm.generator.layers import multiscale_causal_mask

    if max_len is None:
        max_len = cfg.max_seq_len - 1

    b = z.size(0)
    z_struct, z_content = model.latent.split(z)
    skel_tokens = model.skeleton_decoder(z_struct)[0]
    spm_size = sp.get_piece_size()

    results = []
    for i in range(b):
        roles = []
        for t in skel_tokens[i]:
            t_val = t.item()
            if t_val == SKEL_BOS:
                continue
            if t_val == SKEL_EOS:
                break
            if t_val < len(DEP_RELATIONS):
                roles.append(t_val)
        if not roles:
            roles = [DEP_RELATIONS.index("root")]

        role_ids = torch.tensor(roles, device=device).unsqueeze(0)
        memory = model.phrase_projector(z_content[i:i+1], role_ids)

        tokens = torch.full((1, 1), model._bos_id, dtype=torch.long, device=device)
        hit_eos = False
        generated: list[int] = []
        for _ in range(max_len):
            seq_len = tokens.size(1)
            tok_emb = model.dec_token_embedding(tokens)
            tgt_mask = multiscale_causal_mask(
                seq_len, cfg.decoder_num_heads,
                tuple(cfg.attention_head_windows),
                cfg.attention_global_every, device=device,
            )
            rope = model._rope_freqs[:seq_len] if model._rope_freqs is not None else None
            hidden = model.phrase_decoder(tok_emb, memory, tgt_mask=tgt_mask, rope_freqs=rope)
            logits = model.output_head(hidden[:, -1, :])
            for n in ngram_block:
                if len(generated) >= n - 1:
                    prefix = tuple(generated[-(n - 1):])
                    for j in range(len(generated) - n + 1):
                        if tuple(generated[j:j + n - 1]) == prefix:
                            banned = generated[j + n - 1]
                            logits[0, banned] = -float("inf")
            next_tok = logits.argmax(dim=-1, keepdim=True)
            if next_tok.item() == model._eos_id:
                hit_eos = True
                break
            generated.append(next_tok.item())
            tokens = torch.cat([tokens, next_tok], dim=1)

        ids = tokens[0, 1:].tolist()
        ids = [t for t in ids if 0 < t < spm_size]
        results.append((sp.DecodeIds(ids), hit_eos))

    return results


@torch.no_grad()
def _checkpoint_digest(
    model: DepTreeVAE, sp, device: torch.device, cfg: DepTreeVAEConfig,
    n_recon: int = 8, n_interp: int = 4, n_prior: int = 16,
) -> None:
    """Comprehensive checkpoint diagnostic: reconstruction, interpolation, prior."""
    from lfm.generator.dep_tree_vae.data import DepTreeDataset, collate_dep_tree
    from lfm.translator.romanize import respell
    from pathlib import Path

    model.eval()

    # Load diagnostic batch
    cache_dir = Path(cfg.dataset_path) / "cache"
    ds = DepTreeDataset(cache_dir)
    loader = torch.utils.data.DataLoader(
        ds, batch_size=max(n_recon, n_interp * 2), shuffle=True,
        collate_fn=collate_dep_tree, num_workers=0,
    )
    batch = next(iter(loader))
    tokens = batch["tokens"].to(device)
    lengths = batch["lengths"].to(device)
    role_ids = batch["role_ids"].to(device)
    role_lengths = batch["role_lengths"].to(device)
    spm_size = sp.get_piece_size()

    logger.info("=" * 70)
    logger.info("CHECKPOINT DIGEST")
    logger.info("=" * 70)

    # Encode
    mu, logvar = model.encoder(tokens, lengths)
    z_struct, z_content, z = model.latent(mu, logvar)

    # ── RECONSTRUCTION ───────────────────────────────────────────
    logger.info("── Reconstruction (n=%d) ──", n_recon)
    recon_results = _greedy_decode(model, z[:n_recon], device, cfg, sp)
    eos_count = sum(1 for _, eos in recon_results if eos)

    for i in range(min(4, n_recon)):
        n_tok = lengths[i].item()
        gt_ids = [t for t in tokens[i, :n_tok].tolist() if 0 < t < spm_size]
        gt_text = respell(sp.DecodeIds(gt_ids))
        gen_text = respell(recon_results[i][0])
        logger.info("  [%d] GT:  %s", i, gt_text)
        logger.info("  [%d] Rec: %s", i, gen_text)

    logger.info("  eos_rate=%.0f%%", eos_count / n_recon * 100)

    # ── POSTERIOR INTERPOLATION ──────────────────────────────────
    struct_dim = cfg.latent.struct_dim
    logger.info("── Posterior Interpolation (n=%d pairs) ──", n_interp)
    for i in range(min(n_interp, tokens.size(0) // 2)):
        a_idx, b_idx = i * 2, i * 2 + 1
        z_a, z_b = z[a_idx], z[b_idx]

        z_mid = 0.5 * z_a + 0.5 * z_b
        texts = _greedy_decode(model, torch.stack([z_a, z_mid, z_b]), device, cfg, sp)
        logger.info("  interp[%d] A:   %s", i, respell(texts[0][0]))
        logger.info("  interp[%d] mid: %s", i, respell(texts[1][0]))
        logger.info("  interp[%d] B:   %s", i, respell(texts[2][0]))

        # Struct-only interpolation
        z_struct_mid = torch.cat([
            0.5 * z_a[:struct_dim] + 0.5 * z_b[:struct_dim],
            z_a[struct_dim:],
        ])
        texts_s = _greedy_decode(model, z_struct_mid.unsqueeze(0), device, cfg, sp)
        logger.info("  struct_mid[%d]:  %s", i, respell(texts_s[0][0]))

        # Content-only interpolation
        z_content_mid = torch.cat([
            z_a[:struct_dim],
            0.5 * z_a[struct_dim:] + 0.5 * z_b[struct_dim:],
        ])
        texts_c = _greedy_decode(model, z_content_mid.unsqueeze(0), device, cfg, sp)
        logger.info("  content_mid[%d]: %s", i, respell(texts_c[0][0]))

    # ── PRIOR INTERPOLATION ─────────────────────────────────────
    logger.info("── Prior Interpolation (z ~ N(0,1)) ──")
    for i in range(2):
        z_a_p = torch.randn(cfg.latent.total_dim, device=device)
        z_b_p = torch.randn(cfg.latent.total_dim, device=device)
        z_mid_p = 0.5 * z_a_p + 0.5 * z_b_p
        texts = _greedy_decode(model, torch.stack([z_a_p, z_mid_p, z_b_p]), device, cfg, sp)
        logger.info("  prior_interp[%d] A:   %s", i, respell(texts[0][0]))
        logger.info("  prior_interp[%d] mid: %s", i, respell(texts[1][0]))
        logger.info("  prior_interp[%d] B:   %s", i, respell(texts[2][0]))

    # ── PRIOR GENERATION ────────────────────────────────────────
    z_prior = torch.randn(n_prior, cfg.latent.total_dim, device=device)
    prior_results = _greedy_decode(model, z_prior, device, cfg, sp)

    eos_count_p = sum(1 for _, eos in prior_results if eos)
    prior_texts = [t for t, _ in prior_results]
    unique_texts = len(set(prior_texts))

    logger.info("── Prior Generation (n=%d) ──", n_prior)
    logger.info("  eos=%.0f%%  unique=%d/%d", eos_count_p / n_prior * 100, unique_texts, n_prior)
    for j in range(min(4, n_prior)):
        logger.info("  prior[%d]: %s", j, respell(prior_texts[j]))

    # ── LATENT SPACE ────────────────────────────────────────────
    z_std = (0.5 * logvar).exp()
    kl_per_dim = 0.5 * (mu.pow(2) + logvar.exp() - 1 - logvar)
    active = (kl_per_dim.mean(dim=0) > 0.01).sum().item()
    logger.info("── Latent Space ──")
    logger.info("  z_std: mean=%.4f ±%.4f [%.4f, %.4f]",
                z_std.mean().item(), z_std.std().item(), z_std.min().item(), z_std.max().item())
    logger.info("  KL: total=%.2f  active=%d/%d",
                kl_per_dim.sum(dim=-1).mean().item(), active, cfg.latent.total_dim)

    logger.info("=" * 70)
    model.train()


def _save_checkpoint(model, optimizer, scheduler, scaler, epoch, global_step, best_val_loss, out_dir, name="resume.pt"):
    state = model.state_dict()
    ckpt = {
        "model_state": state,
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict(),
        "scaler_state": scaler.state_dict(),
        "epoch": epoch,
        "global_step": global_step,
        "best_val_loss": best_val_loss,
    }
    torch.save(ckpt, Path(out_dir) / name)
    logger.info("Saved %s at step %d (epoch %d, best_val=%.4f)", name, global_step, epoch, best_val_loss)

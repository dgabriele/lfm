"""VAE pretrainer orchestrator and training loop."""

from __future__ import annotations

import atexit
import logging
import signal
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F  # noqa: N812
from torch import Tensor, nn

from .checkpoint import (
    _file_hash,
    load_resume_checkpoint,
    save_best_checkpoint,
    save_resume_checkpoint,
)
from .config import VAEPretrainConfig
from .data_setup import load_and_preprocess
from .diagnostics import run_epoch_diagnostics
from .forward import _dip_covariance_loss, _free_run_decode, _info_nce_loss, _vae_forward
from .model import build_model
from .validation import (
    log_epoch_summary,
    run_contrastive_alignment_diagnostic,
    run_validation,
)
from lfm.utils.oom import shrink_on_oom

logger = logging.getLogger(__name__)


def _derive_tag_ids(sp: Any, device: torch.device) -> tuple[Tensor, Tensor]:
    """Scan the text backend's vocabulary for `<TAG>` / `</TAG>` pairs and
    return two tensors of token IDs (opens, closes).  Only the SPM path
    is meaningful; phoneme tokenizers return empty tensors.  Constant
    across training, so called once at trainer startup."""
    opens: list[int] = []
    closes: list[int] = []
    if hasattr(sp, "id_to_piece") and hasattr(sp, "vocab_size"):
        for tid in range(int(sp.vocab_size())):
            piece = sp.id_to_piece(tid)
            if piece.startswith("</") and piece.endswith(">"):
                closes.append(tid)
            elif piece.startswith("<") and piece.endswith(">") and not piece.startswith("</"):
                # Skip SentencePiece's own <unk>, <s>, </s>, <pad> via
                # control-id check — user_defined_symbols are regular,
                # so we rely on the control check indirectly by
                # looking at piece text only.  This may include <unk>
                # in the open set, but v13 sets bos/eos/pad=-1 so only
                # <unk> would be picked up — and that's harmless
                # because its expected count is ~0 anyway.  Filter it
                # out explicitly for safety.
                if piece.lower() not in ("<unk>", "<s>", "<pad>"):
                    opens.append(tid)
    return (
        torch.tensor(opens, dtype=torch.long, device=device),
        torch.tensor(closes, dtype=torch.long, device=device),
    )


class VAEPretrainer:
    """Pretrain a VAE encoder-decoder on multilingual text data.

    Uses modular corpus loaders via the registry system, mixed precision
    training, and gradient accumulation for memory-constrained GPUs.

    Args:
        config: Pretraining configuration.
    """

    def __init__(self, config: VAEPretrainConfig) -> None:
        self.config = config

    def pretrain(self) -> dict[str, float]:
        """Run the full pretraining pipeline.

        Returns:
            Metrics dict with ``train_loss``, ``val_loss``,
            ``num_samples``, and ``active_latent_dims``.
        """
        cfg = self.config
        torch.manual_seed(cfg.seed)
        device = torch.device(cfg.device)

        # -- Data preprocessing --
        data, cfg = load_and_preprocess(cfg)
        output_dir = str(Path(cfg.output_path).parent)

        # Unpack frequently-used fields
        full_vocab = data.full_vocab
        bos_id = data.bos_id
        eos_id = data.eos_id
        vocab_size = data.vocab_size
        # Text backend: either SentencePieceProcessor (IPA/SPM path) or
        # PhonemeTokenizer (phoneme alphabet path).  Diagnostics decode
        # via _backend_decode, which duck-types both.
        sp = data.sp if data.sp is not None else data.phoneme_tokenizer
        spm_path = data.spm_path
        token_ids_list = data.token_ids_list
        languages_list = data.languages_list
        dataset = data.dataset
        train_loader = data.train_loader
        val_loader = data.val_loader
        val_dataset = data.val_dataset
        interleaved_loader = data.interleaved_loader
        corpus_embeddings = data.corpus_embeddings
        _use_contrastive = data.use_contrastive

        # 5. Build VAE model components
        hidden = cfg.decoder_hidden_dim
        modules = build_model(cfg, hidden, full_vocab, device)

        # Derive open/close tag ID tensors for the tag-balance auxiliary
        # loss.  Scans the SPM vocab once; `<XYZ>` goes to opens,
        # `</XYZ>` to closes.  Works only for the SPM path (phoneme
        # tokenizers have no bracket tags).  If tag_balance_weight=0
        # the tensors are ignored.
        _tag_open_ids, _tag_close_ids = _derive_tag_ids(sp, device)
        if getattr(cfg, "tag_balance_weight", 0.0) > 0:
            logger.info(
                "tag_balance_weight=%.3f: %d open / %d close tags detected",
                cfg.tag_balance_weight,
                int(_tag_open_ids.numel()),
                int(_tag_close_ids.numel()),
            )

        # 5b. Phonetic embedding initialization
        phonetic_sim_matrix: Tensor | None = None
        if cfg.phonetic_init or cfg.phonetic_label_smoothing > 0:
            from lfm.generator.phonetic_embeddings import (
                build_phonetic_similarity_matrix,
                build_token_feature_matrix,
                init_embeddings_from_features,
            )

            feature_matrix = build_token_feature_matrix(sp, vocab_size)
            if cfg.phonetic_init:
                init_embeddings_from_features(
                    modules["enc_token_embedding"], feature_matrix, cfg.phonetic_init_scale,
                )
                init_embeddings_from_features(
                    modules["dec_token_embedding"], feature_matrix, cfg.phonetic_init_scale,
                )
            if cfg.phonetic_label_smoothing > 0:
                phonetic_sim_matrix = build_phonetic_similarity_matrix(
                    feature_matrix
                )  # Keep on CPU — rows moved to GPU per-batch to save VRAM
                logger.info("Built phonetic similarity matrix for label smoothing")

        # Contrastive projection: z_dim -> embed_dim (if dimensions differ)
        contrastive_proj: nn.Module | None = None
        if _use_contrastive and corpus_embeddings is not None:
            embed_dim = corpus_embeddings.shape[1]
            if cfg.latent_dim != embed_dim:
                contrastive_proj = nn.Linear(cfg.latent_dim, embed_dim).to(device)
                logger.info(
                    "Created contrastive projection: %d -> %d", cfg.latent_dim, embed_dim,
                )

        all_params: list[nn.Parameter] = []
        for m in modules.values():
            if isinstance(m, nn.Module):
                all_params.extend(m.parameters())
        if contrastive_proj is not None:
            all_params.extend(contrastive_proj.parameters())

        optimizer = torch.optim.AdamW(all_params, lr=cfg.lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=cfg.num_epochs, eta_min=cfg.lr_min,
        )
        scaler = torch.amp.GradScaler(
            enabled=cfg.use_amp,
            init_scale=2**8,
            growth_factor=1.001,
            backoff_factor=0.5,
            growth_interval=100000,
        )

        # 5c. Phonetic distance cache for topo loss
        _topo_dist_cache = None
        if cfg.topo_weight > 0:
            from lfm.data.loaders.phonetic_distance import PhoneticDistanceCache
            _topo_dist_cache = PhoneticDistanceCache()

        # 6. Resume from checkpoint if available
        import math
        import time

        best_val_loss = float("inf")
        best_metrics: dict[str, float] = {}
        global_step = 0
        start_epoch = 0

        resume_path = Path(output_dir) / "vae_resume.pt"
        current_spm_hash = _file_hash(spm_path)
        if resume_path.exists():
            start_epoch, global_step, best_val_loss, _resume_batch = load_resume_checkpoint(
                resume_path,
                device=device,
                modules=modules,
                optimizer=optimizer,
                scheduler=scheduler,
                current_spm_hash=current_spm_hash,
                contrastive_proj=contrastive_proj,
            )
            cfg._resume_batch_idx = _resume_batch

        # 7. Save frozen config snapshot for provenance
        import yaml as _yaml

        config_snapshot_path = Path(output_dir) / "config.yaml"
        if not config_snapshot_path.exists():
            Path(output_dir).mkdir(parents=True, exist_ok=True)
            with open(config_snapshot_path, "w") as _cf:
                _yaml.dump(
                    cfg.model_dump() if hasattr(cfg, "model_dump") else vars(cfg),
                    _cf,
                    default_flow_style=False,
                )
            logger.info("Saved config snapshot to %s", config_snapshot_path)

        # 8. Training history
        from lfm.generator.training_history import TrainingHistory

        history = TrainingHistory(output_dir)
        history.start_session(
            start_epoch=start_epoch,
            config=cfg,
            spm_hash=current_spm_hash,
        )

        # -- Graceful shutdown: save history on SIGTERM/SIGINT/atexit --
        _session_ended = False
        _shutdown_state: dict[str, Any] = {
            "epoch": start_epoch,
            "best_val_loss": best_val_loss,
        }

        def _end_session_once() -> None:
            nonlocal _session_ended
            if _session_ended:
                return
            _session_ended = True
            history.end_session(
                end_epoch=_shutdown_state["epoch"],
                best_val_loss=_shutdown_state["best_val_loss"],
            )

        def _signal_handler(signum: int, frame: Any) -> None:
            _end_session_once()
            signal.signal(signum, signal.SIG_DFL)
            signal.raise_signal(signum)

        signal.signal(signal.SIGTERM, _signal_handler)
        signal.signal(signal.SIGINT, _signal_handler)
        atexit.register(_end_session_once)

        # Training loop setup
        accum = cfg.gradient_accumulation_steps
        log_every = 50
        num_batches = len(interleaved_loader) if interleaved_loader is not None else len(train_loader)
        if getattr(cfg, "max_batches_per_epoch", None) is not None:
            num_batches = min(num_batches, cfg.max_batches_per_epoch)

        total_params = sum(
            p.numel()
            for m in modules.values()
            if isinstance(m, nn.Module)
            for p in m.parameters()
        )
        logger.info(
            "Model: %d params (%.1fM), %d train batches/epoch",
            total_params, total_params / 1e6, num_batches,
        )

        z_running_mean = torch.zeros(cfg.latent_dim, device=device)
        z_running_std = torch.ones(cfg.latent_dim, device=device)
        z_stats_initialized = False
        z_stats_momentum = 0.01

        # Dynamic per-step batch cap.  Starts at the configured batch
        # size; :func:`shrink_on_oom` reduces it on CUDA OOM so rare
        # long-sequence batches (length-boosted sampling) don't kill
        # the run.  Incoming batches are sliced to this cap before
        # forward, so the DataLoader can keep its original batch size
        # and only the problematic tail is trimmed.
        _batch_cap = cfg.batch_size

        for epoch in range(start_epoch, cfg.num_epochs):
            # -- Word dropout annealing --
            if cfg.word_dropout > 0 and cfg.word_dropout_anneal_epochs > 0:
                wd_frac = min(epoch / cfg.word_dropout_anneal_epochs, 1.0)
                wd_p = cfg.word_dropout + wd_frac * (cfg.word_dropout_min - cfg.word_dropout)
            else:
                wd_p = cfg.word_dropout

            # -- Scheduled sampling annealing --
            ss_p = 0.0
            if (
                cfg.scheduled_sampling_target > 0
                and epoch >= cfg.scheduled_sampling_start_epoch
            ):
                ss_frac = min(
                    (epoch - cfg.scheduled_sampling_start_epoch)
                    / max(cfg.scheduled_sampling_warmup_epochs, 1),
                    1.0,
                )
                ss_p = ss_frac * cfg.scheduled_sampling_target

            # -- Train --
            for m in modules.values():
                if isinstance(m, nn.Module):
                    m.train()

            train_ce_sum = 0.0
            train_kl_sum = 0.0
            train_zvar_sum = 0.0
            train_dip_sum = 0.0
            train_cl_sum = 0.0
            train_klb_sum = 0.0
            train_vq_sum = 0.0
            train_bow_sum = 0.0
            train_tagbal_sum = 0.0
            train_acc_correct = 0
            train_acc_total = 0
            train_count = 0
            last_grad_norm = 0.0
            _BUCKET_NAMES = ["short(<20)", "med(20-50)", "long(>50)"]
            _bucket_ce_sum = [0.0, 0.0, 0.0]
            _bucket_ce_count = [0, 0, 0]
            _batch_bucket_ce_sum = [0.0, 0.0, 0.0]
            _batch_bucket_ce_count = [0, 0, 0]
            optimizer.zero_grad()
            epoch_start = time.time()
            batch_start = time.time()

            _epoch_loader = interleaved_loader if interleaved_loader is not None else train_loader

            # On resume mid-epoch, skip batches already trained.
            # The checkpoint stores the batch index within the epoch;
            # we fast-forward the dataloader to avoid redundant work.
            _resume_batch = 0
            if epoch == start_epoch and hasattr(cfg, '_resume_batch_idx'):
                _resume_batch = cfg._resume_batch_idx
                if _resume_batch > 0:
                    logger.info("Skipping %d batches (resume mid-epoch)", _resume_batch)

            for i, batch_data in enumerate(_epoch_loader):
                if i >= num_batches:
                    break
                if i < _resume_batch:
                    continue
                enc_tokens_override = None
                enc_lengths_override = None
                batch_indices = None
                is_constituent = False

                if interleaved_loader is not None:
                    is_constituent, raw_batch = batch_data
                    if is_constituent:
                        enc_tokens_override = raw_batch[0].to(device)
                        enc_lengths_override = torch.as_tensor(raw_batch[1], device=device)
                        batch_tokens = raw_batch[2].to(device)
                        batch_lengths = torch.as_tensor(raw_batch[3], device=device)
                    else:
                        batch_tokens = raw_batch[0].to(device)
                        batch_lengths = torch.as_tensor(raw_batch[1], device=device)
                elif _use_contrastive:
                    batch_tokens, batch_lengths, batch_indices = batch_data
                    batch_tokens = batch_tokens.to(device)
                    batch_lengths = torch.as_tensor(batch_lengths, device=device)
                else:
                    batch_tokens, batch_lengths = batch_data
                    batch_tokens = batch_tokens.to(device)
                    batch_lengths = torch.as_tensor(batch_lengths, device=device)

                # Apply dynamic per-step cap (may have shrunk on past OOM).
                if batch_tokens.size(0) > _batch_cap:
                    batch_tokens = batch_tokens[:_batch_cap]
                    batch_lengths = batch_lengths[:_batch_cap]
                    if batch_indices is not None:
                        batch_indices = batch_indices[:_batch_cap]
                    if enc_tokens_override is not None:
                        enc_tokens_override = enc_tokens_override[:_batch_cap]
                        enc_lengths_override = enc_lengths_override[:_batch_cap]
                b = batch_tokens.size(0)

                try:
                    with torch.amp.autocast(
                        device_type=device.type, enabled=cfg.use_amp,
                    ):
                        _do_kl = cfg.kl_weight > 0 or cfg.kl_beta > 0
                        (ce_loss, kl_loss, kl_per_dim_train,
                         z_batch, dec_hidden, mu_batch, logvar_batch,
                         vq_loss_batch, bow_loss, tag_balance_loss) = (
                            _vae_forward(
                                batch_tokens,
                                batch_lengths,
                                bos_id=bos_id,
                                full_vocab=full_vocab,
                                kl_free_bits=cfg.kl_free_bits,
                                compute_kl=_do_kl,
                                scheduled_sampling_p=ss_p,
                                _word_dropout_p=wd_p,
                                _phonetic_sim=phonetic_sim_matrix,
                                _phonetic_smoothing=cfg.phonetic_label_smoothing,
                                encoder_tokens=enc_tokens_override,
                                encoder_lengths=enc_lengths_override,
                                _tag_open_ids=_tag_open_ids,
                                _tag_close_ids=_tag_close_ids,
                                **modules,
                            )
                        )

                        # Track z distribution for latent calibration
                        with torch.no_grad():
                            batch_z_mean = z_batch.mean(dim=0)
                            batch_z_std = z_batch.std(dim=0).clamp(min=1e-6)
                            if not z_stats_initialized:
                                z_running_mean.copy_(batch_z_mean)
                                z_running_std.copy_(batch_z_std)
                                z_stats_initialized = True
                            else:
                                z_running_mean.lerp_(batch_z_mean, z_stats_momentum)
                                z_running_std.lerp_(batch_z_std, z_stats_momentum)

                        # KL: cyclical annealing
                        if _do_kl:
                            cycle_pos = global_step % max(cfg.kl_warmup_steps, 1)
                            kl_scale = (
                                min(cycle_pos / max(cfg.kl_warmup_steps, 1), 1.0)
                                * cfg.kl_weight
                            )
                        else:
                            kl_scale = 0.0

                        z_var_loss = torch.tensor(0.0, device=device)
                        if cfg.z_var_weight > 0 and b >= 4:
                            z_var = z_batch.var(dim=0)
                            z_var_loss = (z_var - cfg.z_var_target).pow(2).mean()

                        dip_loss = torch.tensor(0.0, device=device)
                        if cfg.dip_weight > 0 and b >= 4:
                            dip_loss = _dip_covariance_loss(z_batch)

                        cl_loss = torch.tensor(0.0, device=device)
                        if _use_contrastive and batch_indices is not None:
                            batch_embs = corpus_embeddings[batch_indices].to(device)
                            cl_loss = _info_nce_loss(
                                z_batch, batch_embs, cfg.contrastive_temperature,
                                projection=contrastive_proj,
                            )

                        kl_beta_loss = torch.tensor(0.0, device=device)
                        if cfg.kl_beta > 0:
                            raw_kl = 0.5 * (mu_batch.pow(2) + logvar_batch.exp() - 1 - logvar_batch)
                            kl_beta_loss = raw_kl.sum(dim=-1).mean()

                        vq_loss = (
                            vq_loss_batch if vq_loss_batch is not None
                            else torch.tensor(0.0, device=device)
                        )

                        loss = (
                            ce_loss + kl_scale * kl_loss
                            + vq_loss
                            + cfg.z_var_weight * z_var_loss
                            + cfg.dip_weight * dip_loss
                            + cfg.contrastive_weight * cl_loss
                            + cfg.kl_beta * kl_beta_loss
                            + cfg.bow_weight * bow_loss
                            + getattr(cfg, "tag_balance_weight", 0.0) * tag_balance_loss
                        ) / accum

                    scaler.scale(loss).backward()
                except RuntimeError as e:
                    # OOM recovery: shrink the per-step batch cap so
                    # subsequent batches are sliced down.  ``shrink_on_oom``
                    # clears the CUDA cache and zero-grads the optimizer.
                    # On repeated OOM we eventually hit the floor and
                    # surface the error.
                    _batch_cap = shrink_on_oom(
                        e, _batch_cap,
                        label=f"step {global_step}", optimizer=optimizer,
                    )
                    continue

                if (i + 1) % accum == 0 or (i + 1) == len(train_loader):
                    scaler.unscale_(optimizer)
                    last_grad_norm = nn.utils.clip_grad_norm_(
                        all_params, max_norm=5.0
                    ).item()
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()
                    global_step += 1

                # Token accuracy
                with torch.no_grad():
                    _logits = modules["output_head"](dec_hidden)
                    _preds = _logits.argmax(dim=-1)
                    _src_mask = (
                        torch.arange(batch_tokens.size(1), device=device).unsqueeze(0)
                        < batch_lengths.unsqueeze(1)
                    )
                    _correct = ((_preds == batch_tokens) & _src_mask).sum().item()
                    _total = _src_mask.sum().item()
                    _batch_acc = _correct / max(_total, 1)

                    _flat_logits = _logits.reshape(-1, full_vocab)
                    _flat_targets = batch_tokens.reshape(-1)
                    _per_tok_ce = F.cross_entropy(
                        _flat_logits, _flat_targets, reduction="none",
                    ).reshape(b, -1)
                    _per_sample_ce = (
                        (_per_tok_ce * _src_mask.float()).sum(dim=1)
                        / batch_lengths.float().clamp(min=1)
                    )
                    for _s_idx in range(b):
                        _slen = batch_lengths[_s_idx].item()
                        _bkt = 0 if _slen < 20 else (1 if _slen <= 50 else 2)
                        _bucket_ce_sum[_bkt] += _per_sample_ce[_s_idx].item()
                        _bucket_ce_count[_bkt] += 1
                        _batch_bucket_ce_sum[_bkt] += _per_sample_ce[_s_idx].item()
                        _batch_bucket_ce_count[_bkt] += 1

                train_ce_sum += ce_loss.item() * b
                train_kl_sum += kl_loss.item() * b
                train_zvar_sum += z_var_loss.item() * b
                train_dip_sum += dip_loss.item() * b
                train_cl_sum += cl_loss.item() * b
                train_klb_sum += kl_beta_loss.item() * b
                train_vq_sum += vq_loss.item() * b
                train_bow_sum += bow_loss.item() * b
                train_tagbal_sum += tag_balance_loss.item() * b
                train_acc_correct += _correct
                train_acc_total += _total
                train_count += b

                # -- Per-batch logging --
                if (i + 1) % log_every == 0 or (i + 1) == num_batches:
                    elapsed = time.time() - batch_start
                    tokens_per_sec = (
                        log_every * b * cfg.max_seq_len / max(elapsed, 0.01)
                    )
                    mem_mb = torch.cuda.memory_allocated(device) / 1e6 if device.type == "cuda" else 0.0
                    ppl = min(math.exp(ce_loss.item()), 99999.0)
                    extra_parts: list[str] = [
                        f"PPL={ppl:.2f}", f"acc={_batch_acc:.1%}", f"gnorm={last_grad_norm:.2f}",
                    ]
                    if _do_kl:
                        raw_kl = kl_per_dim_train.detach()
                        active = int((raw_kl.mean(dim=0) > 0.1).sum().item())
                        extra_parts.append(
                            f"KL={kl_loss.item():.3f} kl_scale={kl_scale:.4f} active={active}/{cfg.latent_dim}"
                        )
                    if z_stats_initialized:
                        extra_parts.append(f"z_std={z_running_std.mean().item():.4f}")
                    if cfg.z_var_weight > 0:
                        extra_parts.append(f"zvar={z_var_loss.item():.4f}")
                    if cfg.dip_weight > 0:
                        extra_parts.append(f"dip={dip_loss.item():.6f}")
                    if _use_contrastive:
                        extra_parts.append(f"CL={cl_loss.item():.4f}")
                    if cfg.kl_beta > 0:
                        extra_parts.append(f"KL\u03b2={kl_beta_loss.item():.4f}")
                    if cfg.use_vq:
                        rvq = modules["_residual_vq"]
                        extra_parts.append(f"VQ={vq_loss.item():.4f}")
                        if hasattr(rvq, "quant_errors"):
                            qe = rvq.quant_errors
                            extra_parts.append(f"qe={sum(qe)/len(qe):.4f}")
                        if hasattr(rvq, "_last_balance_loss"):
                            extra_parts.append(f"bal={rvq._last_balance_loss:.3f}")
                        _vq_util_str = " ".join(f"{u:.0%}" for u in rvq.utilization)
                    if cfg.bow_weight > 0:
                        extra_parts.append(f"BoW={bow_loss.item():.3f}")
                    if getattr(cfg, "tag_balance_weight", 0.0) > 0:
                        extra_parts.append(f"TagBal={tag_balance_loss.item():.4f}")
                    extra_str = (" " + " ".join(extra_parts)) if extra_parts else ""
                    logger.info(
                        "  ep%d batch=%d/%d step=%d CE=%.3f %.0f tok/s %.0fMB%s",
                        epoch, i + 1, num_batches, global_step, ce_loss.item(),
                        tokens_per_sec, mem_mb, extra_str,
                    )
                    if cfg.use_vq:
                        logger.info("    util: %s", _vq_util_str)
                    _bkt_parts = []
                    for _bi in range(3):
                        if _batch_bucket_ce_count[_bi] > 0:
                            _bkt_ce = _batch_bucket_ce_sum[_bi] / _batch_bucket_ce_count[_bi]
                            _bkt_parts.append(
                                f"{_BUCKET_NAMES[_bi]}={_bkt_ce:.2f}({_batch_bucket_ce_count[_bi]})"
                            )
                    if _bkt_parts:
                        logger.info("    CE by len: %s", " | ".join(_bkt_parts))
                    _batch_bucket_ce_sum = [0.0, 0.0, 0.0]
                    _batch_bucket_ce_count = [0, 0, 0]
                    batch_start = time.time()

                # Mid-epoch diagnostics
                if (
                    cfg.diagnostic_every > 0
                    and (i + 1) % cfg.diagnostic_every == 0
                    and (i + 1) < num_batches
                ):
                    for m in modules.values():
                        if isinstance(m, nn.Module):
                            m.eval()
                    run_epoch_diagnostics(
                        epoch=epoch,
                        modules=modules,
                        cfg=cfg,
                        val_dataset=data.val_dataset,
                        val_loader=data.val_loader,
                        dataset=data.dataset,
                        languages_list=data.languages_list,
                        bos_id=bos_id,
                        eos_id=eos_id,
                        vocab_size=vocab_size,
                        full_vocab=full_vocab,
                        sp=sp,
                        z_running_mean=z_running_mean,
                        z_running_std=z_running_std,
                        device=device,
                        label=f"step {global_step}",
                        constituent_dataset=data.constituent_dataset,
                    )
                    for m in modules.values():
                        if isinstance(m, nn.Module):
                            m.train()

                # Mid-epoch resumable checkpoint
                _ckpt_every = getattr(cfg, "checkpoint_every_steps", None)
                if (
                    _ckpt_every is not None
                    and (i + 1) % _ckpt_every == 0
                    and (i + 1) < num_batches
                ):
                    save_resume_checkpoint(
                        str(Path(cfg.output_path).parent),
                        epoch=epoch,
                        global_step=global_step,
                        best_val_loss=best_val_loss,
                        spm_path=spm_path,
                        z_running_mean=z_running_mean,
                        z_running_std=z_running_std,
                        modules=modules,
                        optimizer=optimizer,
                        scheduler=scheduler,
                        scaler=scaler,
                        cfg=cfg,
                        batch_idx=i + 1,
                    )
                    logger.info("Mid-epoch checkpoint at step %d (batch %d)", global_step, i + 1)

            train_ce = train_ce_sum / max(train_count, 1)
            train_kl = train_kl_sum / max(train_count, 1)
            train_zvar = train_zvar_sum / max(train_count, 1)
            train_dip = train_dip_sum / max(train_count, 1)
            train_cl = train_cl_sum / max(train_count, 1)
            train_klb = train_klb_sum / max(train_count, 1)
            train_vq = train_vq_sum / max(train_count, 1)
            train_bow = train_bow_sum / max(train_count, 1)
            train_loss = (
                train_ce + cfg.kl_weight * train_kl
                + cfg.z_var_weight * train_zvar + cfg.dip_weight * train_dip
                + cfg.contrastive_weight * train_cl + cfg.kl_beta * train_klb
                + cfg.bow_weight * train_bow
            )
            train_acc = train_acc_correct / max(train_acc_total, 1)

            # -- Validate --
            val = run_validation(
                cfg=cfg, modules=modules, val_loader=val_loader,
                device=device, full_vocab=full_vocab, bos_id=bos_id,
                do_kl=_do_kl,
            )
            val_loss = val.val_loss
            epoch_time = time.time() - epoch_start

            # -- Epoch summary --
            log_epoch_summary(
                epoch=epoch, cfg=cfg, epoch_time=epoch_time,
                train_ce=train_ce, train_kl=train_kl,
                train_zvar=train_zvar, train_dip=train_dip,
                train_cl=train_cl, train_klb=train_klb,
                train_bow=train_bow, train_vq=train_vq,
                train_loss=train_loss, train_acc=train_acc,
                val=val, z_running_std=z_running_std,
                ss_p=ss_p, wd_p=wd_p, do_kl=_do_kl,
                use_contrastive=_use_contrastive, modules=modules,
                scheduler=scheduler,
                bucket_ce_sum=_bucket_ce_sum,
                bucket_ce_count=_bucket_ce_count,
                train_count=train_count,
            )

            # -- Contrastive alignment diagnostic --
            if _use_contrastive and corpus_embeddings is not None:
                run_contrastive_alignment_diagnostic(
                    epoch=epoch, cfg=cfg, modules=modules,
                    train_loader=train_loader, device=device,
                    full_vocab=full_vocab, bos_id=bos_id,
                    corpus_embeddings=corpus_embeddings,
                    use_contrastive=_use_contrastive,
                )

            # -- Epoch-end diagnostics --
            with torch.no_grad():
                run_epoch_diagnostics(
                    epoch=epoch, cfg=cfg, modules=modules,
                    device=device, vocab_size=vocab_size,
                    bos_id=bos_id, eos_id=eos_id, sp=sp,
                    val_dataset=val_dataset, val_loader=val_loader,
                    dataset=dataset, languages_list=languages_list,
                    z_running_mean=z_running_mean,
                    z_running_std=z_running_std,
                )

            # Save best checkpoint
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_metrics = {
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                    "val_ce": val.val_ce,
                    "val_kl": val.val_kl,
                    "num_samples": float(len(token_ids_list)),
                    "epoch": float(epoch + 1),
                }
                save_best_checkpoint(
                    cfg, modules, vocab_size=vocab_size,
                    train_loss=train_loss, val_loss=val_loss,
                    z_running_mean=z_running_mean,
                    z_running_std=z_running_std,
                    spm_path=spm_path,
                )

            # -- Early termination checks --
            if val_loss > best_val_loss * 2.0 and epoch > 5:
                logger.warning(
                    "EARLY STOP: val_loss=%.4f is >2x best=%.4f — "
                    "gradient explosion detected. Best checkpoint preserved.",
                    val_loss, best_val_loss,
                )
                break

            z_std_mean = z_running_std.mean().item()
            if epoch > 5 and z_std_mean < 0.001 and val_loss >= best_val_loss:
                logger.warning(
                    "EARLY STOP: z_std=%.6f below 0.001 with no val improvement — "
                    "latent space collapse. Best checkpoint preserved.",
                    z_std_mean,
                )
                break

            scheduler.step()

            _shutdown_state["epoch"] = epoch + 1
            _shutdown_state["best_val_loss"] = best_val_loss
            history.update_epoch(epoch + 1, best_val_loss)

            save_resume_checkpoint(
                output_dir, epoch=epoch + 1, global_step=global_step,
                best_val_loss=best_val_loss, spm_path=spm_path,
                z_running_mean=z_running_mean,
                z_running_std=z_running_std,
                modules=modules, optimizer=optimizer,
                scheduler=scheduler, scaler=scaler,
                contrastive_proj=contrastive_proj,
                cfg=cfg,
            )

        _shutdown_state["epoch"] = epoch + 1 if epoch >= start_epoch else start_epoch
        _shutdown_state["best_val_loss"] = best_val_loss
        _end_session_once()
        atexit.unregister(_end_session_once)
        return best_metrics


def pretrain_vae_decoder(config: VAEPretrainConfig) -> dict[str, float]:
    """Convenience function: create ``VAEPretrainer`` and run.

    Args:
        config: Pretraining configuration.

    Returns:
        Metrics dict with ``train_loss``, ``val_loss``,
        ``active_latent_dims``, and ``num_samples``.
    """
    if not logging.root.handlers:
        import sys

        sys.stderr = open(sys.stderr.fileno(), "w", buffering=1, closefd=False)
        handler = logging.StreamHandler(sys.stderr)
        handler.setFormatter(
            logging.Formatter("%(asctime)s %(name)s %(message)s")
        )
        handler.setLevel(logging.INFO)
        logging.root.addHandler(handler)
        logging.root.setLevel(logging.INFO)
    return VAEPretrainer(config).pretrain()

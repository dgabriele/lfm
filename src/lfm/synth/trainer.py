"""Two-phase trainer for SynthLM.

AlienLMTrainer (Phase 1):
    Cipher-encodes English text and trains the model as a causal alien LM.
    Trains: _alien_emb, _alien_head, body (with differential learning rates).

ConditioningTrainer (Phase 2):
    Freezes Phase 1 weights; trains PrefixProjector + LengthHead on
    (source_embedding → alien text) pairs.
"""

from __future__ import annotations

import itertools
import json
import logging
import math
import os
import random
from pathlib import Path
from typing import Iterator

import h5py
import numpy as np
import torch
from torch.optim import AdamW
from transformers import PreTrainedTokenizerFast

from lfm.synth.cipher import WordCipher
from lfm.synth.config import SynthConfig
from lfm.synth.model import SynthLM

logger = logging.getLogger(__name__)


# ── data readers ─────────────────────────────────────────────────────────────

def _iter_raw_sentences(dataset_path: str, shuffle: bool = True, seed: int = 42) -> Iterator[str]:
    """Yield raw English strings indefinitely from HDF5, .jsonl, or .txt."""
    path = Path(dataset_path)
    rng = random.Random(seed)

    if path.is_dir():
        with h5py.File(path / "samples.h5", "r") as f:
            raws = [x.decode("utf-8") if isinstance(x, bytes) else x for x in f["samples"]["raw"][:]]
    elif path.suffix == ".jsonl":
        raws = [json.loads(l)["text"] for l in path.read_text().splitlines() if l.strip()]
    else:
        raws = [l.strip() for l in path.read_text().splitlines() if l.strip()]

    if shuffle:
        rng.shuffle(raws)
    idx, n = 0, len(raws)
    while True:
        yield raws[idx % n]
        idx += 1
        if shuffle and idx % n == 0:
            rng.shuffle(raws)


def _load_all_sentences(dataset_path: str) -> list[str]:
    """Load every sentence from the corpus into memory (no shuffle, no loop)."""
    path = Path(dataset_path)
    if path.is_dir():
        with h5py.File(path / "samples.h5", "r") as f:
            return [x.decode("utf-8") if isinstance(x, bytes) else x for x in f["samples"]["raw"][:]]
    if path.suffix == ".jsonl":
        return [json.loads(l)["text"] for l in path.read_text().splitlines() if l.strip()]
    return [l.strip() for l in path.read_text().splitlines() if l.strip()]


def _iter_in_memory(items: list[str], shuffle: bool, seed: int) -> Iterator[str]:
    """Cycle through an in-memory list indefinitely with optional reshuffle each pass."""
    rng = random.Random(seed)
    pool = list(items)
    if shuffle:
        rng.shuffle(pool)
    idx, n = 0, len(pool)
    while True:
        yield pool[idx % n]
        idx += 1
        if shuffle and idx % n == 0:
            rng.shuffle(pool)


def _load_store(store_dir: str) -> tuple[np.ndarray, list[str]]:
    """Load embeddings + texts from an embedding store directory."""
    store_dir = Path(store_dir)
    embeddings = np.load(store_dir / "embeddings.npy", mmap_mode="r")
    texts = [
        json.loads(l)["text"]
        for l in (store_dir / "passages.jsonl").read_text().splitlines()
        if l.strip()
    ]
    assert len(texts) == len(embeddings), (
        f"embeddings ({len(embeddings)}) and passages ({len(texts)}) count mismatch"
    )
    return embeddings, texts


# ── Phase 1 ───────────────────────────────────────────────────────────────────

class AlienLMTrainer:
    """Train the model as a causal alien LM on cipher-encoded English text.

    Uses differential learning rates: phase1_lr for alien emb/head (random init),
    phase1_body_lr for transformer layers (pretrained, nudge gently).
    """

    _DIAG_N = 16
    _SAMPLE_N = 5

    def __init__(
        self,
        model: SynthLM,
        config: SynthConfig,
        cipher: WordCipher,
        alien_tokenizer: PreTrainedTokenizerFast,
    ) -> None:
        self.model = model
        self.config = config
        self.cipher = cipher
        self.alien_tok = alien_tokenizer
        self.device = torch.device(config.device)
        model.backend.unfreeze_body()
        self.opt = AdamW([
            {"params": model.backend.cipher_params(), "lr": config.phase1_lr},
            {"params": model.backend.body_params(),   "lr": config.phase1_body_lr},
        ])
        self.mse_weight = config.phase1_hidden_mse_weight
        self.grad_accum = max(1, config.phase1_grad_accum)
        self.warmup_steps = max(0, config.phase1_body_warmup_steps)
        self._body_frozen = False  # mirrors the actual freeze state of the body
        logger.info(
            "AlienLMTrainer  model=%s  batch=%d×%d  cipher_lr=%g  body_lr=%g  "
            "lr_min=%g  lr_schedule=%s  body_warmup=%d  mse_weight=%g  steps=%d  "
            "max_len=%d  filter_truncated=%s  diag_every=%d  device=%s  out=%s",
            config.base_model_name, config.phase1_batch_size, self.grad_accum,
            config.phase1_lr, config.phase1_body_lr, config.phase1_lr_min,
            config.phase1_lr_schedule, self.warmup_steps, self.mse_weight,
            config.phase1_steps, config.phase1_max_len, config.phase1_filter_truncated,
            config.phase1_diag_every, config.device, config.output_dir,
        )
        self._diag_sentences = list(itertools.islice(
            _iter_raw_sentences(config.phase1_dataset_dir, shuffle=True, seed=0), self._DIAG_N,
        ))

    def _scheduled_lr(self, step: int, peak_lr: float) -> float:
        """Cosine decay from peak_lr to phase1_lr_min over phase1_steps. Constant if disabled."""
        cfg = self.config
        if cfg.phase1_lr_schedule != "cosine":
            return peak_lr
        progress = min(1.0, step / max(cfg.phase1_steps, 1))
        return cfg.phase1_lr_min + 0.5 * (peak_lr - cfg.phase1_lr_min) * (1 + math.cos(math.pi * progress))

    def _apply_schedule(self, step: int) -> tuple[float, float]:
        """Set per-group LRs and freeze/unfreeze body to match warmup phase.

        Freezing the body (requires_grad=False) during warmup avoids allocating
        gradient and Adam-state memory for the body — important on tight VRAM.
        """
        in_warmup = step < self.warmup_steps
        if in_warmup and not self._body_frozen:
            self.model.backend.freeze_body()
            self._body_frozen = True
            logger.info("body FROZEN at step %d (warmup until step %d)", step, self.warmup_steps)
        elif not in_warmup and self._body_frozen:
            self.model.backend.unfreeze_body()
            self._body_frozen = False
            logger.info("body UNFROZEN at step %d", step)
        cipher_lr = self._scheduled_lr(step, self.config.phase1_lr)
        body_lr = 0.0 if in_warmup else self._scheduled_lr(step, self.config.phase1_body_lr)
        self.opt.param_groups[0]["lr"] = cipher_lr
        self.opt.param_groups[1]["lr"] = body_lr
        return cipher_lr, body_lr

    def _filter_truncated(self, sentences: list[str]) -> list[str]:
        """Drop sentences whose cipher tokenisation would exceed phase1_max_len."""
        ml = self.config.phase1_max_len
        cipher_texts = self.cipher.encode_batch(sentences)
        encs = self.alien_tok(cipher_texts, padding=False, truncation=False, add_special_tokens=True)["input_ids"]
        return [s for s, ids in zip(sentences, encs) if len(ids) <= ml]

    def _make_batch(self, sentences: list[str]) -> dict[str, torch.Tensor]:
        dec = self.alien_tok(
            self.cipher.encode_batch(sentences),
            padding=True, truncation=True,
            max_length=self.config.phase1_max_len, return_tensors="pt",
        )
        labels = dec["input_ids"].clone()
        labels[labels == self.alien_tok.pad_token_id] = -100
        return {
            "alien_ids":    dec["input_ids"].to(self.device),
            "alien_labels": labels.to(self.device),
        }

    @torch.no_grad()
    def _run_diagnostics(self, step: int) -> None:
        self.model.eval()
        batch = self._make_batch(self._diag_sentences)
        inputs_embeds = self.model.backend.embed_alien(batch["alien_ids"][:, :-1])
        hidden = self.model.backend.forward_hidden(inputs_embeds)
        logits = self.model.backend.alien_logits(hidden)
        pred_ids = logits.argmax(dim=-1)
        targets = batch["alien_labels"][:, 1:]
        valid = targets != -100
        lm_acc = (pred_ids[valid] == targets[valid]).float().mean().item()
        body_drift = self._body_drift(inputs_embeds, hidden) if self.model.backend.has_reference else None
        if body_drift is not None:
            cos, rel_rmse = body_drift
            logger.info(
                "phase1 diag  step=%d  lm_acc=%.3f  entropy=%.2f  rep_rate=%.3f  body_cos=%.4f  body_rel_rmse=%.3f",
                step, lm_acc, self._entropy(pred_ids[valid]), self._rep_rate(pred_ids, valid), cos, rel_rmse,
            )
        else:
            logger.info(
                "phase1 diag  step=%d  lm_acc=%.3f  entropy=%.2f  rep_rate=%.3f",
                step, lm_acc, self._entropy(pred_ids[valid]), self._rep_rate(pred_ids, valid),
            )
        self.model.train()

    def _body_drift(self, inputs_embeds: torch.Tensor, hidden: torch.Tensor) -> tuple[float, float]:
        """Cosine similarity (direction) and relative RMSE (magnitude) between
        trainable body and frozen reference body hidden states. Scale-aware drift readout."""
        ref_hidden = self.model.backend.reference_hidden(inputs_embeds)
        cos = torch.nn.functional.cosine_similarity(hidden, ref_hidden, dim=-1).mean().item()
        rel_rmse = (
            (hidden - ref_hidden).pow(2).mean().sqrt()
            / ref_hidden.pow(2).mean().sqrt().clamp(min=1e-8)
        ).item()
        return cos, rel_rmse

    @torch.no_grad()
    def _log_samples(self, step: int) -> None:
        """Log 5 English/alien pairs: ground-truth cipher vs model's autoregressive output."""
        self.model.eval()
        sentences = self._diag_sentences[:self._SAMPLE_N]
        eos_id = self.alien_tok.eos_token_id
        pad_id = self.alien_tok.pad_token_id

        for sent in sentences:
            ground_truth = self.cipher.encode_sentence(sent)
            # Seed with first cipher token, generate the rest autoregressively.
            seed_ids = self.alien_tok(
                self.cipher.encode_for_tokenizer(sent),
                return_tensors="pt",
            )["input_ids"][:, :1].to(self.device)
            context = self.model.backend.embed_alien(seed_ids)
            generated = [seed_ids[0, 0].item()]
            for _ in range(self.config.phase1_max_len):
                hidden  = self.model.backend.forward_hidden(context)
                logits  = self.model.backend.alien_logits(hidden[:, -1:]).squeeze(1)
                probs   = torch.softmax(logits.float(), dim=-1)
                next_id = torch.multinomial(probs, num_samples=1).squeeze(-1)
                generated.append(next_id.item())
                if next_id.item() == eos_id:
                    break
                context = torch.cat([context, self.model.backend.embed_alien(next_id.unsqueeze(1))], dim=1)
            gen_text = self.alien_tok.decode(generated, skip_special_tokens=True)
            logger.info("sample  EN: %s", sent)
            logger.info("        GT: %s", ground_truth)
            logger.info("       GEN: %s", gen_text)

        self.model.train()

    @staticmethod
    def _entropy(ids: torch.Tensor) -> float:
        counts = torch.bincount(ids, minlength=1).float()
        probs = counts / counts.sum()
        return -(probs * (probs + 1e-9).log()).sum().item()

    @staticmethod
    def _rep_rate(pred_ids: torch.Tensor, valid: torch.Tensor) -> float:
        num = den = 0
        for b in range(pred_ids.size(0)):
            toks = pred_ids[b][valid[b]]
            if toks.numel() > 1:
                num += (toks[1:] == toks[:-1]).sum().item()
                den += toks.numel() - 1
        return num / den if den > 0 else 0.0

    def train(self, start_step: int = 0) -> None:
        cfg = self.config
        out_dir = Path(cfg.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        self.model.to(self.device).train()
        self.model.backend.move_reference_to(self.device, self.model.backend.dtype)

        # Build the training corpus once. If filter_truncated is set, drop any
        # sentence whose tokenised form would exceed max_len — keeps every batch
        # composed of naturally-terminating samples (no truncation artefacts).
        if cfg.phase1_filter_truncated:
            all_sents = _load_all_sentences(cfg.phase1_dataset_dir)
            n_before = len(all_sents)
            logger.info("filtering corpus by tokenised length ≤ %d (%d sentences)...", cfg.phase1_max_len, n_before)
            filtered: list[str] = []
            chunk = 1024
            for i in range(0, n_before, chunk):
                filtered.extend(self._filter_truncated(all_sents[i : i + chunk]))
            logger.info("kept %d / %d sentences (%.1f%%)", len(filtered), n_before, 100 * len(filtered) / max(n_before, 1))
            data = _iter_in_memory(filtered, shuffle=True, seed=cfg.seed)
        else:
            data = _iter_raw_sentences(cfg.phase1_dataset_dir, seed=cfg.seed)

        for _ in range(start_step * cfg.phase1_batch_size * self.grad_accum):
            next(data)

        step, running_ce, running_mse = start_step, 0.0, 0.0

        while step < cfg.phase1_steps:
            cipher_lr, body_lr = self._apply_schedule(step)
            try:
                self.opt.zero_grad()
                accum_ce = accum_mse = 0.0
                for _ in range(self.grad_accum):
                    batch = self._make_batch([next(data) for _ in range(cfg.phase1_batch_size)])
                    ce, mse = self.model.forward_phase1(**batch)
                    total = (ce + self.mse_weight * mse) / self.grad_accum
                    total.backward()
                    accum_ce += ce.item() / self.grad_accum
                    accum_mse += mse.item() / self.grad_accum
                torch.nn.utils.clip_grad_norm_(
                    self.model.backend.cipher_params() + self.model.backend.body_params(), 1.0
                )
                self.opt.step()
            except torch.cuda.OutOfMemoryError as e:
                logger.warning("CUDA OOM at step %d (cipher_lr=%g body_lr=%g): %s", step, cipher_lr, body_lr, e)
                torch.cuda.empty_cache()
                # If micro-batch is already at 4, give up — the user must intervene.
                if cfg.phase1_batch_size <= 4:
                    raise
                new_bs = max(4, cfg.phase1_batch_size // 2)
                new_accum = self.grad_accum * (cfg.phase1_batch_size // new_bs)
                logger.warning("OOM recovery: batch %d → %d, grad_accum %d → %d (effective batch unchanged)",
                               cfg.phase1_batch_size, new_bs, self.grad_accum, new_accum)
                # Mutate config in-memory; persists for the rest of the run.
                object.__setattr__(cfg, "phase1_batch_size", new_bs)
                self.grad_accum = new_accum
                continue  # retry this step with the smaller batch

            running_ce += accum_ce
            running_mse += accum_mse
            step += 1

            if step % cfg.phase1_log_every == 0:
                n = cfg.phase1_log_every
                logger.info(
                    "phase1 step=%d  ce=%.4f  mse=%.6f  cipher_lr=%.2e  body_lr=%.2e",
                    step, running_ce / n, running_mse / n, cipher_lr, body_lr,
                )
                running_ce = running_mse = 0.0

            if cfg.phase1_diag_every > 0 and step % cfg.phase1_diag_every == 0:
                self._run_diagnostics(step)

            if step % cfg.phase1_checkpoint_every == 0:
                ckpt = str(out_dir / "phase1_checkpoint.pt")
                self._save_checkpoint(ckpt, step)
                logger.info("phase1 checkpoint -> %s  (step %d)", ckpt, step)
                self._log_samples(step)

        self.model.save_phase1(str(out_dir / "phase1_final.pt"))
        logger.info("phase1 training complete")

    def _save_checkpoint(self, path: str, step: int) -> None:
        tmp = path + ".tmp"
        torch.save({"step": step, "model": self.model.phase1_state(), "optimizer": self.opt.state_dict()}, tmp)
        os.replace(tmp, path)

    def load_checkpoint(self, path: str) -> int:
        ckpt = torch.load(path, map_location="cpu", weights_only=True)
        self.model.load_phase1_state(ckpt["model"])
        self.opt.load_state_dict(ckpt["optimizer"])
        for state in self.opt.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(self.device)
        return ckpt["step"]


# ── Phase 2 ───────────────────────────────────────────────────────────────────

class ConditioningTrainer:
    """Train PrefixProjector + LengthHead. Phase 1 weights are frozen."""

    def __init__(
        self,
        model: SynthLM,
        config: SynthConfig,
        cipher: WordCipher,
        alien_tokenizer: PreTrainedTokenizerFast,
    ) -> None:
        self.model = model
        self.config = config
        self.cipher = cipher
        self.alien_tok = alien_tokenizer
        self.device = torch.device(config.device)
        self._freeze_phase1()
        self.opt = AdamW(
            list(model.projector.parameters()) + list(model.length_head.parameters()),
            lr=config.phase2_lr,
        )
        logger.info(
            "ConditioningTrainer  batch=%d  lr=%g  steps=%d  len_weight=%g  device=%s  out=%s",
            config.phase2_batch_size, config.phase2_lr, config.phase2_steps,
            config.phase2_length_loss_weight, config.device, config.output_dir,
        )

    def _freeze_phase1(self) -> None:
        self.model.backend.freeze_body()
        for p in self.model.backend._alien_emb.parameters(): p.requires_grad_(False)
        for p in self.model.backend._alien_head.parameters(): p.requires_grad_(False)

    def _make_batch(
        self, embeddings: np.ndarray, texts: list[str], indices: list[int]
    ) -> dict[str, torch.Tensor]:
        dec = self.alien_tok(
            self.cipher.encode_batch([texts[i] for i in indices]),
            padding=True, truncation=True,
            max_length=self.config.phase2_max_len, return_tensors="pt",
        )
        labels = dec["input_ids"].clone()
        labels[labels == self.alien_tok.pad_token_id] = -100
        emb = torch.tensor(embeddings[indices].astype(np.float32), dtype=torch.float32)
        return {
            "source_embedding": emb.to(self.device),
            "alien_ids":        dec["input_ids"].to(self.device),
            "alien_labels":     labels.to(self.device),
        }

    def train(self, start_step: int = 0) -> None:
        cfg = self.config
        out_dir = Path(cfg.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        self.model.to(self.device).train()

        embeddings, texts = _load_store(cfg.phase2_store_dir)
        N = len(texts)
        rng = random.Random(cfg.seed)
        for _ in range(start_step):
            rng.sample(range(N), min(cfg.phase2_batch_size, N))

        step, running_lm, running_len = start_step, 0.0, 0.0
        w = cfg.phase2_length_loss_weight

        while step < cfg.phase2_steps:
            batch = self._make_batch(embeddings, texts, rng.sample(range(N), min(cfg.phase2_batch_size, N)))
            self.opt.zero_grad()
            lm_loss, len_loss = self.model.forward_phase2(**batch)
            (lm_loss + w * len_loss).backward()
            torch.nn.utils.clip_grad_norm_(
                list(self.model.projector.parameters()) + list(self.model.length_head.parameters()), 1.0
            )
            self.opt.step()
            running_lm += lm_loss.item()
            running_len += len_loss.item()
            step += 1

            if step % cfg.phase2_log_every == 0:
                n = cfg.phase2_log_every
                logger.info("phase2 step=%d  lm=%.4f  len=%.4f", step, running_lm / n, running_len / n)
                running_lm = running_len = 0.0

            if step % cfg.phase2_checkpoint_every == 0:
                ckpt = str(out_dir / "phase2_checkpoint.pt")
                self._save_checkpoint(ckpt, step)
                logger.info("phase2 checkpoint -> %s  (step %d)", ckpt, step)

        self.model.save_phase2(str(out_dir / "phase2_final.pt"))
        logger.info("phase2 training complete")

    def _save_checkpoint(self, path: str, step: int) -> None:
        tmp = path + ".tmp"
        torch.save({"step": step, "model": self.model.phase2_state(), "optimizer": self.opt.state_dict()}, tmp)
        os.replace(tmp, path)

    def load_checkpoint(self, path: str) -> int:
        ckpt = torch.load(path, map_location="cpu", weights_only=True)
        self.model.load_phase2_state(ckpt["model"])
        self.opt.load_state_dict(ckpt["optimizer"])
        for state in self.opt.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(self.device)
        return ckpt["step"]

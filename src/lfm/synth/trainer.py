"""Two-phase trainer for SynthLM (decoder-only architecture).

Phase 1 — CipherTrainer:
    Reads English text, applies word cipher, trains the backend's alien
    embedding table and alien projection head on (English → alien) pairs.
    The LM body is always frozen; only _alien_emb + _alien_head train.

Phase 2 — ConditioningTrainer:
    Reads (embedding, passage) pairs from the embedding store.
    Freezes alien emb/head from Phase 1; trains only PrefixProjector + LengthHead.
"""

from __future__ import annotations

import itertools
import json
import logging
import random
from pathlib import Path
from typing import Iterator

import h5py
import numpy as np
import torch
from torch.optim import AdamW
from transformers import AutoTokenizer, PreTrainedTokenizerFast

from lfm.synth.cipher import WordCipher
from lfm.synth.config import SynthConfig
from lfm.synth.model import SynthLM

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data readers
# ---------------------------------------------------------------------------

def _iter_raw_sentences(dataset_path: str, shuffle: bool = True, seed: int = 42) -> Iterator[str]:
    """Yield raw English strings indefinitely from HDF5, .jsonl, or .txt."""
    path = Path(dataset_path)
    rng = random.Random(seed)

    if path.is_dir():
        with h5py.File(path / "samples.h5", "r") as f:
            raws = [
                x.decode("utf-8") if isinstance(x, bytes) else x
                for x in f["samples"]["raw"][:]
            ]
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


def _load_store(store_dir: str) -> tuple[np.ndarray, list[str]]:
    """Load embeddings + passage texts from an embedding store directory."""
    store_dir = Path(store_dir)
    embeddings = np.load(store_dir / "embeddings.npy", mmap_mode="r")
    texts = [
        json.loads(line)["text"]
        for line in (store_dir / "passages.jsonl").read_text().splitlines()
        if line.strip()
    ]
    assert len(texts) == len(embeddings), (
        f"embeddings ({len(embeddings)}) and passages ({len(texts)}) count mismatch"
    )
    return embeddings, texts


# ---------------------------------------------------------------------------
# Phase 1 — Cipher fine-tuning
# ---------------------------------------------------------------------------

class CipherTrainer:
    """Train backend._alien_emb + _alien_head to produce alien tokens from English.

    The LM body is frozen at CausalDecoderBackend construction time and never
    modified here.  Only the alien cipher sub-vocabulary trains.
    """

    _DIAG_N = 16

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
        self.native_tok = AutoTokenizer.from_pretrained(config.base_model_name)
        self.device = torch.device(config.device)
        # Unfreeze body so the transformer layers learn to produce alien tokens.
        # alien_emb+head use phase1_lr (random init, needs fast updates);
        # body layers use phase1_body_lr (pretrained, nudge gently).
        model.backend.unfreeze_body()
        self.opt = AdamW([
            {"params": model.backend.cipher_params(), "lr": config.phase1_lr},
            {"params": model.backend.body_params(),   "lr": config.phase1_body_lr},
        ])
        logger.info(
            "CipherTrainer  model=%s  batch=%d  cipher_lr=%g  body_lr=%g  steps=%d  "
            "src_len=%d  tgt_len=%d  diag_every=%d  device=%s  out=%s",
            config.base_model_name, config.phase1_batch_size,
            config.phase1_lr, config.phase1_body_lr,
            config.phase1_steps, config.phase1_max_source_len,
            config.phase1_max_target_len, config.phase1_diag_every,
            config.device, config.output_dir,
        )
        if config.phase1_diag_every > 0:
            self._diag_sentences = list(itertools.islice(
                _iter_raw_sentences(config.phase1_dataset_dir, shuffle=False),
                self._DIAG_N,
            ))

    def _make_batch(self, sentences: list[str]) -> dict[str, torch.Tensor]:
        alien_strings = self.cipher.encode_batch(sentences)

        enc = self.native_tok(
            sentences, padding=True, truncation=True,
            max_length=self.config.phase1_max_source_len, return_tensors="pt",
        )
        dec = self.alien_tok(
            alien_strings, padding=True, truncation=True,
            max_length=self.config.phase1_max_target_len, return_tensors="pt",
        )
        labels = dec["input_ids"].clone()
        labels[labels == self.alien_tok.pad_token_id] = -100

        return {
            "native_ids": enc["input_ids"].to(self.device),
            "native_mask": enc["attention_mask"].to(self.device),
            "alien_labels": labels.to(self.device),
        }

    @torch.no_grad()
    def _run_diagnostics(self, step: int) -> None:
        self.model.eval()
        batch = self._make_batch(self._diag_sentences)

        # Teacher-forced accuracy
        logits = self.model.alien_logits_phase1(**batch)  # (B, T, V)
        pred_ids = logits.argmax(dim=-1)                  # (B, T)
        labels = batch["alien_labels"]
        valid = labels != -100
        pred_valid = pred_ids[valid]
        label_valid = labels[valid]
        tf_acc = (pred_valid == label_valid).float().mean().item()

        token_entropy = self._entropy(pred_valid)
        rep_rate = self._rep_rate(pred_ids, valid)

        # Free-run accuracy
        gen_ids = self.model.generate_phase1(
            batch["native_ids"],
            batch["native_mask"],
            alien_stop_id=self.alien_tok.eos_token_id,
            alien_pad_id=self.alien_tok.pad_token_id,
            max_length=self.config.phase1_max_target_len,
        )
        free_acc = self._free_acc(gen_ids, labels, self.alien_tok)

        logger.info(
            "phase1 diag  step=%d  tf_acc=%.3f  free_acc=%.3f  "
            "entropy=%.2f  rep_rate=%.3f",
            step, tf_acc, free_acc, token_entropy, rep_rate,
        )
        self.model.train()

    @staticmethod
    def _entropy(ids: torch.Tensor) -> float:
        counts = torch.bincount(ids, minlength=1).float()
        probs = counts / counts.sum()
        return -(probs * (probs + 1e-9).log()).sum().item()

    @staticmethod
    def _rep_rate(pred_ids: torch.Tensor, valid_mask: torch.Tensor) -> float:
        num = den = 0
        for b in range(pred_ids.size(0)):
            toks = pred_ids[b][valid_mask[b]]
            if toks.numel() > 1:
                num += (toks[1:] == toks[:-1]).sum().item()
                den += toks.numel() - 1
        return num / den if den > 0 else 0.0

    @staticmethod
    def _free_acc(
        gen_ids: torch.Tensor,
        labels: torch.Tensor,
        alien_tok: PreTrainedTokenizerFast,
    ) -> float:
        ignore = {alien_tok.eos_token_id, alien_tok.pad_token_id}
        accs = []
        for b in range(labels.size(0)):
            gen = [t for t in gen_ids[b].tolist() if t not in ignore]
            tgt = [t for t in labels[b].tolist() if t != -100 and t not in ignore]
            n = min(len(gen), len(tgt))
            if n:
                accs.append(sum(g == t for g, t in zip(gen[:n], tgt[:n])) / n)
        return sum(accs) / len(accs) if accs else 0.0

    def train(self, start_step: int = 0) -> None:
        cfg = self.config
        out_dir = Path(cfg.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        self.model.to(self.device).train()

        data = _iter_raw_sentences(cfg.phase1_dataset_dir, seed=cfg.seed)
        for _ in range(start_step * cfg.phase1_batch_size):
            next(data)

        step = start_step
        running_loss = 0.0

        while step < cfg.phase1_steps:
            batch = self._make_batch([next(data) for _ in range(cfg.phase1_batch_size)])

            self.opt.zero_grad()
            loss = self.model.forward_phase1(**batch)
            loss.backward()
            all_p1_params = self.model.backend.cipher_params() + self.model.backend.body_params()
            torch.nn.utils.clip_grad_norm_(all_p1_params, 1.0)
            self.opt.step()

            running_loss += loss.item()
            step += 1

            if step % cfg.phase1_log_every == 0:
                logger.info(
                    "phase1 step=%d  loss=%.4f",
                    step, running_loss / cfg.phase1_log_every,
                )
                running_loss = 0.0

            if cfg.phase1_diag_every > 0 and step % cfg.phase1_diag_every == 0:
                self._run_diagnostics(step)

            if step % cfg.phase1_checkpoint_every == 0:
                ckpt_path = str(out_dir / "phase1_checkpoint.pt")
                self._save_checkpoint(ckpt_path, step)
                logger.info("phase1 checkpoint -> %s  (step %d)", ckpt_path, step)

        self.model.save_phase1(str(out_dir / "phase1_final.pt"))
        logger.info("phase1 training complete")

    def _save_checkpoint(self, path: str, step: int) -> None:
        torch.save({
            "step": step,
            "model": self.model.phase1_state(),
            "optimizer": self.opt.state_dict(),
        }, path)

    def load_checkpoint(self, path: str) -> int:
        ckpt = torch.load(path, map_location="cpu", weights_only=True)
        self.model.load_phase1_state(ckpt["model"])
        self.opt.load_state_dict(ckpt["optimizer"])
        for state in self.opt.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(self.device)
        return ckpt["step"]


# ---------------------------------------------------------------------------
# Phase 2 — Embedding conditioning
# ---------------------------------------------------------------------------

class ConditioningTrainer:
    """Train PrefixProjector + LengthHead on (source_embedding → alien) pairs.

    Backend body and alien emb/head are frozen.  Only projector and length_head train.
    """

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
        self._freeze_cipher()
        self.opt = AdamW(self._trainable_params(), lr=config.phase2_lr)
        logger.info(
            "ConditioningTrainer  batch=%d  lr=%g  steps=%d  "
            "tgt_len=%d  len_weight=%g  device=%s  out=%s",
            config.phase2_batch_size, config.phase2_lr, config.phase2_steps,
            config.phase2_max_target_len, config.phase2_length_loss_weight,
            config.device, config.output_dir,
        )

    def _freeze_cipher(self) -> None:
        """Freeze all Phase 1 trained weights; only projector + length_head train."""
        self.model.backend.freeze_body()
        for p in self.model.backend._alien_emb.parameters():
            p.requires_grad_(False)
        for p in self.model.backend._alien_head.parameters():
            p.requires_grad_(False)

    def _trainable_params(self) -> list:
        return (
            list(self.model.projector.parameters())
            + list(self.model.length_head.parameters())
        )

    def _make_batch(
        self,
        embeddings: np.ndarray,
        texts: list[str],
        indices: list[int],
    ) -> dict[str, torch.Tensor]:
        alien_strings = self.cipher.encode_batch([texts[i] for i in indices])

        dec = self.alien_tok(
            alien_strings, padding=True, truncation=True,
            max_length=self.config.phase2_max_target_len, return_tensors="pt",
        )
        labels = dec["input_ids"].clone()
        labels[labels == self.alien_tok.pad_token_id] = -100

        emb = torch.tensor(
            embeddings[indices].astype(np.float32), dtype=torch.float32,
        )
        return {
            "source_embedding": emb.to(self.device),
            "alien_labels": labels.to(self.device),
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

        step = start_step
        running_lm = running_len = 0.0
        w_len = cfg.phase2_length_loss_weight

        while step < cfg.phase2_steps:
            indices = rng.sample(range(N), min(cfg.phase2_batch_size, N))
            batch = self._make_batch(embeddings, texts, indices)

            self.opt.zero_grad()
            lm_loss, len_loss = self.model.forward_phase2(**batch)
            loss = lm_loss + w_len * len_loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.projector.parameters(), 1.0)
            torch.nn.utils.clip_grad_norm_(self.model.length_head.parameters(), 1.0)
            self.opt.step()

            running_lm += lm_loss.item()
            running_len += len_loss.item()
            step += 1

            if step % cfg.phase2_log_every == 0:
                n = cfg.phase2_log_every
                logger.info(
                    "phase2 step=%d  lm=%.4f  len=%.4f",
                    step, running_lm / n, running_len / n,
                )
                running_lm = running_len = 0.0

            if step % cfg.phase2_checkpoint_every == 0:
                ckpt_path = str(out_dir / "phase2_checkpoint.pt")
                self._save_checkpoint(ckpt_path, step)
                logger.info("phase2 checkpoint -> %s  (step %d)", ckpt_path, step)

        self.model.save_phase2(str(out_dir / "phase2_final.pt"))
        logger.info("phase2 training complete")

    def _save_checkpoint(self, path: str, step: int) -> None:
        torch.save({
            "step": step,
            "model": self.model.phase2_state(),
            "optimizer": self.opt.state_dict(),
        }, path)

    def load_checkpoint(self, path: str) -> int:
        ckpt = torch.load(path, map_location="cpu", weights_only=True)
        self.model.load_phase2_state(ckpt["model"])
        self.opt.load_state_dict(ckpt["optimizer"])
        for state in self.opt.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(self.device)
        return ckpt["step"]

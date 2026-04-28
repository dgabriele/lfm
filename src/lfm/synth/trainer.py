"""Two-phase trainer for SynthLM.

Phase 1 — CipherTrainer:
    Reads English text from the DepTreeVAE HDF5 dataset (raw field),
    applies the word cipher, trains the mT5 decoder on (English -> alien) pairs.
    Encoder weights are frozen; only decoder layers + alien embed/lm_head train.

Phase 2 — ConditioningTrainer:
    Reads (embedding, passage) pairs from the embedding store.
    Applies cipher to passages to produce alien targets.
    Freezes the mT5 decoder; trains only the EmbeddingProjector + LengthHead.
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
import torch.nn.functional as F
from torch.optim import AdamW
from transformers import AutoTokenizer, PreTrainedTokenizerFast

from lfm.synth.cipher import WordCipher
from lfm.synth.config import SynthConfig
from lfm.synth.model import SynthLM
from lfm.synth.vocab import AlienVocab

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Thin data readers (no lfm imports)
# ---------------------------------------------------------------------------

def _iter_raw_sentences(dataset_path: str, shuffle: bool = True, seed: int = 42) -> Iterator[str]:
    """Yield raw English strings indefinitely.

    Accepts three source formats:
      * Directory containing ``samples.h5`` (DepTreeVAE HDF5 dataset)
      * ``.jsonl`` file with ``{"text": "..."}`` objects
      * ``.txt`` file with one sentence per line
    """
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
    passages_path = store_dir / "passages.jsonl"
    texts = [json.loads(line)["text"] for line in passages_path.read_text().splitlines() if line.strip()]
    assert len(texts) == len(embeddings), (
        f"embeddings ({len(embeddings)}) and passages ({len(texts)}) count mismatch"
    )
    return embeddings, texts


# ---------------------------------------------------------------------------
# Phase 1 — Cipher fine-tuning
# ---------------------------------------------------------------------------

class CipherTrainer:
    """Train the mT5 decoder to produce alien tokens from English input.

    Only decoder layers, alien embed_tokens, and lm_head are trainable.
    The mT5 encoder is frozen throughout.
    """

    _DIAG_N = 16  # fixed sentences used for diagnostics

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
        self.english_tok = AutoTokenizer.from_pretrained(config.base_model_name)
        self.device = torch.device(config.device)
        self._freeze_encoder()
        self.opt = AdamW(self._trainable_params(), lr=config.phase1_lr)
        self.mse_weight = config.phase1_hidden_mse_weight
        self.frozen_decoder = None
        if self.mse_weight > 0:
            frozen = model._frozen_decoder
            if frozen is None:
                raise ValueError("phase1_hidden_mse_weight > 0 but model._frozen_decoder is None")
            self.frozen_decoder = frozen.to(self.device)
        logger.info(
            "CipherTrainer  model=%s  batch=%d  lr=%g  steps=%d  "
            "src_len=%d  tgt_len=%d  hidden_mse_weight=%g  ar_weight=%g  "
            "device=%s  out=%s",
            config.base_model_name, config.phase1_batch_size, config.phase1_lr,
            config.phase1_steps, config.phase1_max_source_len,
            config.phase1_max_target_len, config.phase1_hidden_mse_weight,
            config.phase1_ar_weight, config.device, config.output_dir,
        )
        # Fixed diagnostic sentences — loaded once, never shuffled, same across runs.
        if config.phase1_diag_every > 0:
            self._diag_sentences = list(itertools.islice(
                _iter_raw_sentences(config.phase1_dataset_dir, shuffle=False),
                self._DIAG_N,
            ))

    def _freeze_encoder(self) -> None:
        for p in self.model.mt5.encoder.parameters():
            p.requires_grad_(False)

    def _trainable_params(self) -> list:
        return (
            list(self.model.mt5.decoder.parameters())
            + list(self.model.mt5.lm_head.parameters())
        )

    @torch.no_grad()
    def _run_diagnostics(self, step: int) -> None:
        """Log linguistic fidelity metrics on the fixed diagnostic batch."""
        self.model.eval()
        batch = self._make_batch(self._diag_sentences)

        # ---- teacher-forced metrics ----
        out = self.model.mt5(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            labels=batch["labels"],
        )
        pred_ids = out.logits.argmax(dim=-1)   # (B, T)
        label_ids = batch["labels"]             # (B, T), -100 = pad

        valid = label_ids != -100
        pred_valid = pred_ids[valid]
        label_valid = label_ids[valid]

        cipher_acc = (pred_valid == label_valid).float().mean().item()

        counts = torch.bincount(pred_valid, minlength=len(self.alien_tok)).float()
        probs = counts / counts.sum()
        token_entropy = -(probs * (probs + 1e-9).log()).sum().item()

        rep_num = rep_den = 0
        for b in range(pred_ids.size(0)):
            toks = pred_ids[b][valid[b]]
            if toks.numel() > 1:
                rep_num += (toks[1:] == toks[:-1]).sum().item()
                rep_den += toks.numel() - 1
        rep_rate = rep_num / rep_den if rep_den > 0 else 0.0

        # ---- free-run metrics ----
        gen_ids = self.model.mt5.generate(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            max_length=self.config.phase1_max_target_len,
            num_beams=1,
        )
        # align_acc: token-level accuracy on the shorter of gen vs target
        free_accs = []
        for b in range(len(self._diag_sentences)):
            gen = [t for t in gen_ids[b].tolist()
                   if t not in (self.alien_tok.bos_token_id,
                                self.alien_tok.eos_token_id,
                                self.alien_tok.pad_token_id)]
            tgt = [t for t in label_ids[b].tolist() if t != -100
                   and t not in (self.alien_tok.eos_token_id,
                                 self.alien_tok.pad_token_id)]
            n = min(len(gen), len(tgt))
            if n:
                free_accs.append(sum(g == t for g, t in zip(gen[:n], tgt[:n])) / n)
        free_cipher_acc = sum(free_accs) / len(free_accs) if free_accs else 0.0

        # mean generated length vs target length
        mean_gen_len = sum(
            len([t for t in gen_ids[b].tolist()
                 if t not in (self.alien_tok.bos_token_id,
                              self.alien_tok.eos_token_id,
                              self.alien_tok.pad_token_id)])
            for b in range(len(self._diag_sentences))
        ) / len(self._diag_sentences)
        mean_tgt_len = valid.float().sum(dim=-1).mean().item()

        logger.info(
            "phase1 diag  step=%d  "
            "tf_acc=%.3f  free_acc=%.3f  "
            "entropy=%.2f  rep_rate=%.3f  "
            "gen_len=%.1f  tgt_len=%.1f",
            step, cipher_acc, free_cipher_acc,
            token_entropy, rep_rate,
            mean_gen_len, mean_tgt_len,
        )
        self.model.train()

    def _ar_loss(self, sentences: list[str]) -> torch.Tensor:
        """CE loss with model's own generated tokens as decoder context.

        Runs generate() under no_grad to get the free-run sequence, then
        does a differentiable forward pass using that sequence as decoder
        input against the cipher targets.  Gradient flows through the
        forward pass only — not through the discrete generate() call.
        """
        batch = self._make_batch(sentences)
        T_label = batch["labels"].size(1)

        with torch.no_grad():
            gen_ids = self.model.mt5.generate(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                max_length=self.config.phase1_max_target_len,
                num_beams=1,
            )  # (B, T_gen) — starts with BOS

        # decoder_input_ids must be exactly T_label tokens so logits match labels shape.
        # gen_ids starts with BOS: [BOS, g1, g2, ...]; take first T_label tokens.
        T_gen = gen_ids.size(1)
        if T_gen < T_label:
            pad = torch.full(
                (gen_ids.size(0), T_label - T_gen),
                self.alien_tok.pad_token_id,
                device=self.device,
            )
            gen_ids = torch.cat([gen_ids, pad], dim=1)
        decoder_input = gen_ids[:, :T_label]

        return self.model.mt5(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            decoder_input_ids=decoder_input,
            labels=batch["labels"],
        ).loss

    def _hidden_mse_loss(
        self,
        batch: dict,
        enc_hidden: torch.Tensor,
        dec_hidden: torch.Tensor,
    ) -> torch.Tensor:
        """MSE between current and frozen pretrained decoder last hidden states.

        Rationale: Phase 2 relies on the pretrained mT5 decoder's structural prior —
        the EmbeddingProjector is trained to drive *that specific pretrained distribution*
        into linguistically structured output.  If Phase 1 overwrites the transformer
        layers with cipher-optimised weights, Phase 2 is conditioning a decoder that
        has drifted out of the pretrained regime, undermining the entire two-phase
        design.  This loss is a correctness condition, not a performance optimisation:
        it keeps the transformer body close to its pretrained state so that vocabulary
        adaptation (embed_tokens + lm_head) is the primary thing Phase 1 learns.

        Uses inputs_embeds from the current (alien) embed_tokens so both decoders
        receive identical continuous representations — bypassing vocabulary mismatch.
        """
        dec_inp = self.model.mt5._shift_right(batch["labels"])
        dec_embeds = self.model.mt5.decoder.embed_tokens(dec_inp)
        with torch.no_grad():
            frozen_out = self.frozen_decoder(
                inputs_embeds=dec_embeds.detach(),
                encoder_hidden_states=enc_hidden.detach(),
                encoder_attention_mask=batch["attention_mask"],
            )
        valid = batch["labels"] != -100
        return F.mse_loss(dec_hidden[valid], frozen_out.last_hidden_state[valid])

    def _make_batch(self, sentences: list[str]) -> dict[str, torch.Tensor]:
        alien = self.cipher.encode_batch(sentences)

        enc = self.english_tok(
            sentences, padding=True, truncation=True,
            max_length=self.config.phase1_max_source_len, return_tensors="pt",
        )
        dec = self.alien_tok(
            alien, padding=True, truncation=True,
            max_length=self.config.phase1_max_target_len, return_tensors="pt",
        )
        labels = dec["input_ids"].clone()
        labels[labels == self.alien_tok.pad_token_id] = -100

        return {
            "input_ids": enc["input_ids"].to(self.device),
            "attention_mask": enc["attention_mask"].to(self.device),
            "labels": labels.to(self.device),
        }

    def train(self, start_step: int = 0) -> None:
        cfg = self.config
        out_dir = Path(cfg.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        self.model.to(self.device).train()

        data = _iter_raw_sentences(cfg.phase1_dataset_dir, seed=cfg.seed)
        # Advance the data iterator to match start_step so we don't repeat data.
        B = cfg.phase1_batch_size
        for _ in range(start_step * B):
            next(data)

        step = start_step
        running_loss = running_ce = running_mse = 0.0

        while step < cfg.phase1_steps:
            batch_sentences = [next(data) for _ in range(B)]
            batch = self._make_batch(batch_sentences)

            self.opt.zero_grad()
            if self.mse_weight > 0:
                mt5_out = self.model.mt5(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    labels=batch["labels"],
                    output_hidden_states=True,
                )
                ce_loss = mt5_out.loss
                mse_loss = self._hidden_mse_loss(
                    batch,
                    mt5_out.encoder_last_hidden_state,
                    mt5_out.decoder_hidden_states[-1],
                )
                loss = ce_loss + self.mse_weight * mse_loss
                running_ce += ce_loss.item()
                running_mse += mse_loss.item()
            else:
                loss = self.model.forward_phase1(**batch)
            if (cfg.phase1_ar_weight > 0
                    and cfg.phase1_ar_every > 0
                    and step % cfg.phase1_ar_every == 0):
                ar_sents = batch_sentences[: cfg.phase1_ar_batch_size]
                ar_loss = self._ar_loss(ar_sents)
                loss = loss + cfg.phase1_ar_weight * ar_loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self._trainable_params(), 1.0)
            self.opt.step()

            running_loss += loss.item()
            step += 1

            if step % cfg.phase1_log_every == 0:
                n = cfg.phase1_log_every
                if self.mse_weight > 0:
                    logger.info(
                        "phase1 step=%d  loss=%.4f  ce=%.4f  hidden_mse=%.4f",
                        step, running_loss / n, running_ce / n, running_mse / n,
                    )
                else:
                    logger.info("phase1 step=%d  loss=%.4f", step, running_loss / n)
                running_loss = running_ce = running_mse = 0.0

            if cfg.phase1_diag_every > 0 and step % cfg.phase1_diag_every == 0:
                self._run_diagnostics(step)

            if step % cfg.phase1_checkpoint_every == 0:
                ckpt_path = str(out_dir / f"phase1_step{step}.pt")
                self._save_checkpoint(ckpt_path, step)
                logger.info("phase1 checkpoint -> %s", ckpt_path)

        self.model.save_phase1(str(out_dir / "phase1_final.pt"))
        logger.info("phase1 training complete")

    def _save_checkpoint(self, path: str, step: int) -> None:
        torch.save({
            "step": step,
            "model_lm_head": self.model.mt5.lm_head.state_dict(),
            "model_decoder_body": self.model.mt5.decoder.state_dict(),
            "optimizer": self.opt.state_dict(),
        }, path)

    def load_checkpoint(self, path: str) -> int:
        """Load trainer checkpoint; returns the saved step count."""
        ckpt = torch.load(path, map_location="cpu", weights_only=True)
        self.model.mt5.lm_head.load_state_dict(ckpt["model_lm_head"])
        self.model.mt5.decoder.load_state_dict(ckpt["model_decoder_body"])
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
    """Train EmbeddingProjector + LengthHead on (embedding -> alien) pairs.

    The mT5 decoder is fully frozen.  Only the projector and length head train.
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
        self._freeze_mt5()
        self.opt = AdamW(self._trainable_params(), lr=config.phase2_lr)
        logger.info(
            "ConditioningTrainer  batch=%d  lr=%g  steps=%d  "
            "tgt_len=%d  len_weight=%g  device=%s  out=%s",
            config.phase2_batch_size, config.phase2_lr, config.phase2_steps,
            config.phase2_max_target_len, config.phase2_length_loss_weight,
            config.device, config.output_dir,
        )

    def _freeze_mt5(self) -> None:
        for p in self.model.mt5.parameters():
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
        sents = [texts[i] for i in indices]
        alien = self.cipher.encode_batch(sents)

        dec = self.alien_tok(
            alien, padding=True, truncation=True,
            max_length=self.config.phase2_max_target_len, return_tensors="pt",
        )
        labels = dec["input_ids"].clone()
        labels[labels == self.alien_tok.pad_token_id] = -100

        emb = torch.tensor(
            embeddings[indices].astype(np.float32), dtype=torch.float32,
        )
        return {
            "source_embedding": emb.to(self.device),
            "labels": labels.to(self.device),
        }

    def train(self, start_step: int = 0) -> None:
        cfg = self.config
        out_dir = Path(cfg.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        self.model.to(self.device).train()

        embeddings, texts = _load_store(cfg.phase2_store_dir)
        N = len(texts)
        # Advance RNG to match start_step so resumed sampling is consistent.
        rng = random.Random(cfg.seed)
        for _ in range(start_step):
            rng.sample(range(N), min(cfg.phase2_batch_size, N))

        B = cfg.phase2_batch_size
        step = start_step
        running_lm = running_len = 0.0
        w_len = cfg.phase2_length_loss_weight

        while step < cfg.phase2_steps:
            indices = rng.sample(range(N), min(B, N))
            batch = self._make_batch(embeddings, texts, indices)

            self.opt.zero_grad()
            lm_loss, len_loss = self.model.forward_phase2(**batch)
            loss = lm_loss + w_len * len_loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.projector.parameters(), 1.0)
            torch.nn.utils.clip_grad_norm_(self.model.length_head.parameters(), 1.0)
            self.opt.step()

            running_lm  += lm_loss.item()
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
                ckpt_path = str(out_dir / f"phase2_step{step}.pt")
                self._save_checkpoint(ckpt_path, step)
                logger.info("phase2 checkpoint -> %s", ckpt_path)

        self.model.save_phase2(str(out_dir / "phase2_final.pt"))
        logger.info("phase2 training complete")

    def _save_checkpoint(self, path: str, step: int) -> None:
        torch.save({
            "step": step,
            "model_projector": self.model.projector.state_dict(),
            "model_length_head": self.model.length_head.state_dict(),
            "optimizer": self.opt.state_dict(),
        }, path)

    def load_checkpoint(self, path: str) -> int:
        """Load trainer checkpoint; returns the saved step count."""
        ckpt = torch.load(path, map_location="cpu", weights_only=True)
        self.model.projector.load_state_dict(ckpt["model_projector"])
        self.model.length_head.load_state_dict(ckpt["model_length_head"])
        self.opt.load_state_dict(ckpt["optimizer"])
        for state in self.opt.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(self.device)
        return ckpt["step"]

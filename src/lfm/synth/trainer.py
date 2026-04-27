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
from lfm.synth.vocab import AlienVocab

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Thin data readers (no lfm imports)
# ---------------------------------------------------------------------------

def _iter_raw_sentences(dataset_dir: str, shuffle: bool = True, seed: int = 42) -> Iterator[str]:
    """Yield raw English strings from DepTreeVAE HDF5 dataset indefinitely."""
    h5_path = Path(dataset_dir) / "samples.h5"
    rng = random.Random(seed)
    with h5py.File(h5_path, "r") as f:
        raws = [
            x.decode("utf-8") if isinstance(x, bytes) else x
            for x in f["samples"]["raw"][:]
        ]
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

    def _freeze_encoder(self) -> None:
        for p in self.model.mt5.encoder.parameters():
            p.requires_grad_(False)

    def _trainable_params(self) -> list:
        return (
            list(self.model.mt5.decoder.parameters())
            + list(self.model.mt5.lm_head.parameters())
        )

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

    def train(self) -> None:
        cfg = self.config
        out_dir = Path(cfg.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        self.model.to(self.device).train()

        data = _iter_raw_sentences(cfg.phase1_dataset_dir, seed=cfg.seed)
        B = cfg.phase1_batch_size
        step = 0
        running_loss = 0.0

        while step < cfg.phase1_steps:
            batch_sentences = [next(data) for _ in range(B)]
            batch = self._make_batch(batch_sentences)

            self.opt.zero_grad()
            loss = self.model.forward_phase1(**batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self._trainable_params(), 1.0)
            self.opt.step()

            running_loss += loss.item()
            step += 1

            if step % cfg.phase1_log_every == 0:
                avg = running_loss / cfg.phase1_log_every
                logger.info("phase1 step=%d  loss=%.4f", step, avg)
                running_loss = 0.0

            if step % cfg.phase1_checkpoint_every == 0:
                ckpt_path = str(out_dir / f"phase1_step{step}.pt")
                self.model.save_phase1(ckpt_path)
                logger.info("phase1 checkpoint -> %s", ckpt_path)

        self.model.save_phase1(str(out_dir / "phase1_final.pt"))
        logger.info("phase1 training complete")


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
        self._freeze_decoder()
        self.opt = AdamW(self._trainable_params(), lr=config.phase2_lr)

    def _freeze_decoder(self) -> None:
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
            max_length=self.config.phase1_max_target_len, return_tensors="pt",
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

    def train(self) -> None:
        cfg = self.config
        out_dir = Path(cfg.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        self.model.to(self.device).train()

        embeddings, texts = _load_store(cfg.phase2_store_dir)
        N = len(texts)
        rng = random.Random(cfg.seed)
        B = cfg.phase2_batch_size
        step = 0
        running_lm = running_len = 0.0
        w_len = cfg.phase2_length_loss_weight

        while step < cfg.phase2_steps:
            indices = rng.sample(range(N), min(B, N))
            batch = self._make_batch(embeddings, texts, indices)

            self.opt.zero_grad()
            lm_loss, len_loss = self.model.forward_phase2(**batch)
            loss = lm_loss + w_len * len_loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self._trainable_params(), 1.0)
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
                self.model.save_phase2(ckpt_path)
                logger.info("phase2 checkpoint -> %s", ckpt_path)

        self.model.save_phase2(str(out_dir / "phase2_final.pt"))
        logger.info("phase2 training complete")

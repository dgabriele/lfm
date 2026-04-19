#!/usr/bin/env python
"""Train the deterministic p2g seq2seq baseline.

Same data + training loop as train_p2g_vae.py; only the model differs.
Used to head-to-head compare VAE-bottleneck vs cross-attention seq2seq
on the same compute budget.
"""

from __future__ import annotations

import argparse
import logging
import random
from functools import partial
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

from lfm.p2g.data import P2GDataset, build_vocabs, collate
from lfm.p2g.seq2seq import P2GSeq2Seq, P2GSeq2SeqConfig

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger(__name__)

PSEUDOWORDS = [
    "bɹɪndʌlɪŋ",
    "snɔɹvɛnt",
    "flʌbɹæʃɪz",
    "kɹimɔɪstɝ",
    "fʌʃʌnʌbʌl",
    "ɔɹθʌɡɹɑfi",
    "kɑnʃʌsnʌs",
    "mʌʃin",
]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("config", type=Path)
    ap.add_argument("--resume", type=Path, default=None,
                    help="Resume from checkpoint (latest.pt)")
    args = ap.parse_args()
    cfg = yaml.safe_load(args.config.read_text())

    set_seed(cfg["seed"])
    out_dir = Path(cfg["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    ipa_vocab, sp_vocab = build_vocabs(cfg["train_h5"])
    ipa_vocab.to_json(out_dir / "ipa_vocab.json")
    sp_vocab.to_json(out_dir / "spelling_vocab.json")
    logger.info(f"  ipa_vocab={ipa_vocab.size}  spelling_vocab={sp_vocab.size}")

    train_ds = P2GDataset(
        cfg["train_h5"], ipa_vocab, sp_vocab,
        cfg["max_ipa_len"], cfg["max_spelling_len"],
    )
    val_ds = P2GDataset(
        cfg["val_h5"], ipa_vocab, sp_vocab,
        cfg["max_ipa_len"], cfg["max_spelling_len"],
    )
    collate_fn = partial(
        collate,
        max_ipa_len=cfg["max_ipa_len"],
        max_spelling_len=cfg["max_spelling_len"],
    )
    train_loader = DataLoader(
        train_ds, batch_size=cfg["batch_size"], shuffle=True,
        num_workers=2, collate_fn=collate_fn, drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=cfg["batch_size"], shuffle=False,
        num_workers=2, collate_fn=collate_fn,
    )
    logger.info(f"train={len(train_ds):,}  val={len(val_ds):,}")

    model_cfg = P2GSeq2SeqConfig(
        input_vocab_size=ipa_vocab.size,
        output_vocab_size=sp_vocab.size,
        d_model=cfg["decoder_dim"],
        encoder_layers=cfg["encoder_layers"],
        decoder_layers=cfg["decoder_layers"],
        nhead=cfg["decoder_heads"],
        max_ipa_len=cfg["max_ipa_len"],
        max_spelling_len=cfg["max_spelling_len"],
        dropout=cfg["dropout"],
    )
    device = torch.device(cfg["device"] if torch.cuda.is_available() else "cpu")
    model = P2GSeq2Seq(model_cfg).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    logger.info(f"p2g seq2seq: {n_params/1e6:.2f}M params  device={device}")

    optim = AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    total_steps = len(train_loader) * cfg["num_epochs"]
    sched = CosineAnnealingLR(optim, T_max=total_steps, eta_min=cfg["lr_min"])

    best_val = float("inf")
    best_word_acc = 0.0
    step = 0
    start_epoch = 0

    if args.resume:
        logger.info(f"Resuming from {args.resume}")
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state"])
        if "optimizer_state" in ckpt:
            optim.load_state_dict(ckpt["optimizer_state"])
        if "scheduler_state" in ckpt:
            sched.load_state_dict(ckpt["scheduler_state"])
        start_epoch = ckpt.get("epoch", 0) + 1
        step = ckpt.get("step", 0)
        best_val = ckpt.get("val_loss", float("inf"))
        best_word_acc = ckpt.get("val_word_acc", 0.0)
        logger.info(f"  resumed at epoch={start_epoch} step={step} best_val={best_val:.3f} best_word_acc={best_word_acc:.3f}")

    for epoch in range(start_epoch, cfg["num_epochs"]):
        model.train()
        running_loss = 0.0
        running_n = 0
        for batch in train_loader:
            batch_t = {
                k: v.to(device) for k, v in batch.items()
                if isinstance(v, torch.Tensor)
            }
            lw = batch_t.get("loss_weight") if cfg.get("use_loss_weights") else None
            out = model(
                batch_t["ipa_ids"], batch_t["spelling_ids"],
                batch_t["spelling_lens"], step=step, loss_weight=lw,
            )
            optim.zero_grad()
            out["loss"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optim.step()
            sched.step()
            step += 1
            running_loss += out["loss"].item()
            running_n += 1
            if step % 50 == 0:
                logger.info(
                    f"  step={step} loss={out['loss'].item():.4f} "
                    f"char_acc={out['char_acc'].item():.3f} "
                    f"word_acc={out['word_acc'].item():.3f} "
                    f"lr={sched.get_last_lr()[0]:.6f}"
                )

        # Validation
        model.eval()
        val_loss = 0.0
        val_acc = 0.0
        val_word_acc = 0.0
        nb = 0
        nw = 0
        with torch.no_grad():
            for batch in val_loader:
                batch_t = {
                    k: v.to(device) for k, v in batch.items()
                    if isinstance(v, torch.Tensor)
                }
                out = model(
                    batch_t["ipa_ids"], batch_t["spelling_ids"],
                    batch_t["spelling_lens"], step=step,
                )
                val_loss += out["loss"].item()
                val_acc += out["char_acc"].item()
                nb += 1
                preds = model.generate(batch_t["ipa_ids"])
                for pred_ids, tgt in zip(preds, batch["spelling_text"]):
                    pred_text = sp_vocab.decode(pred_ids)
                    if pred_text == tgt:
                        val_word_acc += 1
                    nw += 1
        val_loss /= nb
        val_acc /= nb
        val_word_acc /= max(1, nw)
        logger.info(
            f"[val] epoch={epoch} loss={val_loss:.3f} "
            f"char_acc={val_acc:.3f} word_acc={val_word_acc:.3f}"
        )

        if (epoch + 1) % cfg["sample_every_epochs"] == 0:
            with torch.no_grad():
                logger.info("  pseudowords:")
                for ipa in PSEUDOWORDS:
                    enc = ipa_vocab.encode(ipa)[: cfg["max_ipa_len"]]
                    padded = torch.zeros(
                        1, cfg["max_ipa_len"], dtype=torch.long, device=device,
                    )
                    padded[0, : len(enc)] = torch.tensor(enc, device=device)
                    pred = model.generate(padded)[0]
                    pred_text = sp_vocab.decode(pred)
                    logger.info(f"    {ipa!r:>22} → {pred_text!r}")

        ckpt = {
            "model_state": model.state_dict(),
            "optimizer_state": optim.state_dict(),
            "scheduler_state": sched.state_dict(),
            "cfg": cfg,
            "ipa_vocab": ipa_vocab.chars,
            "sp_vocab": sp_vocab.chars,
            "epoch": epoch,
            "step": step,
            "val_loss": val_loss,
            "val_word_acc": val_word_acc,
        }
        torch.save(ckpt, out_dir / "latest.pt")
        if val_loss < best_val:
            best_val = val_loss
            torch.save(ckpt, out_dir / "best.pt")
        if val_word_acc > best_word_acc:
            best_word_acc = val_word_acc
            torch.save(ckpt, out_dir / "best_word_acc.pt")
            logger.info(f"  ✓ new best word_acc ({val_word_acc:.3f})")
            logger.info(f"  ✓ new best (val_loss={val_loss:.3f})")

    logger.info(f"done. best val_loss={best_val:.3f}")


if __name__ == "__main__":
    main()

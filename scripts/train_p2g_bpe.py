#!/usr/bin/env python
"""Train BPE-output p2g seq2seq.

Reuses P2GSeq2Seq architecture; only the output vocab differs (Qwen-BPE
sub-words instead of chars).  Same data scale + budget as char baseline
for fair comparison.
"""

from __future__ import annotations

import argparse
import json
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

from lfm.p2g.bpe_data import (
    P2GBPEDataset,
    build_ipa_vocab_from_bpe_h5,
    collate_bpe,
)
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


def render_bpe_ids(local_ids: list[int], id_to_token_str: list[str]) -> str:
    """Concat BPE token strings into a spelling (Qwen 'Ġ'-prefix → space)."""
    parts = []
    for i in local_ids:
        if i < 3:           # specials
            continue
        s = id_to_token_str[i]
        # Qwen BPE leading 'Ġ' = space; we tokenized words bare so this
        # generally won't appear, but handle it for safety.
        if s.startswith("Ġ"):
            s = " " + s[1:]
        parts.append(s)
    return "".join(parts).strip()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("config", type=Path)
    ap.add_argument("--resume", type=Path, default=None)
    args = ap.parse_args()
    cfg = yaml.safe_load(args.config.read_text())

    set_seed(cfg["seed"])
    out_dir = Path(cfg["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    bpe_vocab_meta = json.loads(
        Path(cfg["bpe_vocab_path"]).read_text(),
    )
    id_to_token_str = bpe_vocab_meta["tokens"]
    output_vocab_size = bpe_vocab_meta["vocab_size"]
    logger.info(f"BPE output vocab: {output_vocab_size:,}")

    ipa_vocab = build_ipa_vocab_from_bpe_h5(cfg["train_h5"])
    ipa_vocab.to_json(out_dir / "ipa_vocab.json")
    logger.info(f"IPA char vocab: {ipa_vocab.size}")

    train_ds = P2GBPEDataset(
        cfg["train_h5"], ipa_vocab,
        cfg["max_ipa_len"], cfg["max_bpe_len"],
    )
    val_ds = P2GBPEDataset(
        cfg["val_h5"], ipa_vocab,
        cfg["max_ipa_len"], cfg["max_bpe_len"],
    )
    collate_fn = partial(
        collate_bpe,
        max_ipa_len=cfg["max_ipa_len"],
        max_bpe_len=cfg["max_bpe_len"],
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
        output_vocab_size=output_vocab_size,
        d_model=cfg["d_model"],
        encoder_layers=cfg["encoder_layers"],
        decoder_layers=cfg["decoder_layers"],
        nhead=cfg["nhead"],
        max_ipa_len=cfg["max_ipa_len"],
        max_spelling_len=cfg["max_bpe_len"],
        dropout=cfg["dropout"],
        label_smoothing=cfg.get("label_smoothing", 0.0),
    )
    device = torch.device(cfg["device"] if torch.cuda.is_available() else "cpu")
    model = P2GSeq2Seq(model_cfg).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    logger.info(f"BPE seq2seq: {n_params/1e6:.2f}M params  device={device}")

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
        start_epoch = ckpt.get("epoch", 0) + 1
        step = ckpt.get("step", 0)
        best_val = ckpt.get("val_loss", float("inf"))
        best_word_acc = ckpt.get("val_word_acc", 0.0)

        # Warm restart: fresh cosine over remaining epochs at reduced peak LR.
        # Scale decreases with each restart to progressively fine-tune.
        remaining_epochs = cfg["num_epochs"] - start_epoch
        remaining_steps = len(train_loader) * remaining_epochs
        # Scale by how far into training we are (earlier = larger restart)
        progress = start_epoch / cfg["num_epochs"]
        restart_lr = cfg["lr"] * max(0.05, 0.25 * (1.0 - progress))
        for pg in optim.param_groups:
            pg["lr"] = restart_lr
        sched = CosineAnnealingLR(optim, T_max=remaining_steps, eta_min=cfg["lr_min"])
        logger.info(
            f"  resumed at epoch={start_epoch} step={step} "
            f"best_val={best_val:.3f} best_word_acc={best_word_acc:.3f}"
        )
        logger.info(
            f"  warm restart: lr={restart_lr:.6f} → {cfg['lr_min']:.6f} "
            f"over {remaining_epochs} epochs ({remaining_steps} steps)"
        )

    for epoch in range(start_epoch, cfg["num_epochs"]):
        model.train()
        for batch in train_loader:
            batch_t = {
                k: v.to(device) for k, v in batch.items()
                if isinstance(v, torch.Tensor)
            }
            out = model(
                batch_t["ipa_ids"], batch_t["bpe_ids"],
                batch_t["bpe_lens"], step=step,
            )
            optim.zero_grad()
            out["loss"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optim.step()
            sched.step()
            step += 1
            if step % 50 == 0:
                logger.info(
                    f"  step={step} loss={out['loss'].item():.4f} "
                    f"char_acc={out['char_acc'].item():.3f} "
                    f"word_acc={out.get('word_acc', torch.tensor(0)).item():.3f} "
                    f"lr={sched.get_last_lr()[0]:.6f}"
                )
            if step % 5000 == 0:
                mid_ckpt = {
                    "model_state": model.state_dict(),
                    "optimizer_state": optim.state_dict(),
                    "scheduler_state": sched.state_dict(),
                    "cfg": cfg,
                    "ipa_vocab": ipa_vocab.chars,
                    "epoch": epoch,
                    "step": step,
                    "val_loss": best_val,
                    "val_word_acc": best_word_acc,
                }
                torch.save(mid_ckpt, out_dir / "latest.pt")
                logger.info(f"  mid-epoch checkpoint at step {step}")

        # Validation
        model.eval()
        val_loss = 0.0
        val_tok_acc = 0.0
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
                    batch_t["ipa_ids"], batch_t["bpe_ids"],
                    batch_t["bpe_lens"], step=step,
                )
                val_loss += out["loss"].item()
                val_tok_acc += out["char_acc"].item()
                nb += 1
                preds = model.generate(batch_t["ipa_ids"])
                for pred_ids, tgt in zip(preds, batch["spelling_text"]):
                    pred_text = render_bpe_ids(pred_ids, id_to_token_str)
                    if pred_text == tgt:
                        val_word_acc += 1
                    nw += 1
        val_loss /= nb
        val_tok_acc /= nb
        val_word_acc /= max(1, nw)
        logger.info(
            f"[val] epoch={epoch} loss={val_loss:.3f} "
            f"tok_acc={val_tok_acc:.3f} word_acc={val_word_acc:.3f}"
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
                    pred_text = render_bpe_ids(pred, id_to_token_str)
                    pred_toks = [id_to_token_str[i] for i in pred if i >= 3]
                    logger.info(f"    {ipa!r:>22} → {pred_text!r:<22} {pred_toks}")

        ckpt = {
            "model_state": model.state_dict(),
            "optimizer_state": optim.state_dict(),
            "scheduler_state": sched.state_dict(),
            "cfg": cfg,
            "ipa_vocab": ipa_vocab.chars,
            "epoch": epoch,
            "step": step,
            "val_loss": val_loss,
            "val_word_acc": val_word_acc,
        }
        torch.save(ckpt, out_dir / "latest.pt")
        if val_loss < best_val:
            best_val = val_loss
            torch.save(ckpt, out_dir / "best.pt")
            logger.info(f"  ✓ new best (val_loss={val_loss:.3f})")

    logger.info(f"done. best val_loss={best_val:.3f}")


if __name__ == "__main__":
    main()

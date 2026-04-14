#!/usr/bin/env python
"""Train the v10 p2g VAE.

Usage:
    poetry run python scripts/train_p2g_vae.py configs/p2g_v10.yaml
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
from lfm.p2g.model import P2GConfig, P2GVAE

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger(__name__)


# Pseudowords to watch each epoch: plausible English phonotactics that
# are NOT real words.  If the VAE is interpolating cleanly, these should
# decode to coherent real-looking spellings (not gibberish).
PSEUDOWORDS = [
    "bɹɪndʌlɪŋ",
    "snɔɹvɛnt",
    "flʌbɹæʃɪz",
    "kɹimɔɪstɝ",
    "fʌʃʌnʌbʌl",
    "ɔɹθʌɡɹɑfi",  # near-miss for 'orthography'
    "kɑnʃʌsnʌs",  # near-miss for 'consciousness'
    "mʌʃin",      # near-miss for 'machine'
]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_cfg(path: Path) -> dict:
    return yaml.safe_load(path.read_text())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("config", type=Path)
    args = ap.parse_args()
    cfg = load_cfg(args.config)

    set_seed(cfg["seed"])
    out_dir = Path(cfg["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Vocabs ──
    logger.info("building vocabs from train split")
    ipa_vocab, sp_vocab = build_vocabs(cfg["train_h5"])
    ipa_vocab.to_json(out_dir / "ipa_vocab.json")
    sp_vocab.to_json(out_dir / "spelling_vocab.json")
    logger.info(f"  ipa_vocab={ipa_vocab.size}  spelling_vocab={sp_vocab.size}")

    # ── Data ──
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

    # ── Model ──
    model_cfg = P2GConfig(
        input_vocab_size=ipa_vocab.size,
        output_vocab_size=sp_vocab.size,
        latent_dim=cfg["latent_dim"],
        encoder_dim=cfg["encoder_dim"],
        encoder_layers=cfg["encoder_layers"],
        encoder_heads=cfg["encoder_heads"],
        decoder_dim=cfg["decoder_dim"],
        decoder_layers=cfg["decoder_layers"],
        decoder_heads=cfg["decoder_heads"],
        max_ipa_len=cfg["max_ipa_len"],
        max_spelling_len=cfg["max_spelling_len"],
        dropout=cfg["dropout"],
        kl_weight_max=cfg["kl_weight_max"],
        kl_warmup_steps=cfg["kl_warmup_steps"],
        kl_free_bits=cfg["kl_free_bits"],
        length_weight=cfg["length_weight"],
    )
    device = torch.device(cfg["device"] if torch.cuda.is_available() else "cpu")
    model = P2GVAE(model_cfg).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    logger.info(f"p2g VAE: {n_params/1e6:.2f}M params  device={device}")

    optim = AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    total_steps = len(train_loader) * cfg["num_epochs"]
    sched = CosineAnnealingLR(optim, T_max=total_steps, eta_min=cfg["lr_min"])

    best_val = float("inf")
    step = 0
    for epoch in range(cfg["num_epochs"]):
        model.train()
        running = {"loss": 0.0, "recon": 0.0, "kl": 0.0, "len": 0.0, "acc": 0.0}
        running_n = 0
        for batch in train_loader:
            batch_t = {
                k: v.to(device) for k, v in batch.items()
                if isinstance(v, torch.Tensor)
            }
            out = model(
                batch_t["ipa_ids"], batch_t["spelling_ids"],
                batch_t["spelling_lens"], step=step,
            )
            optim.zero_grad()
            out["loss"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optim.step()
            sched.step()
            step += 1
            running["loss"] += out["loss"].item()
            running["recon"] += out["recon"].item()
            running["kl"] += out["kl"].item()
            running["len"] += out["length_loss"].item()
            running["acc"] += out["char_acc"].item()
            running_n += 1
            if step % cfg["log_every"] == 0:
                n = running_n
                logger.info(
                    f"epoch={epoch} step={step} "
                    f"loss={running['loss']/n:.3f} "
                    f"recon={running['recon']/n:.3f} "
                    f"kl={running['kl']/n:.3f} (w={out['kl_weight'].item():.2f}) "
                    f"len_loss={running['len']/n:.3f} "
                    f"char_acc={running['acc']/n:.3f} "
                    f"lr={sched.get_last_lr()[0]:.2e}"
                )
                running = {k: 0.0 for k in running}
                running_n = 0

        # ── Validate ──
        model.eval()
        val_loss = 0.0
        val_acc = 0.0
        val_word_acc = 0.0
        n_batches = 0
        n_words = 0
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
                n_batches += 1
                # Exact-word recovery rate
                preds = model.generate(batch_t["ipa_ids"], sample=False)
                for pred_ids, tgt in zip(preds, batch["spelling_text"]):
                    pred_text = sp_vocab.decode(pred_ids)
                    if pred_text == tgt:
                        val_word_acc += 1
                    n_words += 1
        val_loss /= n_batches
        val_acc /= n_batches
        val_word_acc /= max(1, n_words)
        logger.info(
            f"[val] epoch={epoch} loss={val_loss:.3f} "
            f"char_acc={val_acc:.3f} word_acc={val_word_acc:.3f}"
        )

        # ── Qualitative samples ──
        if (epoch + 1) % cfg["sample_every_epochs"] == 0:
            model.eval()
            with torch.no_grad():
                # A few real val words
                real_samples = [
                    (b["ipa_text"][i], b["spelling_text"][i])
                    for b in [next(iter(val_loader))]
                    for i in range(min(5, len(b["ipa_text"])))
                ]
                logger.info("  real words:")
                for ipa, tgt in real_samples:
                    ids = torch.tensor(
                        [ipa_vocab.encode(ipa)[: cfg["max_ipa_len"]]],
                        dtype=torch.long, device=device,
                    )
                    # Pad
                    padded = torch.zeros(
                        1, cfg["max_ipa_len"], dtype=torch.long, device=device,
                    )
                    padded[0, : ids.size(1)] = ids[0]
                    pred = model.generate(padded, sample=False)[0]
                    pred_text = sp_vocab.decode(pred)
                    tag = "✓" if pred_text == tgt else "✗"
                    logger.info(f"    {tag} {ipa!r:>22} → {pred_text!r:<18} (target={tgt!r})")

                logger.info("  pseudowords (interpolation check):")
                for ipa in PSEUDOWORDS:
                    enc = ipa_vocab.encode(ipa)[: cfg["max_ipa_len"]]
                    padded = torch.zeros(
                        1, cfg["max_ipa_len"], dtype=torch.long, device=device,
                    )
                    padded[0, : len(enc)] = torch.tensor(enc, device=device)
                    pred = model.generate(padded, sample=False)[0]
                    pred_text = sp_vocab.decode(pred)
                    logger.info(f"    {ipa!r:>22} → {pred_text!r}")

        # ── Checkpoint ──
        ckpt = {
            "model_state": model.state_dict(),
            "cfg": cfg,
            "ipa_vocab": ipa_vocab.chars,
            "sp_vocab": sp_vocab.chars,
            "epoch": epoch,
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

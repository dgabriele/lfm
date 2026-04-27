"""Statistically significant Phase 1 evaluation.

Runs free-generation (autoregressive, no teacher forcing) against a large
sample and reports:
  - free_cipher_acc     : token-level accuracy vs ground-truth cipher
  - exact_match_rate    : fraction of sentences perfectly ciphered
  - vocab_coverage      : unique tokens / vocab_size
  - token_entropy       : Shannon entropy of generated token distribution
  - rep_rate            : consecutive identical token pair fraction
  - mean/std gen_len    : generated sequence length stats
  - 10 side-by-side examples (source / expected / generated)

Usage:
    poetry run python scripts/synth_phase1_eval.py \\
        --checkpoint data/synth_local/phase1_step32000.pt \\
        --config configs/synth_local_extended.yaml \\
        --n 200
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch
from transformers import AutoTokenizer, PreTrainedTokenizerFast

import yaml
from lfm.synth.cipher import WordCipher
from lfm.synth.config import SynthConfig
from lfm.synth.model import SynthLM
from lfm.synth.vocab import AlienVocab


def load_sentences(dataset_path: str, n: int, seed: int = 0) -> list[str]:
    path = Path(dataset_path)
    if path.suffix == ".jsonl":
        lines = [json.loads(l)["text"] for l in path.read_text().splitlines() if l.strip()]
    else:
        lines = [l.strip() for l in path.read_text().splitlines() if l.strip()]
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(lines), size=min(n, len(lines)), replace=False)
    return [lines[i] for i in sorted(idx)]


def token_entropy(ids: list[int], vocab_size: int) -> float:
    counts = np.bincount(ids, minlength=vocab_size).astype(float)
    probs = counts / counts.sum()
    probs = probs[probs > 0]
    return float(-np.sum(probs * np.log(probs)))


def rep_rate(ids: list[int]) -> float:
    if len(ids) < 2:
        return 0.0
    pairs = sum(a == b for a, b in zip(ids, ids[1:]))
    return pairs / (len(ids) - 1)


def align_acc(pred: list[int], target: list[int]) -> float:
    """Token-level accuracy on the shorter sequence."""
    n = min(len(pred), len(target))
    if n == 0:
        return 0.0
    return sum(p == t for p, t in zip(pred[:n], target[:n])) / n


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--n", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", default=None, help="Override config device (e.g. cpu)")
    args = parser.parse_args()

    cfg = SynthConfig(**yaml.safe_load(Path(args.config).read_text()))
    out_dir = Path(cfg.output_dir)
    device = torch.device(args.device if args.device else cfg.device)

    print(f"Loading model from {args.checkpoint} ...")
    vocab = AlienVocab.load(out_dir)
    alien_tok: PreTrainedTokenizerFast = PreTrainedTokenizerFast.from_pretrained(
        str(out_dir / "alien_tokenizer")
    )
    english_tok = AutoTokenizer.from_pretrained(cfg.base_model_name)
    model = SynthLM(cfg, alien_vocab_size=len(alien_tok))
    model.load_phase1(args.checkpoint)
    model.eval().to(device)
    cipher = WordCipher(vocab)

    print(f"Loading {args.n} sentences from {cfg.phase1_dataset_dir} ...")
    sentences = load_sentences(cfg.phase1_dataset_dir, args.n)

    all_pred_ids: list[list[int]] = []
    all_target_ids: list[list[int]] = []

    print(f"Generating (batch={args.batch_size}) ...")
    for i in range(0, len(sentences), args.batch_size):
        batch_sents = sentences[i : i + args.batch_size]
        cipher_texts = cipher.encode_batch(batch_sents)

        enc = english_tok(
            batch_sents,
            padding=True,
            truncation=True,
            max_length=cfg.phase1_max_source_len,
            return_tensors="pt",
        )
        tgt = alien_tok(
            cipher_texts,
            padding=True,
            truncation=True,
            max_length=cfg.phase1_max_target_len,
            return_tensors="pt",
        )

        with torch.no_grad():
            gen = model.mt5.generate(
                input_ids=enc["input_ids"].to(device),
                attention_mask=enc["attention_mask"].to(device),
                max_length=cfg.phase1_max_target_len,
                num_beams=1,
            )

        for b in range(len(batch_sents)):
            pred = gen[b].tolist()
            # strip BOS/EOS/PAD
            pred = [t for t in pred if t not in (alien_tok.bos_token_id,
                                                  alien_tok.eos_token_id,
                                                  alien_tok.pad_token_id)]
            tgt_row = tgt["input_ids"][b].tolist()
            tgt_row = [t for t in tgt_row if t not in (alien_tok.bos_token_id,
                                                        alien_tok.eos_token_id,
                                                        alien_tok.pad_token_id)]
            all_pred_ids.append(pred)
            all_target_ids.append(tgt_row)

        if (i // args.batch_size) % 2 == 0:
            print(f"  {min(i + args.batch_size, len(sentences))}/{len(sentences)}")

    # ---- aggregate metrics ----
    accs = [align_acc(p, t) for p, t in zip(all_pred_ids, all_target_ids)]
    exact = [p == t for p, t in zip(all_pred_ids, all_target_ids)]
    gen_lens = [len(p) for p in all_pred_ids]
    tgt_lens = [len(t) for t in all_target_ids]

    flat_pred = [tok for seq in all_pred_ids for tok in seq]
    flat_tgt  = [tok for seq in all_target_ids  for tok in seq]

    rr = rep_rate(flat_pred)
    ent = token_entropy(flat_pred, len(alien_tok))
    vcov = len(set(flat_pred)) / len(alien_tok)

    # ---- print report ----
    n = len(sentences)
    print(f"\n{'='*60}")
    print(f"Phase 1 Free-Generation Eval  (n={n}, ckpt={Path(args.checkpoint).name})")
    print(f"{'='*60}")
    print(f"  free_cipher_acc   : {np.mean(accs):.3f}  ±{np.std(accs):.3f}")
    print(f"  exact_match_rate  : {np.mean(exact):.3f}  ({sum(exact)}/{n})")
    print(f"  vocab_coverage    : {vcov:.3f}  ({len(set(flat_pred))}/{len(alien_tok)} tokens seen)")
    print(f"  token_entropy     : {ent:.2f}  (max={math.log(len(alien_tok)):.2f})")
    print(f"  rep_rate          : {rr:.4f}")
    print(f"  mean gen_len      : {np.mean(gen_lens):.1f}  ±{np.std(gen_lens):.1f}")
    print(f"  mean tgt_len      : {np.mean(tgt_lens):.1f}  ±{np.std(tgt_lens):.1f}")
    print(f"  len_delta (gen-tgt): {np.mean(np.array(gen_lens)-np.array(tgt_lens)):.1f}")

    # ---- per-decile cipher_acc ----
    deciles = np.percentile(accs, [10, 25, 50, 75, 90])
    print(f"\n  cipher_acc percentiles:")
    print(f"    p10={deciles[0]:.3f}  p25={deciles[1]:.3f}  p50={deciles[2]:.3f}  p75={deciles[3]:.3f}  p90={deciles[4]:.3f}")

    # ---- 10 examples ----
    print(f"\n{'='*60}")
    print("Examples (src / expected / generated)")
    print(f"{'='*60}")
    for i in range(min(10, n)):
        src = sentences[i]
        exp = alien_tok.decode(all_target_ids[i], skip_special_tokens=True)
        gen_str = alien_tok.decode(all_pred_ids[i], skip_special_tokens=True)
        acc_i = accs[i]
        print(f"\n[{i}] acc={acc_i:.2f}")
        print(f"  SRC: {src[:90]}")
        print(f"  EXP: {exp[:90]}")
        print(f"  GEN: {gen_str[:90]}")


if __name__ == "__main__":
    main()

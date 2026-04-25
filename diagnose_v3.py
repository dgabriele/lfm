"""Root-cause diagnostic for v3.3 reconstruction failures.

Stratifies a held-out sample by token rarity, runs encode→decode with
instrumentation, then quantifies:

  - cycling rate (any token appearing ≥3 times in any 5-token window)
  - length error (decoded - GT)
  - length-head prediction error
  - longest verbatim recovery
  - per-step next-token entropy and argmax confidence
  - z-statistics (norm, logvar) at the encoder output

Aggregates by rarity bin and reports correlations between rarity, cycling,
length error, and verbatim recovery. The output identifies which failure
mode dominates and what the failure correlates with — guiding architectural
recommendations rather than guessing at config tweaks.

Usage (on vast or local with the model present):
    python diagnose_v3.py [--ckpt PATH] [--n SAMPLES]

Output: structured tables + JSON dump at /tmp/diagnose_v3_results.json
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import sentencepiece as spm
import torch
import torch.nn.functional as F
import yaml

sys.path.insert(0, "/workspace/lfm/src")
sys.path.insert(0, "/home/daniel/projects/lfm/src")
from lfm.generator.dep_tree_vae.config import (
    DepTreeVAEConfig,
    NUM_DEP_RELATIONS,
)
from lfm.generator.dep_tree_vae.model import DepTreeVAE
from lfm.generator.dep_tree_vae.trainer import _greedy_decode
from lfm.translator.romanize import respell


def detect_cycling(words: list[str], window: int = 5, min_count: int = 3) -> int:
    """Number of windows where any single token appears ≥min_count times."""
    cycles = 0
    for i in range(max(0, len(words) - window + 1)):
        w = Counter(words[i : i + window])
        if any(c >= min_count for c in w.values()):
            cycles += 1
    return cycles


def adjacent_repeats(words: list[str]) -> int:
    return sum(1 for i in range(1, len(words)) if words[i] == words[i - 1])


def longest_verbatim_run(rec: list[str], gt: list[str]) -> int:
    """Longest contiguous span of rec that appears verbatim in gt."""
    if not rec or not gt:
        return 0
    gt_str = " ".join(gt)
    best = 0
    for i in range(len(rec)):
        for j in range(i + 1, min(len(rec) + 1, i + len(gt) + 1)):
            run = " ".join(rec[i:j])
            if run in gt_str:
                best = max(best, j - i)
            else:
                break  # extend only if shorter span matched
    return best


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default=None, help="Checkpoint path (default: vast best.pt)")
    p.add_argument("--config", default=None)
    p.add_argument("--n", type=int, default=1024)
    p.add_argument("--bins", type=int, default=5)
    p.add_argument("--out", default="/tmp/diagnose_v3_results.json")
    args = p.parse_args()

    # ---- defaults (vast) ----
    if Path("/workspace/lfm").exists():
        ckpt_default = "/workspace/lfm/data/models/dep_tree_vae_v1/best.pt"
        cfg_default = "/workspace/lfm/configs/dep_tree_vae_vast.yaml"
        cache_dir = Path("/workspace/lfm/data/datasets/english-dep-trees-v16/cache_depth4")
        spm_path_default = "/workspace/lfm/data/models/v15b_ipa/spm.model"
    else:
        ckpt_default = "/home/daniel/projects/lfm/data/models/dep_tree_vae_v1/best.pt"
        cfg_default = "/home/daniel/projects/lfm/configs/dep_tree_vae_vast.yaml"
        cache_dir = Path("/home/daniel/projects/lfm/data/datasets/english-dep-trees-v16/cache_depth4")
        spm_path_default = "/home/daniel/projects/lfm/data/models/v15b_ipa/spm.model"

    ckpt_path = args.ckpt or ckpt_default
    cfg_path = args.config or cfg_default

    # ---- model ----
    cfg_dict = yaml.safe_load(open(cfg_path))
    cfg_dict.pop("model_type", None)
    cfg_dict["spm_model_path"] = spm_path_default
    cfg_dict["dataset_path"] = str(cache_dir.parent)
    cfg_dict["output_dir"] = "/tmp/diag_dummy"

    # Peek at checkpoint to disable heads not in state dict
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    state = ckpt["model_state"]
    if not any(k.startswith("length_head") for k in state):
        cfg_dict["length_pred_weight"] = 0.0
        cfg_dict["use_predicted_length_at_decode"] = False
    if not any(k.startswith("tokens_per_role_head") for k in state):
        cfg_dict["tokens_per_role_weight"] = 0.0

    cfg = DepTreeVAEConfig(**cfg_dict)
    sp = spm.SentencePieceProcessor(model_file=cfg.spm_model_path)
    spm_size = sp.get_piece_size()
    vocab_size = 8050

    model = DepTreeVAE(cfg, vocab_size).to(device)
    model.load_state_dict(state, strict=False)
    model.eval()
    step = ckpt.get("global_step", "?")
    print(f"Loaded {ckpt_path} @ step {step} (val_best={ckpt.get('best_val_loss', '?')})")

    # ---- token frequencies (sample-based) ----
    print("Computing token frequencies on cache sample...")
    interleaved = np.load(cache_dir / "interleaved.npy", mmap_mode="r")
    index = np.load(cache_dir / "index.npy")

    # Sample 500K positions to estimate frequencies cheaply
    rng = np.random.default_rng(0)
    sample_pos = rng.choice(len(interleaved), size=500_000, replace=False)
    freq: Counter[int] = Counter()
    for pos in sample_pos:
        t = int(interleaved[pos])
        if 0 < t < spm_size:
            freq[t] += 1
    total_freq = sum(freq.values())
    UNK_LOG_FREQ = float(np.log(0.5 / total_freq))  # smoothed for unseen tokens
    log_freq = {tid: float(np.log(c / total_freq)) for tid, c in freq.items()}

    def sentence_rarity(seq: np.ndarray) -> float:
        contents = [int(t) for t in seq.tolist() if 0 < t < spm_size]
        if not contents:
            return -np.inf
        lf = [log_freq.get(t, UNK_LOG_FREQ) for t in contents]
        return float(np.mean(lf))

    # ---- sample N sentences ----
    print(f"Selecting {args.n} samples...")
    candidate = rng.permutation(len(index))
    chosen: list[dict] = []
    for i in candidate:
        start = int(index[i, 0])
        end = start + int(index[i, 1])
        seq = np.asarray(interleaved[start:end])
        if len(seq) > cfg.max_seq_len:
            continue
        content_ids = [int(t) for t in seq.tolist() if 0 < t < spm_size]
        if len(content_ids) < 4:
            continue
        chosen.append({
            "seq": seq,
            "content_ids": content_ids,
            "gt_len": len(content_ids),
            "rarity": sentence_rarity(seq),
            "gt_text": respell(sp.DecodeIds(content_ids)),
        })
        if len(chosen) >= args.n:
            break
    print(f"Got {len(chosen)} usable samples.")

    # ---- stratify by rarity ----
    chosen.sort(key=lambda s: s["rarity"])
    bins = np.array_split(chosen, args.bins)
    bin_labels = ["rarest", "rare", "medium", "common", "commonest"][: args.bins]

    # ---- diagnose batch ----
    @torch.no_grad()
    def diag_batch(samples: list[dict]) -> list[dict]:
        bsz = len(samples)
        max_len = max(len(s["seq"]) for s in samples)
        tokens = torch.zeros(bsz, max_len, dtype=torch.long, device=device)
        lengths = torch.zeros(bsz, dtype=torch.long, device=device)
        for j, s in enumerate(samples):
            tokens[j, : len(s["seq"])] = torch.tensor(s["seq"].copy(), dtype=torch.long, device=device)
            lengths[j] = len(s["seq"])

        mu, logvar = model.encoder(tokens, lengths)
        z = mu

        length_pred = None
        if hasattr(model, "length_head"):
            length_pred = model.length_head(z).argmax(dim=-1)

        decoded = _greedy_decode(model, z, device, cfg, sp)

        # Per-sample analysis
        out = []
        for j, s in enumerate(samples):
            text, eos = decoded[j]
            rec_words = text.split()
            gt_words = s["gt_text"].split()
            len_err = len(rec_words) - s["gt_len"]
            cycles = detect_cycling(rec_words)
            adj = adjacent_repeats(rec_words)
            verb = longest_verbatim_run(rec_words, gt_words)
            len_head_err = (
                int(length_pred[j].item()) - s["gt_len"]
                if length_pred is not None else None
            )
            out.append({
                "rarity": s["rarity"],
                "gt_len": s["gt_len"],
                "rec_len": len(rec_words),
                "len_err": len_err,
                "cycles": cycles,
                "adj_repeats": adj,
                "longest_verbatim": verb,
                "len_head_err": len_head_err,
                "z_norm": float(z[j].norm()),
                "logvar_mean": float(logvar[j].mean()),
                "logvar_max": float(logvar[j].max()),
            })
        return out

    # ---- run per bin ----
    bin_results: dict[str, list[dict]] = {}
    BS = 32
    for label, bin_samples in zip(bin_labels, bins):
        bin_samples = list(bin_samples)
        rrs = []
        for i in range(0, len(bin_samples), BS):
            rrs.extend(diag_batch(bin_samples[i : i + BS]))
        bin_results[label] = rrs

        cycle_rate = np.mean([r["cycles"] > 0 for r in rrs])
        mean_len_err = np.mean([r["len_err"] for r in rrs])
        mean_abs_err = np.mean([abs(r["len_err"]) for r in rrs])
        verbatim = np.mean([r["longest_verbatim"] for r in rrs])
        mean_rare = np.mean([r["rarity"] for r in rrs])
        lh_err = (
            np.mean([abs(r["len_head_err"]) for r in rrs if r["len_head_err"] is not None])
            if any(r["len_head_err"] is not None for r in rrs) else float("nan")
        )
        z_norm_m = np.mean([r["z_norm"] for r in rrs])
        logvar_m = np.mean([r["logvar_mean"] for r in rrs])

        print(
            f"\n[{label:>10}]  rarity={mean_rare:>6.2f}  "
            f"cycle_rate={cycle_rate:.0%}  "
            f"len_err={mean_len_err:+5.2f}  abs={mean_abs_err:4.2f}  "
            f"verbatim={verbatim:4.2f}  "
            f"lh_err={lh_err:4.2f}  "
            f"z_norm={z_norm_m:5.2f}  logvar={logvar_m:+.3f}"
        )

    # ---- correlations ----
    flat = [r for rs in bin_results.values() for r in rs]
    print("\n\n========== CORRELATIONS ==========")
    rar = np.array([r["rarity"] for r in flat])
    cyc = np.array([r["cycles"] for r in flat])
    abs_le = np.array([abs(r["len_err"]) for r in flat])
    verb = np.array([r["longest_verbatim"] for r in flat])
    z_norms = np.array([r["z_norm"] for r in flat])
    logvars = np.array([r["logvar_mean"] for r in flat])
    adj_r = np.array([r["adj_repeats"] for r in flat])

    def corr(a: np.ndarray, b: np.ndarray) -> float:
        if a.std() == 0 or b.std() == 0:
            return float("nan")
        return float(np.corrcoef(a, b)[0, 1])

    print(f"  rarity      ↔ cycles:        {corr(rar, cyc):+.3f}")
    print(f"  rarity      ↔ |len_err|:     {corr(rar, abs_le):+.3f}")
    print(f"  rarity      ↔ verbatim:      {corr(rar, verb):+.3f}")
    print(f"  rarity      ↔ z_norm:        {corr(rar, z_norms):+.3f}")
    print(f"  rarity      ↔ logvar_mean:   {corr(rar, logvars):+.3f}")
    print(f"  cycles      ↔ |len_err|:     {corr(cyc, abs_le):+.3f}")
    print(f"  cycles      ↔ adj_repeats:   {corr(cyc, adj_r):+.3f}")
    print(f"  |len_err|   ↔ z_norm:        {corr(abs_le, z_norms):+.3f}")
    print(f"  |len_err|   ↔ logvar_mean:   {corr(abs_le, logvars):+.3f}")

    if any(r["len_head_err"] is not None for r in flat):
        lhe = np.array([abs(r["len_head_err"]) for r in flat if r["len_head_err"] is not None])
        rar_lh = np.array([r["rarity"] for r in flat if r["len_head_err"] is not None])
        print(f"  rarity      ↔ |lh_err|:      {corr(rar_lh, lhe):+.3f}")

    # ---- failure-mode quantification ----
    print("\n\n========== FAILURE MODE COUNTS ==========")
    n = len(flat)
    n_cycle = sum(1 for r in flat if r["cycles"] > 0)
    n_overshoot = sum(1 for r in flat if r["len_err"] > 5)
    n_undershoot = sum(1 for r in flat if r["len_err"] < -5)
    n_clean = sum(1 for r in flat if r["cycles"] == 0 and abs(r["len_err"]) <= 2)
    n_perfect = sum(1 for r in flat if r["longest_verbatim"] >= r["gt_len"])
    print(f"  total={n}")
    print(f"  cycling             : {n_cycle:>5} ({n_cycle/n:.0%})")
    print(f"  length overshoot >5 : {n_overshoot:>5} ({n_overshoot/n:.0%})")
    print(f"  length undershoot<-5: {n_undershoot:>5} ({n_undershoot/n:.0%})")
    print(f"  clean (no cycle, |err|≤2): {n_clean:>5} ({n_clean/n:.0%})")
    print(f"  perfect verbatim    : {n_perfect:>5} ({n_perfect/n:.0%})")

    # ---- save ----
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump({"step": step, "bins": bin_results}, f, indent=2, default=str)
    print(f"\nSaved per-sample results to {args.out}")


if __name__ == "__main__":
    main()

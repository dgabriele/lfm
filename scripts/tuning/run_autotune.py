"""Run DepTreeVAE.autotune() on a small held-out batch from the dep-tree cache.

Loads best.pt locally, encodes ~64 cached samples to posterior mu, then
grid-searches decode-time knobs and prints ranked results.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import sentencepiece as spm
import torch
import yaml
from sentence_transformers import SentenceTransformer

sys.path.insert(0, "/home/daniel/projects/lfm/src")
from lfm.generator.dep_tree_vae.config import DepTreeVAEConfig
from lfm.generator.dep_tree_vae.model import DepTreeVAE
from lfm.translator.romanize import respell


def main() -> None:
    device = torch.device("cuda")

    cfg_dict = yaml.safe_load(open("/home/daniel/projects/lfm/configs/dep_tree_vae_vast.yaml"))
    cfg_dict.pop("model_type", None)
    cfg_dict["dataset_path"] = "/home/daniel/projects/lfm/data/datasets/english-dep-trees-v16"
    cfg_dict["spm_model_path"] = "/home/daniel/projects/lfm/data/models/v15b_ipa/spm.model"
    cfg_dict["output_dir"] = "/tmp/autotune_dummy"
    sp = spm.SentencePieceProcessor(model_file=cfg_dict["spm_model_path"])
    spm_size = sp.get_piece_size()
    vocab_size = 8050  # matches trainer log

    # Peek at the checkpoint so we can disable training-only heads that
    # weren't present in this older state dict.
    ckpt_path = "/home/daniel/projects/lfm/data/models/dep_tree_vae_v1/best.pt"
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    if not any(k.startswith("length_head") for k in ckpt["model_state"]):
        cfg_dict["length_pred_weight"] = 0.0
        cfg_dict["use_predicted_length_at_decode"] = False

    cfg = DepTreeVAEConfig(**cfg_dict)
    model = DepTreeVAE(cfg, vocab_size).to(device)
    model.load_state_dict(ckpt["model_state"], strict=False)
    model.eval()
    print(f"Loaded best.pt @ step {ckpt.get('global_step', '?')}")

    # ---- pick samples and encode ----
    cache_dir = Path("/home/daniel/projects/lfm/data/datasets/english-dep-trees-v16/cache_depth4")
    interleaved = np.load(cache_dir / "interleaved.npy", mmap_mode="r")
    index = np.load(cache_dir / "index.npy")  # (N, 4) — start, len, skel_start, skel_len

    np.random.seed(0)
    candidate = np.random.permutation(len(index))
    chosen, source_texts = [], []
    target_n = 512
    for i in candidate:
        start = int(index[i, 0]); end = start + int(index[i, 1])
        seq = np.asarray(interleaved[start:end])
        if len(seq) > cfg.max_seq_len:
            continue
        ids = [int(t) for t in seq.tolist() if 0 < t < spm_size]
        if len(ids) < 4:
            continue
        chosen.append(seq)
        source_texts.append(respell(sp.DecodeIds(ids)))
        if len(chosen) >= target_n:
            break
    print(f"Selected {len(chosen)} samples for tuning batch")

    max_len = max(len(s) for s in chosen)
    tokens = torch.zeros(len(chosen), max_len, dtype=torch.long, device=device)
    lengths = torch.zeros(len(chosen), dtype=torch.long, device=device)
    for j, s in enumerate(chosen):
        tokens[j, : len(s)] = torch.tensor(s.copy(), dtype=torch.long, device=device)
        lengths[j] = len(s)

    with torch.no_grad():
        mu, _ = model.encoder(tokens, lengths)
    val_z = mu

    # ---- semantic scorer ----
    st = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device=str(device))

    # ---- run autotune ----
    print()
    print("Grid-searching decode-time knobs...")
    print("=" * 110)
    results = model.autotune(
        val_z=val_z,
        sp=sp,
        source_texts=source_texts,
        st_model=st,
        eos_boosts=(0.0, 1.0, 2.0, 3.0, 5.0, 8.0),
        expected_lens=(10, 12, 13, 14, 15),
        ngram_blocks=((3,), (3, 4), (2, 3, 4), (3, 4, 5)),
        verbose=True,
    )

    # ---- summarize ----
    print()
    print("=" * 110)
    print("TOP 8 BY COMPOSITE")
    print("=" * 110)
    for cfg_, m in results[:8]:
        print(f"  {cfg_.short()}  →  {m.short()}")

    print()
    print("BEST BY EACH METRIC")
    print("-" * 110)
    from lfm.generator.dep_tree_vae.autotune import DecodeAutotuner
    tuner_for_top = DecodeAutotuner(  # reuse the cache via fresh wrapper not needed; use static
        model, sp, cfg, device, val_z, source_texts=source_texts, st_model=st,
    )
    # We already have results — slice them
    for metric in ("semantic_score", "length_mae", "repetition_rate", "uniqueness", "eos_rate"):
        top = tuner_for_top.top_by(results, metric, k=1)
        if top:
            cfg_, m = top[0]
            print(f"  {metric:>16s}: {cfg_.short()}  →  {m.short()}")

    # ---- baseline comparison ----
    from lfm.generator.dep_tree_vae.autotune import DecodeConfig
    baseline = DecodeConfig(eos_boost=3.0, expected_len=13, ngram_block=(3, 4))
    baseline_metrics = next((m for c, m in results if c == baseline), None)
    print()
    print(f"BASELINE (current trainer defaults): {baseline.short()}")
    if baseline_metrics:
        print(f"  → {baseline_metrics.short()}")
    best_cfg, best_m = results[0]
    print(f"BEST CONFIG: {best_cfg.short()}")
    print(f"  → {best_m.short()}")
    if baseline_metrics:
        d_comp = best_m.composite - baseline_metrics.composite
        d_sem = (best_m.semantic_score or 0) - (baseline_metrics.semantic_score or 0)
        d_mae = (best_m.length_mae or 0) - (baseline_metrics.length_mae or 0)
        d_rep = best_m.repetition_rate - baseline_metrics.repetition_rate
        print(
            f"  Δ vs baseline: composite{d_comp:+.3f}  "
            f"sem{d_sem:+.3f}  mae{d_mae:+.2f}  rep{d_rep:+.2%}"
        )


if __name__ == "__main__":
    main()

"""Large-N linguistic analysis of a checkpoint.

Generates N posterior samples (encode from cache, decode) and N prior
samples (z ~ N(0, 1), decode), then reports:

  - Syllable distribution: histogram of words by syllable count, plus
    fraction of mono-syllabic vs ≥3-syllable words. Direct read on the
    "is the model producing word-shaped output or fragments" question.
  - Char count distribution.
  - Word-frequency Top-20 (compared to training corpus).
  - Diversity: TTR, distinct-1/2/3, hapax rate.
  - Reconstruction quality (chrF) on posterior samples.
  - Bigram coverage in training corpus (well-formedness proxy).
  - Zipf exponent fit on the generated corpus.

Compared side-by-side: posterior vs prior vs training data baseline.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import sentencepiece as spm
import torch
import yaml

sys.path.insert(0, "/workspace/lfm/src")
sys.path.insert(0, "/home/daniel/projects/lfm/src")
from lfm.generator.dep_tree_vae.config import DepTreeVAEConfig
from lfm.generator.dep_tree_vae.model import DepTreeVAE
from lfm.generator.dep_tree_vae.trainer import _greedy_decode
from lfm.generator.dep_tree_vae._lingmetrics import (
    chrf,
    distinct_n,
    mean_chars,
    mean_chrf,
    mean_syllables,
    syllable_count,
)
from lfm.translator.romanize import respell


def fit_zipf_exponent(freqs: list[int]) -> float:
    """Slope of log(rank) vs log(freq), an estimate of the Zipf exponent."""
    if len(freqs) < 5:
        return float("nan")
    sorted_f = sorted(freqs, reverse=True)
    ranks = np.log(np.arange(1, len(sorted_f) + 1, dtype=np.float64))
    fs = np.log(np.array(sorted_f, dtype=np.float64))
    # OLS slope: -s where freq ~ rank^(-s)
    slope, _ = np.polyfit(ranks, fs, deg=1)
    return float(-slope)


def analyze(texts: list[str], label: str, ref_bigrams: set | None = None) -> dict:
    all_words = [w for t in texts for w in t.split()]
    n_words = len(all_words)
    if n_words == 0:
        return {"label": label, "n_samples": len(texts), "n_words": 0}

    # Syllable distribution
    syl_counts = [syllable_count(w) for w in all_words]
    syl_hist = Counter(syl_counts)
    syl_keys = sorted(syl_hist)
    syl_pct = {f"syl_{k}_pct": syl_hist[k] / n_words for k in syl_keys}
    mono_pct = syl_hist.get(1, 0) / n_words
    multi_pct = sum(v for k, v in syl_hist.items() if k >= 2) / n_words
    triplus_pct = sum(v for k, v in syl_hist.items() if k >= 3) / n_words

    # Char-count distribution
    char_lengths = [len(w) for w in all_words]
    short_pct = sum(1 for c in char_lengths if c <= 2) / n_words
    long_pct = sum(1 for c in char_lengths if c >= 6) / n_words

    # Word frequency
    word_freq = Counter(all_words)
    types = len(word_freq)
    ttr = types / n_words
    hapax_rate = sum(1 for c in word_freq.values() if c == 1) / max(types, 1)
    top20 = word_freq.most_common(20)

    # Bigram coverage in training (well-formedness proxy)
    bigram_cov = None
    if ref_bigrams is not None:
        gen_bigrams = []
        for t in texts:
            words = t.split()
            for i in range(len(words) - 1):
                gen_bigrams.append((words[i], words[i + 1]))
        if gen_bigrams:
            covered = sum(1 for bg in gen_bigrams if bg in ref_bigrams)
            bigram_cov = covered / len(gen_bigrams)

    # Diversity
    d1 = distinct_n(texts, 1)
    d2 = distinct_n(texts, 2)
    d3 = distinct_n(texts, 3)

    # Zipf
    zipf_s = fit_zipf_exponent(list(word_freq.values()))

    return {
        "label": label,
        "n_samples": len(texts),
        "n_words": n_words,
        "n_types": types,
        "ttr": ttr,
        "hapax_rate": hapax_rate,
        "mean_syllables": float(np.mean(syl_counts)),
        "mean_chars": float(np.mean(char_lengths)),
        "mono_syl_pct": mono_pct,
        "multi_syl_pct": multi_pct,
        "triplus_syl_pct": triplus_pct,
        "short_pct_le2chars": short_pct,
        "long_pct_ge6chars": long_pct,
        "syl_distribution": dict(syl_hist),
        "distinct_1": d1,
        "distinct_2": d2,
        "distinct_3": d3,
        "zipf_exponent": zipf_s,
        "bigram_in_training_cov": bigram_cov,
        "top20": [(w, c) for w, c in top20],
    }


def print_stats(s: dict) -> None:
    print(f"\n=== {s['label'].upper()}  (n={s['n_samples']} samples, {s.get('n_words', 0)} words) ===")
    if not s.get("n_words"):
        print("  (no data)")
        return
    print(
        f"  types={s['n_types']:6d}  ttr={s['ttr']:.3f}  hapax={s['hapax_rate']:.3f}  "
        f"distinct-1={s['distinct_1']:.3f}  distinct-2={s['distinct_2']:.3f}  distinct-3={s['distinct_3']:.3f}"
    )
    print(
        f"  mean_syl={s['mean_syllables']:.2f}  mean_chars={s['mean_chars']:.2f}  "
        f"zipf_s={s['zipf_exponent']:.2f}"
    )
    print(
        f"  syllable mix: 1={s['mono_syl_pct']:.0%}  2+={s['multi_syl_pct']:.0%}  3+={s['triplus_syl_pct']:.0%}"
    )
    print(
        f"  short(≤2 chars)={s['short_pct_le2chars']:.0%}  long(≥6 chars)={s['long_pct_ge6chars']:.0%}"
    )
    if s.get("bigram_in_training_cov") is not None:
        print(f"  bigram_in_training_cov={s['bigram_in_training_cov']:.0%}")
    print(f"  syllable histogram: {s['syl_distribution']}")
    print(f"  top-20 words: {[w for w, _ in s['top20']]}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default=None)
    p.add_argument("--config", default=None)
    p.add_argument("--n", type=int, default=512, help="samples per mode")
    p.add_argument("--out", default="/tmp/lingdiag.json")
    args = p.parse_args()

    # Defaults
    if Path("/workspace/lfm").exists():
        ckpt_default = "/workspace/lfm/data/models/dep_tree_vae_v1/resume.pt"
        cfg_default = "/workspace/lfm/configs/dep_tree_vae_vast.yaml"
        cache_dir = Path("/workspace/lfm/data/datasets/english-dep-trees-v16-depth4/cache")
        spm_default = "/workspace/lfm/data/models/v15b_ipa/spm.model"
    else:
        ckpt_default = "/home/daniel/projects/lfm/data/models/dep_tree_vae_v1/best.pt"
        cfg_default = "/home/daniel/projects/lfm/configs/dep_tree_vae_vast.yaml"
        cache_dir = Path("/home/daniel/projects/lfm/data/datasets/english-dep-trees-v16/cache_depth4")
        spm_default = "/home/daniel/projects/lfm/data/models/v15b_ipa/spm.model"

    ckpt_path = args.ckpt or ckpt_default
    cfg_path = args.config or cfg_default

    # Config
    cfg_dict = yaml.safe_load(open(cfg_path))
    cfg_dict.pop("model_type", None)
    cfg_dict["spm_model_path"] = spm_default
    cfg_dict["dataset_path"] = str(cache_dir.parent)
    cfg_dict["output_dir"] = "/tmp/lingdiag_dummy"
    cfg_dict["corpus_unigram_path"] = ""  # don't load during analysis

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    state = ckpt["model_state"]

    # Adapt to old checkpoints: disable heads/role_emb if not present
    if not any(k.startswith("length_head") for k in state):
        cfg_dict["length_pred_weight"] = 0.0
        cfg_dict["use_predicted_length_at_decode"] = False
    if not any(k.startswith("tokens_per_role_head") for k in state):
        cfg_dict["tokens_per_role_weight"] = 0.0
    cfg_dict["use_decoder_role_emb"] = any(k.startswith("decoder_role_emb") for k in state)

    cfg = DepTreeVAEConfig(**cfg_dict)
    sp = spm.SentencePieceProcessor(model_file=cfg.spm_model_path)
    spm_size = sp.get_piece_size()
    vocab_size = 8050

    model = DepTreeVAE(cfg, vocab_size).to(device)
    model.load_state_dict(state, strict=False)
    model.eval()
    print(
        f"Loaded {ckpt_path} @ step {ckpt.get('global_step', '?')}, "
        f"val_best={ckpt.get('best_val_loss', '?')}"
    )

    # ---- collect training reference bigrams (for well-formedness check) ----
    interleaved = np.load(cache_dir / "interleaved.npy", mmap_mode="r")
    index = np.load(cache_dir / "index.npy")
    print("Building training-corpus bigram set + sample text for posterior decode...")
    rng = np.random.default_rng(0)
    train_texts: list[str] = []
    candidate = rng.permutation(len(index))
    for i in candidate[: args.n * 4]:  # oversample then trim
        start = int(index[i, 0])
        end = start + int(index[i, 1])
        seq = np.asarray(interleaved[start:end])
        if len(seq) > cfg.max_seq_len:
            continue
        ids = [int(t) for t in seq.tolist() if 0 < t < spm_size]
        if len(ids) < 4:
            continue
        train_texts.append(respell(sp.DecodeIds(ids)))
        if len(train_texts) >= args.n:
            break

    train_bigrams: set[tuple[str, str]] = set()
    for t in train_texts:
        words = t.split()
        for i in range(len(words) - 1):
            train_bigrams.add((words[i], words[i + 1]))
    print(f"  {len(train_texts)} training texts, {len(train_bigrams)} unique bigrams")

    # ---- posterior decode ----
    print("\nGenerating posterior samples...")
    posterior_texts: list[str] = []
    posterior_gts: list[str] = []
    BS = 32
    cand = rng.permutation(len(index))
    chosen = []
    for i in cand:
        start = int(index[i, 0])
        end = start + int(index[i, 1])
        seq = np.asarray(interleaved[start:end])
        if len(seq) > cfg.max_seq_len:
            continue
        ids = [int(t) for t in seq.tolist() if 0 < t < spm_size]
        if len(ids) < 4:
            continue
        chosen.append((seq, ids))
        if len(chosen) >= args.n:
            break

    with torch.no_grad():
        for batch_start in range(0, len(chosen), BS):
            batch = chosen[batch_start : batch_start + BS]
            bsz = len(batch)
            max_len = max(len(s) for s, _ in batch)
            tokens = torch.zeros(bsz, max_len, dtype=torch.long, device=device)
            lengths = torch.zeros(bsz, dtype=torch.long, device=device)
            for j, (s, _) in enumerate(batch):
                tokens[j, : len(s)] = torch.tensor(s.copy(), dtype=torch.long, device=device)
                lengths[j] = len(s)
            mu, logvar = model.encoder(tokens, lengths)
            decoded = _greedy_decode(model, mu, device, cfg, sp, logvar=logvar)
            for j, (_, ids) in enumerate(batch):
                gt = respell(sp.DecodeIds(ids))
                rec = respell(decoded[j][0])
                posterior_gts.append(gt)
                posterior_texts.append(rec)

    # ---- prior decode ----
    print("Generating prior samples...")
    prior_texts: list[str] = []
    with torch.no_grad():
        for batch_start in range(0, args.n, BS):
            bsz = min(BS, args.n - batch_start)
            z = torch.randn(bsz, cfg.latent.total_dim, device=device)
            decoded = _greedy_decode(model, z, device, cfg, sp)
            prior_texts.extend(respell(t) for t, _ in decoded)

    # ---- analyze ----
    train_stats = analyze(train_texts, "training-data", ref_bigrams=None)
    post_stats = analyze(posterior_texts, "posterior", ref_bigrams=train_bigrams)
    prior_stats = analyze(prior_texts, "prior", ref_bigrams=train_bigrams)

    # ---- chrF for posterior ----
    chrf_score = mean_chrf(posterior_texts, posterior_gts)
    post_stats["mean_chrF_vs_gt"] = chrf_score

    print_stats(train_stats)
    print_stats(post_stats)
    print_stats(prior_stats)

    print(f"\nPosterior chrF vs GT: {chrf_score:.3f}")

    # ---- side-by-side comparison ----
    print("\n\n" + "=" * 60)
    print("SIDE-BY-SIDE")
    print("=" * 60)
    keys = [
        "n_words", "n_types", "ttr", "hapax_rate", "mean_syllables",
        "mean_chars", "mono_syl_pct", "multi_syl_pct", "triplus_syl_pct",
        "short_pct_le2chars", "distinct_1", "distinct_2", "distinct_3",
        "zipf_exponent", "bigram_in_training_cov",
    ]
    print(f"{'metric':<28} {'training':>12} {'posterior':>12} {'prior':>12}")
    for k in keys:
        tv = train_stats.get(k)
        pv = post_stats.get(k)
        rv = prior_stats.get(k)
        def fmt(v):
            if v is None:
                return "    -"
            if isinstance(v, int):
                return f"{v:12d}"
            return f"{v:12.3f}"
        print(f"{k:<28} {fmt(tv)} {fmt(pv)} {fmt(rv)}")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(
            {"training": train_stats, "posterior": post_stats, "prior": prior_stats},
            f, indent=2, default=str,
        )
    print(f"\nSaved to {args.out}")


if __name__ == "__main__":
    main()

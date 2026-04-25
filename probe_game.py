"""Quick probe: how discriminable are AR DepTreeVAE reconstructions?

Procedure:
  1. Sample N passages from the dep-tree cache.
  2. Encode each through the model's posterior → mu (no sampling).
  3. Decode → IPA → respell to ASCII English.
  4. Re-embed both source and decoded text with MiniLM-L6-v2.
  5. Run 16-way contrastive discrimination:
        random distractors        — upper bound (easy task)
        hard distractors (NN-15)  — game-realistic (similar embeddings)

A learnable speaker projector + receiver would do better than this
upper-bound proxy; the proxy nonetheless tells us whether the decoder
preserves enough semantics for any receiver to discriminate.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import sentencepiece as spm
import torch
import torch.nn.functional as F
import yaml
from sentence_transformers import SentenceTransformer

sys.path.insert(0, "/home/daniel/projects/lfm/src")
from lfm.generator.dep_tree_vae.config import DepTreeVAEConfig
from lfm.generator.dep_tree_vae.model import DepTreeVAE
from lfm.generator.dep_tree_vae.trainer import _greedy_decode
from lfm.translator.romanize import respell


def main() -> None:
    device = torch.device("cuda")

    # ---- config ----
    cfg_path = "/home/daniel/projects/lfm/configs/dep_tree_vae_vast.yaml"
    cfg_dict = yaml.safe_load(open(cfg_path))
    cfg_dict.pop("model_type", None)
    cfg_dict["dataset_path"] = "/home/daniel/projects/lfm/data/datasets/english-dep-trees-v16"
    cfg_dict["spm_model_path"] = "/home/daniel/projects/lfm/data/models/v15b_ipa/spm.model"
    cfg_dict["output_dir"] = "/tmp/probe_dummy"
    sp = spm.SentencePieceProcessor(model_file=cfg_dict["spm_model_path"])
    spm_size = sp.get_piece_size()  # 8000

    # vocab size matches "vocab=8050" reported by trainer
    vocab_size = 8050

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
    print(f"Loaded best.pt from step {ckpt.get('global_step', '?')}")

    # ---- dep-tree cache ----
    cache_dir = Path("/home/daniel/projects/lfm/data/datasets/english-dep-trees-v16/cache_depth4")
    interleaved = np.load(cache_dir / "interleaved.npy", mmap_mode="r")
    index = np.load(cache_dir / "index.npy")  # (N, 4): [interleaved_start, interleaved_len, skel_start, skel_len]
    print(f"Cache: {len(index)} samples, interleaved shape {interleaved.shape}")

    # ---- sample passages ----
    N = 512
    np.random.seed(42)
    candidate_idxs = np.random.permutation(len(index))
    chosen, source_texts = [], []
    max_seq_len = cfg.max_seq_len

    for i in candidate_idxs:
        start = int(index[i, 0])
        end = start + int(index[i, 1])
        seq = np.asarray(interleaved[start:end])
        if len(seq) > max_seq_len:
            continue
        ids = [int(t) for t in seq.tolist() if 0 < t < spm_size]
        if len(ids) < 4:  # skip degenerate samples
            continue
        chosen.append(seq)
        source_texts.append(respell(sp.DecodeIds(ids)))
        if len(chosen) >= N:
            break

    print(f"Selected {len(chosen)} samples")
    N = len(chosen)

    # ---- encode + decode in batches ----
    decoded_texts: list[str] = []
    batch_size = 32

    for batch_start in range(0, N, batch_size):
        seqs = chosen[batch_start : batch_start + batch_size]
        bsz = len(seqs)
        max_len = max(len(s) for s in seqs)

        tokens = torch.zeros(bsz, max_len, dtype=torch.long, device=device)
        lengths = torch.zeros(bsz, dtype=torch.long, device=device)
        for j, s in enumerate(seqs):
            tokens[j, : len(s)] = torch.tensor(s.copy(), dtype=torch.long, device=device)
            lengths[j] = len(s)

        with torch.no_grad():
            mu, _ = model.encoder(tokens, lengths)
            z = mu
            decoded = _greedy_decode(model, z, device, cfg, sp)
            decoded_texts.extend(respell(t) for t, _ in decoded)

        if batch_start % (batch_size * 4) == 0:
            print(f"  decoded {batch_start + bsz}/{N}")

    # ---- re-embed both ----
    print("Encoding with MiniLM-L6-v2...")
    st = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device=str(device))
    with torch.no_grad():
        src_embs = st.encode(source_texts, convert_to_tensor=True, device=str(device), batch_size=64)
        dec_embs = st.encode(decoded_texts, convert_to_tensor=True, device=str(device), batch_size=64)

    src_norm = F.normalize(src_embs, dim=-1)
    dec_norm = F.normalize(dec_embs, dim=-1)

    # ---- discrimination at 16-way ----
    NEG = 15
    n_trials = 500
    rng = np.random.default_rng(123)

    # random distractors
    correct_random = 0
    for _ in range(n_trials):
        target = int(rng.integers(N))
        pool = np.array([i for i in range(N) if i != target])
        distractors = rng.choice(pool, size=NEG, replace=False)
        cands = np.concatenate([[target], distractors])
        rng.shuffle(cands)
        tgt_pos = int(np.where(cands == target)[0][0])

        cand_srcs = src_norm[cands]
        sim = cand_srcs @ dec_norm[target]
        if int(sim.argmax().item()) == tgt_pos:
            correct_random += 1

    # hard distractors (top-K nearest in source-embedding space)
    src_sim = src_norm @ src_norm.T  # (N, N)
    src_sim.fill_diagonal_(-1.0)
    correct_hard = 0
    for _ in range(n_trials):
        target = int(rng.integers(N))
        nbrs = src_sim[target].topk(NEG).indices.cpu().numpy()
        cands = np.concatenate([[target], nbrs])
        rng.shuffle(cands)
        tgt_pos = int(np.where(cands == target)[0][0])

        cand_srcs = src_norm[cands]
        sim = cand_srcs @ dec_norm[target]
        if int(sim.argmax().item()) == tgt_pos:
            correct_hard += 1

    # also: full N-way rank of target — what % of trials does target rank top-1, top-3?
    full_sim = dec_norm @ src_norm.T  # (N, N) — for each decoded i, similarity to each src j
    ranks = (-full_sim).argsort(dim=-1)  # (N, N) sorted indices
    target_ranks = (ranks == torch.arange(N, device=device).unsqueeze(1)).int().argmax(dim=-1)
    full_top1 = float((target_ranks == 0).float().mean())
    full_top3 = float((target_ranks < 3).float().mean())
    median_rank = int(target_ranks.float().median().item())

    print()
    print("=" * 70)
    print("DISCRIMINATION RESULTS")
    print("=" * 70)
    print(f"N samples         : {N}")
    print(f"Random baseline   : {1/16:.1%}  (16-way)")
    print(f"16-way RANDOM neg : {correct_random / n_trials:.1%}  ({correct_random}/{n_trials})")
    print(f"16-way HARD neg   : {correct_hard / n_trials:.1%}  ({correct_hard}/{n_trials})")
    print()
    print(f"{N}-way full ranking:")
    print(f"  top-1 acc       : {full_top1:.1%}")
    print(f"  top-3 acc       : {full_top3:.1%}")
    print(f"  median rank     : {median_rank}  (random = {N // 2})")

    # ---- diversity/quality stats ----
    n_unique = len(set(decoded_texts))
    avg_dec_len = np.mean([len(t.split()) for t in decoded_texts])
    avg_src_len = np.mean([len(t.split()) for t in source_texts])
    pairwise_dec = (dec_norm @ dec_norm.T).cpu().numpy()
    np.fill_diagonal(pairwise_dec, np.nan)
    mean_pairwise = float(np.nanmean(pairwise_dec))
    print()
    print(f"Decoded uniqueness: {n_unique}/{N} unique")
    print(f"Length: src avg={avg_src_len:.1f} words, dec avg={avg_dec_len:.1f} words")
    print(f"Mean pairwise cosine (decoded re-embeddings): {mean_pairwise:.3f}")

    # ---- sample qualitative comparison ----
    print()
    print("Sample reconstructions:")
    for i in range(min(8, N)):
        print(f"  [{i}] SRC: {source_texts[i][:140]}")
        print(f"      DEC: {decoded_texts[i][:140]}")
        print(f"      cos(src,dec)={float((src_norm[i] * dec_norm[i]).sum()):.3f}")


if __name__ == "__main__":
    main()

"""Intrinsic discriminability diagnostic on the source embedding store.

Question: how distinguishable are passages from their *own* hard-negative
pool (same KMeans cluster) in the raw Qwen embedding space, before any
agent surgery? If many anchors have a near-clone in their cluster
(cosine > 0.95), the agent's contrastive accuracy is *bounded by source
geometry*, not by anything our architecture can fix.

Reports — purely numerical, no embedding text or token surface ever
surfaced:
  1. Distribution of cos(anchor, max-similar same-cluster passage)
     across many anchors. Tail probability of "near-clone" pairs.
  2. Best-case argmax-accuracy in a simulated 32×32 contrastive pool:
     the source itself as anchor vs. cluster-mates + random-batch
     negatives. This is a *trivial-100%* baseline since query == anchor;
     used to verify the diagnostic harness.
  3. Noise-robustness sweep: same task but the anchor is perturbed by
     scaled Gaussian noise at varying levels. The accuracy curve gives
     a model-agnostic picture of "how much surface noise can tolerated
     before discrimination collapses on this dataset."

Two geometries reported per metric:
  * mean-pool: average across the 8 positions to a flat (D=896) vector,
    then cosine — proxy for "average" similarity.
  * per-position-flat: L2-normalise each position separately, flatten
    to (8*896), dot product. Equivalent to summed per-position cosines
    — same geometry the agent's per-position InfoNCE operates in.

Usage:
  poetry run python scripts/diagnose_embedding_discriminability.py \\
      data/embeddings_qwen_subset
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def _load_store(store_dir: Path) -> tuple[np.ndarray, np.ndarray, dict]:
    emb = np.load(str(store_dir / "embeddings.npy"), mmap_mode="r")
    labels = np.load(str(store_dir / "cluster_labels.npy"), mmap_mode="r")
    with (store_dir / "cluster_index.json").open() as fh:
        idx = {int(k): v for k, v in json.load(fh).items()}
    return emb, labels, idx


def _meanpool_cos(emb_a: np.ndarray, emb_b: np.ndarray) -> np.ndarray:
    """emb_a: (N, P, D); emb_b: (M, P, D). Return cos sim matrix (N, M)
    of mean-pooled-over-P vectors."""
    a = emb_a.mean(axis=1).astype(np.float32)
    b = emb_b.mean(axis=1).astype(np.float32)
    a = a / (np.linalg.norm(a, axis=-1, keepdims=True) + 1e-8)
    b = b / (np.linalg.norm(b, axis=-1, keepdims=True) + 1e-8)
    return a @ b.T


def _per_position_flat_dot(emb_a: np.ndarray, emb_b: np.ndarray) -> np.ndarray:
    """Per-position L2-norm + flatten: result rows are unit-per-position
    flat vectors. Their dot product = sum-of-per-position-cosines, which
    is the agent's per-position InfoNCE summed score."""
    a = emb_a.astype(np.float32)
    b = emb_b.astype(np.float32)
    a = a / (np.linalg.norm(a, axis=-1, keepdims=True) + 1e-8)
    b = b / (np.linalg.norm(b, axis=-1, keepdims=True) + 1e-8)
    a_flat = a.reshape(a.shape[0], -1)
    b_flat = b.reshape(b.shape[0], -1)
    # NB: row norms are sqrt(P), so dot/P = mean-of-per-position-cosines.
    # Return raw dot (= sum of P cosines) to match agent geometry.
    return a_flat @ b_flat.T


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("store_dir")
    p.add_argument("--n-anchors", type=int, default=1000,
                   help="anchors sampled for the within-cluster max-sim distribution")
    p.add_argument("--pool-batches", type=int, default=200,
                   help="batches of size 32 used for the contrastive harness")
    p.add_argument("--pool-batch", type=int, default=32)
    p.add_argument("--num-distractors", type=int, default=31)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    rng = np.random.default_rng(args.seed)
    store_dir = Path(args.store_dir)
    emb, labels, cluster_index = _load_store(store_dir)
    n_total, P, D = emb.shape
    print(f"store: shape={emb.shape}  n_clusters={len(cluster_index)}")

    # ── (1) within-cluster max-similarity distribution ─────────────────
    print("\n[1] within-cluster max-similarity distribution")
    print(f"    sampling {args.n_anchors} anchors with cluster_size >= 2 ...")
    valid_clusters = [cid for cid, members in cluster_index.items() if len(members) >= 2]
    anchors_idx = []
    while len(anchors_idx) < args.n_anchors:
        cid = int(rng.choice(valid_clusters))
        members = cluster_index[cid]
        anchors_idx.append(int(rng.choice(members)))

    max_cos_meanpool = []
    max_cos_perpos = []
    for ai in anchors_idx:
        cid = int(labels[ai])
        members = [m for m in cluster_index[cid] if m != ai]
        if not members:
            continue
        anchor = emb[ai:ai + 1]                                           # (1, P, D)
        mates = emb[members]                                              # (k, P, D)
        sim_mp = _meanpool_cos(anchor, mates)[0]                          # (k,)
        sim_pp = _per_position_flat_dot(anchor, mates)[0]                 # (k,)
        max_cos_meanpool.append(float(sim_mp.max()))
        max_cos_perpos.append(float(sim_pp.max()))
    mp = np.array(max_cos_meanpool)
    pp = np.array(max_cos_perpos)

    def _quantiles(arr, label):
        print(f"    {label}: mean={arr.mean():.4f}  median={np.median(arr):.4f}  "
              f"p90={np.quantile(arr, 0.90):.4f}  p99={np.quantile(arr, 0.99):.4f}  "
              f"max={arr.max():.4f}")
    _quantiles(mp, "mean-pool       (cos in [-1,1])")
    _quantiles(pp, f"per-position flat (raw dot, max ≈ {P})")

    # Tail thresholds
    for thr in (0.90, 0.95, 0.98, 0.99):
        frac_mp = (mp >= thr).mean()
        print(f"    fraction with mean-pool max-sim ≥ {thr:.2f}: {frac_mp*100:5.1f}%")
    # For per-position flat, the equivalent of cos>=0.95 is dot>=P*0.95
    for thr_per_pos in (0.90, 0.95, 0.98, 0.99):
        thr = thr_per_pos * P
        frac_pp = (pp >= thr).mean()
        print(f"    fraction with mean-position-cos ≥ {thr_per_pos:.2f} (dot≥{thr:.2f}): {frac_pp*100:5.1f}%")

    # ── (2) trivial-100% harness sanity-check ──────────────────────────
    print("\n[2] contrastive harness sanity check (anchor vs. self+cluster-mates+random)")
    print(f"    batches={args.pool_batches} batch={args.pool_batch} num_distractors={args.num_distractors}")
    n_correct_mp = 0
    n_correct_pp = 0
    n_total_q = 0
    for _ in range(args.pool_batches):
        batch_idx = rng.integers(0, n_total, size=args.pool_batch)
        # Hard distractors: same-cluster
        dist_idx = np.zeros((args.pool_batch, args.num_distractors), dtype=np.intp)
        for i, ai in enumerate(batch_idx):
            cid = int(labels[ai])
            members = cluster_index[cid]
            if len(members) >= args.num_distractors + 1:
                pool = [m for m in members if m != int(ai)]
                dist_idx[i] = rng.choice(pool, size=args.num_distractors, replace=False)
            else:
                dist_idx[i] = rng.integers(0, n_total, size=args.num_distractors)
        anchors = emb[batch_idx]                                          # (B, P, D)
        # Build full candidate pool: anchor + B-1 in-batch + K distractors
        all_cand_idx = np.concatenate([batch_idx[:, None], dist_idx], axis=1)
        # Mean-pool similarity
        a_pooled = anchors.mean(axis=1).astype(np.float32)                # (B, D)
        a_pooled /= np.linalg.norm(a_pooled, axis=-1, keepdims=True) + 1e-8
        for i in range(args.pool_batch):
            cand_emb = emb[all_cand_idx[i]]                               # (1+K, P, D)
            c_mp = cand_emb.mean(axis=1).astype(np.float32)
            c_mp /= np.linalg.norm(c_mp, axis=-1, keepdims=True) + 1e-8
            scores_mp = c_mp @ a_pooled[i]                                # (1+K,)
            if int(scores_mp.argmax()) == 0:
                n_correct_mp += 1
            # Per-position flat
            a_pp = anchors[i].astype(np.float32)
            a_pp /= np.linalg.norm(a_pp, axis=-1, keepdims=True) + 1e-8
            a_pp_flat = a_pp.reshape(-1)
            c_pp = cand_emb.astype(np.float32)
            c_pp /= np.linalg.norm(c_pp, axis=-1, keepdims=True) + 1e-8
            c_pp_flat = c_pp.reshape(c_pp.shape[0], -1)
            scores_pp = c_pp_flat @ a_pp_flat
            if int(scores_pp.argmax()) == 0:
                n_correct_pp += 1
            n_total_q += 1
    print(f"    sanity-acc mean-pool      : {n_correct_mp/n_total_q*100:.2f}%  "
          f"(expect 100% — query is anchor itself)")
    print(f"    sanity-acc per-position   : {n_correct_pp/n_total_q*100:.2f}%")

    # ── (3) noise-robustness sweep ─────────────────────────────────────
    print("\n[3] noise-robustness sweep — anchor perturbed by scaled Gaussian noise")
    print("    measures the model-agnostic upper bound: at noise σ, how often does")
    print("    the perturbed anchor still rank closer to its source than to a hard neg?")
    print(f"    using same {args.pool_batches}*{args.pool_batch} pool, hard-neg-only contrast")
    sigmas = [0.0, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0]
    print(f"    {'σ':>5}  {'mean-pool':>12}  {'per-pos':>12}")
    rng_noise = np.random.default_rng(args.seed + 100)
    for sigma in sigmas:
        n_correct_mp = 0
        n_correct_pp = 0
        n_q = 0
        for _ in range(args.pool_batches):
            batch_idx = rng.integers(0, n_total, size=args.pool_batch)
            dist_idx = np.zeros((args.pool_batch, args.num_distractors), dtype=np.intp)
            for i, ai in enumerate(batch_idx):
                cid = int(labels[ai])
                members = cluster_index[cid]
                if len(members) >= args.num_distractors + 1:
                    pool = [m for m in members if m != int(ai)]
                    dist_idx[i] = rng.choice(pool, size=args.num_distractors, replace=False)
                else:
                    dist_idx[i] = rng.integers(0, n_total, size=args.num_distractors)
            anchors = emb[batch_idx].astype(np.float32)                   # (B, P, D)
            # Add noise to query
            noise = rng_noise.normal(0, sigma, size=anchors.shape).astype(np.float32)
            queries = anchors + noise * np.linalg.norm(anchors, axis=-1, keepdims=True)
            for i in range(args.pool_batch):
                cand_idx = np.concatenate([[batch_idx[i]], dist_idx[i]])
                cand_emb = emb[cand_idx].astype(np.float32)
                # mean-pool
                q_mp = queries[i].mean(axis=0)
                q_mp /= np.linalg.norm(q_mp) + 1e-8
                c_mp = cand_emb.mean(axis=1)
                c_mp /= np.linalg.norm(c_mp, axis=-1, keepdims=True) + 1e-8
                if int((c_mp @ q_mp).argmax()) == 0:
                    n_correct_mp += 1
                # per-position
                q_pp = queries[i] / (np.linalg.norm(queries[i], axis=-1, keepdims=True) + 1e-8)
                q_pp_flat = q_pp.reshape(-1)
                c_pp = cand_emb / (np.linalg.norm(cand_emb, axis=-1, keepdims=True) + 1e-8)
                c_pp_flat = c_pp.reshape(c_pp.shape[0], -1)
                if int((c_pp_flat @ q_pp_flat).argmax()) == 0:
                    n_correct_pp += 1
                n_q += 1
        print(f"    {sigma:>5.2f}  {n_correct_mp/n_q*100:>11.2f}%  {n_correct_pp/n_q*100:>11.2f}%")


if __name__ == "__main__":
    main()

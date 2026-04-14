"""Semantic topology preservation test for a trained dialogue agent.

Parallels ``scripts/eval_topology.py`` but runs through the full trained
dialogue game (diffusion z-generator + context transformer + frozen VAE
decoder), not just the bare VAE decoder.  This measures whether the
*agent's* learned message-generation preserves topology end-to-end.

Protocol:
  1. Sample ``num_pairs`` random embedding pairs from the store.
  2. For each embedding, run a single-turn dialogue forward → Neuroglot.
  3. For each pair, compute surface distances:
       - Normalized Levenshtein edit distance on the surface string
       - Jaccard distance on the emitted token-id multiset
     and input-space cosine distance on the embeddings.
  4. Report Spearman + Pearson correlations.

Usage::

    poetry run python scripts/eval_dialogue_topology.py \\
        --checkpoint data/dialogue_game_v8_qwen_contrastive/best.pt \\
        --decoder-path data/models/v8/vae_decoder.pt \\
        --spm-path data/phoneme_alphabet_multi.json \\
        --embedding-store data/embeddings_qwen_v2_clean \\
        --generator-name phoneme_vae --generator-vocab-size 30 \\
        --num-pairs 500
"""

from __future__ import annotations

import argparse
import logging

import numpy as np
import torch
from scipy import stats

# Reuse distance metrics + game loader from the other two scripts.
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from eval_topology import (  # type: ignore[import-not-found]
    normalized_edit_distance,
    token_jaccard_distance,
    cosine_distance,
)
from source_retrieval_eval import load_dialogue_game, generate_documents  # type: ignore[import-not-found]

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger(__name__)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--decoder-path", default="data/models/v7/vae_decoder.pt")
    ap.add_argument("--spm-path", default="data/models/v7/spm.model",
                    help=".model (SentencePiece) or .json (phoneme alphabet)")
    ap.add_argument("--embedding-store", default="data/embeddings")
    ap.add_argument("--generator-name", default="multilingual_vae",
                    choices=["multilingual_vae", "phoneme_vae"])
    ap.add_argument("--generator-vocab-size", type=int, default=None,
                    help="v8=30, v9=5001, v9.5=30819")
    ap.add_argument("--num-pairs", type=int, default=500)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    device = torch.device(args.device)
    rng = np.random.default_rng(args.seed)

    # ── Load embeddings ──
    from lfm.embeddings.store import EmbeddingStore
    store = EmbeddingStore(args.embedding_store)
    store.load()
    n = store.num_passages
    logger.info(f"Store: {n} passages, dim={store.embedding_dim}")

    # ── Sample 2*num_pairs distinct embeddings ──
    idx = rng.choice(n, size=2 * args.num_pairs, replace=False)
    idx_a = idx[: args.num_pairs]
    idx_b = idx[args.num_pairs:]
    emb_a = torch.tensor(store._embeddings[idx_a], dtype=torch.float32, device=device)
    emb_b = torch.tensor(store._embeddings[idx_b], dtype=torch.float32, device=device)
    embeddings_all = torch.cat([emb_a, emb_b], dim=0)
    del store

    # ── Load game ──
    logger.info(f"Loading dialogue game from {args.checkpoint}")
    game = load_dialogue_game(
        args.checkpoint, args.decoder_path, args.spm_path,
        args.embedding_store, device,
        generator_name=args.generator_name,
        generator_vocab_size=args.generator_vocab_size,
    )
    vocab_size = game.gen._vocab_size  # noqa: SLF001
    eos_id = game.gen.eos_id

    # ── Generate Neuroglot for all 2N embeddings ──
    all_docs: list[list[str]] = []
    for lo in range(0, embeddings_all.size(0), args.batch_size):
        batch = embeddings_all[lo: lo + args.batch_size]
        docs = generate_documents(game, batch, vocab_size, eos_id)
        all_docs.extend(docs)
        logger.info(f"  generated {lo + batch.size(0)}/{embeddings_all.size(0)}")

    # Flatten each multi-turn doc into one surface string.
    surfaces = [" | ".join(turns) for turns in all_docs]
    surf_a = surfaces[: args.num_pairs]
    surf_b = surfaces[args.num_pairs:]

    # ── Also collect token id streams for jaccard ──
    # Easier to re-emit: tokenize the surface back through the tokenizer.
    # But since we already have surface strings, approximate the token set
    # via whitespace split (good enough for Jaccard signal).
    def tokenize_for_jaccard(s: str) -> list[int]:
        # Hash tokens to ints via built-in hash (mod a large prime for stability).
        return [hash(t) & 0x7fffffff for t in s.replace("|", " ").split()]

    # ── Compute distances ──
    logger.info(f"Computing distances for {args.num_pairs} pairs")
    edit_d: list[float] = []
    jacc_d: list[float] = []
    emb_d: list[float] = []
    for i in range(args.num_pairs):
        a, b = surf_a[i], surf_b[i]
        edit_d.append(normalized_edit_distance(a, b))
        jacc_d.append(
            token_jaccard_distance(
                tokenize_for_jaccard(a), tokenize_for_jaccard(b),
            ),
        )
        emb_d.append(
            cosine_distance(
                emb_a[i].cpu().numpy(), emb_b[i].cpu().numpy(),
            ),
        )
    edit_d = np.asarray(edit_d)
    jacc_d = np.asarray(jacc_d)
    emb_d = np.asarray(emb_d)

    # ── Correlations ──
    logger.info("")
    logger.info("=" * 60)
    logger.info("Semantic topology preservation — trained dialogue agent")
    logger.info("=" * 60)
    logger.info(f"num_pairs = {args.num_pairs}")
    for name, d in [("edit", edit_d), ("jaccard", jacc_d)]:
        sp_r, sp_p = stats.spearmanr(emb_d, d)
        pr_r, pr_p = stats.pearsonr(emb_d, d)
        logger.info(
            f"  {name:>8} vs embedding cosine  "
            f"spearman ρ={sp_r:+.4f} (p={sp_p:.2e})  "
            f"pearson r={pr_r:+.4f} (p={pr_p:.2e})",
        )
    logger.info("")
    logger.info(
        f"surface mean len = {np.mean([len(s) for s in surfaces]):.1f}, "
        f"unique surfaces = {len(set(surfaces))}/{len(surfaces)}",
    )


if __name__ == "__main__":
    main()

"""Interpretation-topology eval for a trained dialogue agent.

The *correct* end-to-end test for LFM: not "does Qwen recover the source"
(reconstruction) but "do similar embeddings produce similar Qwen
interpretations of the agent's Neuroglot" (systematic perception).

Protocol:
  1. Sample N embeddings from the store.
  2. Run the trained dialogue game to produce Neuroglot per embedding.
  3. Prompt Qwen-Instruct to describe each Neuroglot as a scene/topic.
  4. SBERT-encode the interpretations.
  5. Correlate pairwise cosine-distance(interpretations) against
     pairwise cosine-distance(embeddings).

If ρ > 0 and significant, the agent's perception is systematic —
the LLM's interpretation tracks the input embedding topology, which
is what we need for the LLM to learn a coherent interpretation of the
agent's alien-ontology expression.

No source sentence is used; we do not expect interpretation to
reconstruct the source word-for-word.

Usage::

    poetry run python scripts/interpretation_topology_eval.py \\
        --checkpoint data/dialogue_game_v8_qwen_contrastive/best.pt \\
        --decoder-path data/models/v8/vae_decoder.pt \\
        --spm-path data/phoneme_alphabet_multi.json \\
        --embedding-store data/embeddings_qwen_v2_clean \\
        --generator-name phoneme_vae --generator-vocab-size 30 \\
        --num-samples 200
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import torch
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent))
from source_retrieval_eval import (  # type: ignore[import-not-found]
    generate_documents,
    interpret,
    join_turns,
    load_dialogue_game,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger(__name__)


def pairwise_cosine_distance(X: np.ndarray) -> np.ndarray:
    """Return upper-triangular pairwise cosine distances (1 - cos)."""
    Xn = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)
    sim = Xn @ Xn.T
    i, j = np.triu_indices(X.shape[0], k=1)
    return 1.0 - sim[i, j]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--decoder-path", default="data/models/v7/vae_decoder.pt")
    ap.add_argument("--spm-path", default="data/models/v7/spm.model")
    ap.add_argument("--embedding-store", default="data/embeddings")
    ap.add_argument("--generator-name", default="multilingual_vae",
                    choices=["multilingual_vae", "phoneme_vae"])
    ap.add_argument("--generator-vocab-size", type=int, default=None)
    ap.add_argument("--qwen-model", default="Qwen/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--sbert-model", default="sentence-transformers/all-MiniLM-L6-v2")
    ap.add_argument("--num-samples", type=int, default=200)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--output-samples", default=None,
                    help="Optional JSONL path to dump (embedding_idx, neuroglot, interpretation)")
    args = ap.parse_args()

    device = torch.device(args.device)
    rng = np.random.default_rng(args.seed)

    # ── 1. Sample embeddings (diverse: one per cluster, then random fill) ──
    from lfm.embeddings.store import EmbeddingStore
    store = EmbeddingStore(args.embedding_store)
    store.load()
    n = store.num_passages
    logger.info(f"Store: {n} passages, dim={store.embedding_dim}")

    # Diversify by sampling distinct clusters so topology is measured
    # across the whole manifold, not just within one cluster.
    cluster_ids = list(store._cluster_index.keys())
    if len(cluster_ids) >= args.num_samples:
        chosen = rng.choice(cluster_ids, size=args.num_samples, replace=False)
        indices = [
            int(store.sample_from_cluster(int(c), 1, rng=rng)[0])
            for c in chosen
        ]
    else:
        indices = list(rng.choice(n, size=args.num_samples, replace=False).tolist())
    emb = np.asarray(store._embeddings[indices], dtype=np.float32)
    del store

    # ── 2. Generate Neuroglot via the trained game ──
    logger.info(f"Loading dialogue game: {args.checkpoint}")
    game = load_dialogue_game(
        args.checkpoint, args.decoder_path, args.spm_path,
        args.embedding_store, device,
        generator_name=args.generator_name,
        generator_vocab_size=args.generator_vocab_size,
    )
    vocab_size = game.gen._vocab_size  # noqa: SLF001
    eos_id = game.gen.eos_id

    emb_t = torch.tensor(emb, device=device)
    neuroglots: list[str] = []
    for lo in range(0, emb_t.size(0), args.batch_size):
        batch = emb_t[lo: lo + args.batch_size]
        docs = generate_documents(game, batch, vocab_size, eos_id)
        neuroglots.extend(join_turns(turns) for turns in docs)
        logger.info(f"  generated {lo + batch.size(0)}/{emb_t.size(0)}")

    # Free the game + decoder before loading Qwen (VRAM budget).
    del game
    torch.cuda.empty_cache()

    # ── 3. Qwen interpretations ──
    logger.info(f"Loading Qwen-Instruct: {args.qwen_model}")
    from transformers import AutoModelForCausalLM, AutoTokenizer
    qtok = AutoTokenizer.from_pretrained(args.qwen_model)
    qmodel = AutoModelForCausalLM.from_pretrained(
        args.qwen_model, torch_dtype=torch.bfloat16,
    ).to(device)
    qmodel.eval()

    interpretations: list[str] = []
    for i, ng in enumerate(neuroglots):
        interpretations.append(interpret(qmodel, qtok, ng))
        if (i + 1) % 25 == 0:
            logger.info(f"  interpreted {i + 1}/{len(neuroglots)}")

    del qmodel, qtok
    torch.cuda.empty_cache()

    # ── 4. SBERT-encode interpretations ──
    logger.info(f"SBERT-encoding interpretations: {args.sbert_model}")
    from sentence_transformers import SentenceTransformer
    sbert = SentenceTransformer(args.sbert_model, device=device)
    interp_emb = sbert.encode(
        interpretations, convert_to_numpy=True, show_progress_bar=False,
    )

    # ── 5. Pairwise distances + correlation ──
    d_emb = pairwise_cosine_distance(emb)
    d_int = pairwise_cosine_distance(interp_emb)

    sp_r, sp_p = stats.spearmanr(d_emb, d_int)
    pr_r, pr_p = stats.pearsonr(d_emb, d_int)

    logger.info("")
    logger.info("=" * 66)
    logger.info("Interpretation-topology eval — trained dialogue agent")
    logger.info("=" * 66)
    logger.info(f"  N samples      : {args.num_samples}")
    logger.info(f"  N pairs        : {len(d_emb)}")
    logger.info(f"  embedding↔interp  spearman ρ = {sp_r:+.4f} (p={sp_p:.2e})")
    logger.info(f"  embedding↔interp  pearson  r = {pr_r:+.4f} (p={pr_p:.2e})")
    logger.info("")
    logger.info(f"  interp mean len: {np.mean([len(s) for s in interpretations]):.1f}")
    logger.info(
        f"  unique interps : {len(set(interpretations))}/{len(interpretations)}",
    )

    # ── 6. Sample qualitative pairs for sanity ──
    logger.info("")
    logger.info("Sample neuroglot → interpretation pairs:")
    for i in rng.choice(len(neuroglots), size=min(5, len(neuroglots)), replace=False):
        logger.info(f"  NG : {neuroglots[i][:120]}...")
        logger.info(f"  INT: {interpretations[i]}")
        logger.info("")

    if args.output_samples:
        import json
        with open(args.output_samples, "w") as f:
            for idx, ng, ip in zip(indices, neuroglots, interpretations):
                f.write(json.dumps({"idx": idx, "neuroglot": ng, "interpretation": ip}) + "\n")
        logger.info(f"wrote samples to {args.output_samples}")


if __name__ == "__main__":
    main()

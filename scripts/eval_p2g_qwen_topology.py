"""p2g-Qwen topology eval (Option A).

For a given dialogue game checkpoint + p2g seq2seq, measure:
    spearman/pearson ρ ( embedding_cosine_distance, p2g_qwen_cosine_distance )

where:
  - embedding_cosine: distance between source embeddings (Qwen-latent store)
  - p2g_qwen_cosine: distance between Qwen-encoded p2g-renderings of the
    agent's IPA output for each source.

This is the measurement the dialogue game actually cares about for the
downstream LLM-interpretation use case: does the agent's p2g-rendered
English land in a similar part of Qwen's embedding space for similar
inputs?  Ideally ρ > 0.5.

Pure eval — no training-time integration, no gradient path, just a
scoring number.  Compare against the surface edit-distance topology from
``scripts/eval_dialogue_topology.py`` to see whether p2g-Qwen space
tracks the input topology better than raw IPA surface does.

Usage:

    poetry run python scripts/eval_p2g_qwen_topology.py \\
        --checkpoint data/dialogue_game_v8_qwen_contrastive/best.pt \\
        --decoder-path data/models/v8/vae_decoder.pt \\
        --spm-path data/phoneme_alphabet_multi.json \\
        --embedding-store data/embeddings_qwen_v2_clean \\
        --generator-name phoneme_vae --generator-vocab-size 30 \\
        --p2g-checkpoint data/models/p2g_v11/latest.pt \\
        --num-pairs 300
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
from eval_topology import cosine_distance  # noqa: F401  (kept for parity)
from source_retrieval_eval import generate_documents, load_dialogue_game
from test_qwen_p2g_spelling import load_p2g, p2g_spell
import test_qwen_p2g_spelling as tm

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger(__name__)


def pairwise_cosine_distance(X: np.ndarray) -> np.ndarray:
    """Upper-triangular pairwise cosine distances."""
    Xn = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)
    sim = Xn @ Xn.T
    i, j = np.triu_indices(X.shape[0], k=1)
    return 1.0 - sim[i, j]


def p2g_render(sentence: str, p2g, ipa_vocab, sp_vocab, cfg, device) -> str:
    """Word-by-word IPA → approximate English."""
    out: list[str] = []
    for w in sentence.split():
        if not w:
            continue
        out.append(p2g_spell(w, p2g, ipa_vocab, sp_vocab, cfg, device))
    return " ".join(out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--decoder-path", default="data/models/v7/vae_decoder.pt")
    ap.add_argument("--spm-path", default="data/models/v7/spm.model")
    ap.add_argument("--embedding-store", default="data/embeddings_qwen_v2_clean")
    ap.add_argument("--generator-name", default="multilingual_vae",
                    choices=["multilingual_vae", "phoneme_vae"])
    ap.add_argument("--generator-vocab-size", type=int, default=None)
    ap.add_argument("--p2g-checkpoint", default="data/models/p2g_v11/latest.pt")
    ap.add_argument("--qwen-model", default="Qwen/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--num-pairs", type=int, default=300)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    device = torch.device(args.device)
    rng = np.random.default_rng(args.seed)

    # ── 1. Sample diverse embeddings (one per cluster) ──
    from lfm.embeddings.store import EmbeddingStore
    store = EmbeddingStore(args.embedding_store)
    store.load()
    n = store.num_passages
    logger.info(f"Store: {n} passages, dim={store.embedding_dim}")

    cluster_ids = list(store._cluster_index.keys())
    N = min(args.num_pairs * 2, len(cluster_ids))
    chosen = rng.choice(cluster_ids, size=N, replace=False)
    indices = [
        int(store.sample_from_cluster(int(c), 1, rng=rng)[0]) for c in chosen
    ]
    emb = np.asarray(store._embeddings[indices], dtype=np.float32)
    del store

    # ── 2. Load dialogue game, generate IPA per sample ──
    logger.info(f"Loading dialogue game: {args.checkpoint}")
    tm.CKPT_PATH = args.p2g_checkpoint  # patch module-level default
    game = load_dialogue_game(
        args.checkpoint, args.decoder_path, args.spm_path,
        args.embedding_store, device,
        generator_name=args.generator_name,
        generator_vocab_size=args.generator_vocab_size,
    )
    vocab_size = game.gen._vocab_size  # noqa: SLF001
    eos_id = game.gen.eos_id

    emb_t = torch.tensor(emb, device=device)
    ipa_per_sample: list[str] = []
    for lo in range(0, emb_t.size(0), args.batch_size):
        batch = emb_t[lo: lo + args.batch_size]
        docs = generate_documents(game, batch, vocab_size, eos_id)
        # join multi-turn into one IPA string per sample
        ipa_per_sample.extend(" ".join(turns) for turns in docs)
        logger.info(f"  generated {lo + batch.size(0)}/{emb_t.size(0)}")
    del game
    torch.cuda.empty_cache()

    # ── 3. Load p2g, render each sample to approximate English ──
    logger.info(f"Loading p2g: {args.p2g_checkpoint}")
    p2g, ipa_vocab, sp_vocab, p2g_cfg = load_p2g(device)
    english_per_sample: list[str] = [
        p2g_render(s, p2g, ipa_vocab, sp_vocab, p2g_cfg, device)
        for s in ipa_per_sample
    ]
    del p2g
    torch.cuda.empty_cache()

    # ── 4. Encode English via Qwen (mean-pool last hidden) ──
    logger.info(f"Loading Qwen for encoding: {args.qwen_model}")
    from transformers import AutoModelForCausalLM, AutoTokenizer
    qtok = AutoTokenizer.from_pretrained(args.qwen_model)
    qmodel = AutoModelForCausalLM.from_pretrained(
        args.qwen_model, torch_dtype=torch.bfloat16,
    ).to(device).eval()

    p2g_embs: list[np.ndarray] = []
    for i, text in enumerate(english_per_sample):
        inputs = qtok(text, return_tensors="pt", truncation=True, max_length=512).to(device)
        with torch.no_grad():
            out = qmodel(**inputs, output_hidden_states=True)
        # Mean-pool last hidden layer over non-pad positions
        h = out.hidden_states[-1][0]  # (S, D)
        mask = inputs["attention_mask"][0].bool()
        pooled = h[mask].mean(dim=0).float().cpu().numpy()
        p2g_embs.append(pooled)
        if (i + 1) % 50 == 0:
            logger.info(f"  Qwen-encoded {i + 1}/{len(english_per_sample)}")
    p2g_embs_arr = np.stack(p2g_embs, axis=0)
    del qmodel, qtok
    torch.cuda.empty_cache()

    # ── 5. Pairwise distances + correlation ──
    d_input = pairwise_cosine_distance(emb)
    d_p2g_qwen = pairwise_cosine_distance(p2g_embs_arr)
    sp_r, sp_p = stats.spearmanr(d_input, d_p2g_qwen)
    pr_r, pr_p = stats.pearsonr(d_input, d_p2g_qwen)

    logger.info("")
    logger.info("=" * 72)
    logger.info("p2g-Qwen topology (dialogue-checkpoint × p2g-render × Qwen-encode)")
    logger.info("=" * 72)
    logger.info(f"  N samples      : {len(indices)}")
    logger.info(f"  N pairs        : {len(d_input)}")
    logger.info(f"  input↔p2g_qwen  spearman ρ = {sp_r:+.4f} (p={sp_p:.2e})")
    logger.info(f"  input↔p2g_qwen  pearson  r = {pr_r:+.4f} (p={pr_p:.2e})")
    logger.info("")
    logger.info(
        f"  mean IPA len    : {np.mean([len(s) for s in ipa_per_sample]):.1f}",
    )
    logger.info(
        f"  mean English len: {np.mean([len(s) for s in english_per_sample]):.1f}",
    )
    logger.info("")
    logger.info("Sample (idx 0):")
    logger.info(f"  IPA  : {ipa_per_sample[0][:180]}...")
    logger.info(f"  ENG  : {english_per_sample[0][:180]}...")


if __name__ == "__main__":
    main()

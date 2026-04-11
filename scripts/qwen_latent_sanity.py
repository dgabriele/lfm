"""Sanity checks for the 'Qwen hidden state as perception target' proposal.

Two cheap empirical tests before committing to retrain the dialogue
game on Qwen-native target embeddings:

CHECK 1 — Cluster structure in Qwen's latent space
  Take ~1000 Leipzig sentences, encode each with Qwen 0.5B, extract
  the last-token residual at the final layer, and run k-means with
  k=20.  For each cluster, show a handful of member sentences.  If
  the clusters look semantically coherent to a human eye, Qwen's
  latent space has usable discriminative structure.  If the clusters
  look random, Qwen's hidden states may not be a good training
  target for the discrimination game.

CHECK 2 — Input coupling
  Generate Neuroglot documents from the baseline agent for ~20
  diverse source embeddings.  Feed each Neuroglot back into Qwen.
  Does Qwen's hidden state vary meaningfully across different
  Neuroglot inputs, or does it collapse to a near-constant "I don't
  understand this" attractor?  If Qwen's resulting hidden states are
  all mutually near-identical, then the geometric reconstruction
  objective is pinned to a constant target and training will fail.
  If they vary, the mechanism is operational and the approach is
  worth pursuing.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys

import numpy as np
import sentencepiece as spm_lib
import torch
import torch.nn.functional as F

logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stderr)
logger = logging.getLogger(__name__)


def _qwen_encode(
    model, tokenizer, texts: list[str], device,
    batch_size: int = 16,
) -> torch.Tensor:
    """Extract last-token final-layer hidden state for each text."""
    all_embs = []
    for start in range(0, len(texts), batch_size):
        batch = texts[start:start + batch_size]
        enc = tokenizer(
            batch, return_tensors="pt", padding=True, truncation=True,
            max_length=256,
        ).to(device)
        with torch.no_grad():
            out = model(
                **enc, output_hidden_states=True, use_cache=False,
            )
        last_hidden = out.hidden_states[-1]  # (B, T, D)
        # Last non-padding token per sequence
        mask = enc["attention_mask"]
        lengths = mask.sum(dim=1) - 1  # indices of last real token
        pooled = last_hidden[torch.arange(len(batch)), lengths]
        all_embs.append(F.normalize(pooled.float(), dim=-1).cpu())
    return torch.cat(all_embs, dim=0)  # (N, D)


def check_1_cluster_structure(model, tokenizer, device, args):
    print()
    print("=" * 88)
    print("CHECK 1 — Cluster structure of Qwen latent space")
    print("=" * 88)

    # Read the Leipzig passages
    texts: list[str] = []
    with open(f"{args.embedding_store}/passages.jsonl") as f:
        for line in f:
            texts.append(json.loads(line)["text"])
            if len(texts) >= args.n_cluster:
                break
    logger.info("Loaded %d Leipzig sentences", len(texts))

    logger.info("Encoding via Qwen (last layer, last token)...")
    embs = _qwen_encode(model, tokenizer, texts, device)  # (N, D)
    logger.info("Embeddings shape: %s, norm mean: %.3f",
                tuple(embs.shape), embs.norm(dim=-1).mean().item())

    # Pairwise cosine summary
    if embs.size(0) <= 1000:
        sim = embs @ embs.t()
        off_diag = sim[~torch.eye(embs.size(0), dtype=torch.bool)]
        print(f"pairwise cosine (off-diag): mean={off_diag.mean():.4f} "
              f"std={off_diag.std():.4f}  min={off_diag.min():.4f}  "
              f"max={off_diag.max():.4f}")
    # If mean cosine ≈ 1 and std tiny, the space is collapsed (bad)
    # If std is reasonable (say > 0.05), there's real separation

    # K-means and print a few samples per cluster
    from sklearn.cluster import KMeans
    k = args.k_clusters
    logger.info("Running k-means with k=%d...", k)
    km = KMeans(n_clusters=k, random_state=args.seed, n_init=10)
    labels = km.fit_predict(embs.numpy())
    for c in range(k):
        members = [i for i, l in enumerate(labels) if l == c]
        if not members:
            continue
        show = members[: args.samples_per_cluster]
        print(f"\n[cluster {c}] size={len(members)}")
        for i in show:
            t = texts[i].strip().replace("\n", " ")
            print(f"  - {t[:120]}")


def _load_dialogue_game(checkpoint_path, decoder_path, spm_path, embedding_store, device):
    from lfm.agents.games.dialogue import DialogueGame, DialogueGameConfig
    from lfm.faculty.model import LanguageFaculty

    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    game_cfg = DialogueGameConfig(
        decoder_path=decoder_path, spm_path=spm_path,
        embedding_store_dir=embedding_store,
        max_phrases=ckpt.get("max_phrases", 3),
        num_turns=ckpt.get("num_turns", 4),
        device=str(device),
        llm_loss_weight=0.0,
    )
    faculty = LanguageFaculty(game_cfg.build_faculty_config()).to(device)
    game = DialogueGame(game_cfg, faculty).to(device)
    game.load_checkpoint_state(ckpt)
    game.eval()
    return game


def _generate_documents(game, embeddings, sp, vocab_size, eos_id):
    from lfm.agents.decode import rerun_decoder_multiphrase_no_grad
    from lfm.translator.romanize import romanize_iso

    batch = embeddings.size(0)
    targets = embeddings.unsqueeze(1)
    context_summaries: list = []
    docs: list[list[str]] = [[] for _ in range(batch)]
    for turn_idx in range(game.config.num_turns):
        turn_emb = game.turn_embeddings[turn_idx]
        context = torch.stack(context_summaries, dim=1) if context_summaries else None
        conditioning = game.context_transformer(targets, turn_emb, context, target_mask=None)
        z_seq, z_weights, _ = game.z_gen(conditioning)
        tokens, gen_mask, bounds = game.phrase_decoder.decode(z_seq, z_weights)
        hidden = rerun_decoder_multiphrase_no_grad(
            game.gen, z_seq, z_weights, tokens, gen_mask, bounds,
        )
        trimmed_mask = gen_mask[:, :hidden.size(1)]
        summary = game._summarize_turn(hidden, trimmed_mask)
        context_summaries.append(summary)
        for j in range(batch):
            ids = [
                t.item() for t, m in zip(tokens[j], gen_mask[j])
                if m and t.item() != eos_id and t.item() < vocab_size
            ]
            ipa = sp.decode(ids).strip()
            rom = romanize_iso(ipa).strip() if ipa else ""
            if rom:
                docs[j].append(rom)
        del hidden, tokens, gen_mask, bounds, z_seq, z_weights
    return [" ".join(t.rstrip(".") + "." for t in d) for d in docs]


def check_2_input_coupling(model, tokenizer, device, args):
    print()
    print("=" * 88)
    print("CHECK 2 — Input coupling of Qwen hidden state to Neuroglot")
    print("=" * 88)

    # Sample diverse source embeddings
    from lfm.embeddings.store import EmbeddingStore
    rng = np.random.default_rng(args.seed)
    store = EmbeddingStore(args.embedding_store)
    store.load()
    cluster_pool = list(store._cluster_index.keys())
    chosen = rng.choice(cluster_pool, size=args.n_coupling, replace=False)
    indices = [
        int(store.sample_from_cluster(int(c), 1, rng=rng)[0]) for c in chosen
    ]

    # Pull sources for reporting
    target = set(indices)
    passages = {}
    with open(f"{args.embedding_store}/passages.jsonl") as f:
        for i, line in enumerate(f):
            if i in target:
                passages[i] = json.loads(line)["text"]
            if len(passages) == len(target):
                break
    sources = [passages[i] for i in indices]

    embs = torch.tensor(
        store._embeddings[indices], dtype=torch.float32, device=device,
    )
    del store

    # Encode source sentences via Qwen as a reference point
    logger.info("Encoding %d source sentences via Qwen...", len(sources))
    source_qemb = _qwen_encode(model, tokenizer, sources, device)  # (N, D)

    # Generate Neuroglot documents from the baseline agent
    logger.info("Loading baseline dialogue game...")
    game = _load_dialogue_game(
        args.checkpoint, args.decoder_path, args.spm_path,
        args.embedding_store, device,
    )
    sp = spm_lib.SentencePieceProcessor()
    sp.Load(args.spm_path)
    vocab_size = sp.GetPieceSize()
    eos_id = game.gen.eos_id
    logger.info("Generating %d Neuroglot documents...", embs.size(0))
    with torch.no_grad():
        neuroglot_docs = _generate_documents(game, embs, sp, vocab_size, eos_id)
    del game
    torch.cuda.empty_cache()

    logger.info("Encoding %d Neuroglot documents via Qwen...", len(neuroglot_docs))
    ng_qemb = _qwen_encode(model, tokenizer, neuroglot_docs, device)  # (N, D)

    # Diagnostic statistics
    ng_pair_sim = ng_qemb @ ng_qemb.t()
    src_pair_sim = source_qemb @ source_qemb.t()
    cross_sim = ng_qemb @ source_qemb.t()  # (N, N)

    def _off_diag_stats(mat: torch.Tensor) -> tuple[float, float]:
        n = mat.size(0)
        off = mat[~torch.eye(n, dtype=torch.bool)]
        return off.mean().item(), off.std().item()

    ng_mean, ng_std = _off_diag_stats(ng_pair_sim)
    src_mean, src_std = _off_diag_stats(src_pair_sim)

    print()
    print("Pairwise cosine (off-diagonal):")
    print(f"  Neuroglot→Qwen states (N={ng_qemb.size(0)}):  "
          f"mean={ng_mean:.4f}  std={ng_std:.4f}")
    print(f"  Source→Qwen states     (N={source_qemb.size(0)}):  "
          f"mean={src_mean:.4f}  std={src_std:.4f}")
    print()
    print("Interpretation guide:")
    print("  - If Neuroglot std << Source std, Qwen collapses all Neuroglot")
    print("    inputs to ~same point (bad: pinned target).")
    print("  - If Neuroglot std ≈ Source std, Qwen differentiates Neuroglot")
    print("    inputs as strongly as natural English (good).")
    print()

    # Does the Neuroglot for source i end up closer in Qwen space to
    # source i than to other sources?  This is the direct test of
    # whether training on geometric reconstruction has any hope.
    diag = cross_sim.diag()
    off_diag = (cross_sim.sum() - diag.sum()) / (cross_sim.numel() - cross_sim.size(0))
    print("Cross: Neuroglot_i vs source_j in Qwen's space")
    print(f"  mean cos(ng_i, source_i):  {diag.mean().item():.4f}")
    print(f"  mean cos(ng_i, source_j):  {off_diag.item():.4f}")
    print(f"  gap (correct − random):    {(diag.mean() - off_diag).item():+.4f}")

    # Top-1 retrieval accuracy: for each Neuroglot, rank the sources
    n = cross_sim.size(0)
    ranks = []
    for i in range(n):
        row = cross_sim[i]
        r = int((row > row[i]).sum().item()) + 1
        ranks.append(r)
    ranks_arr = np.array(ranks)
    recall1 = float((ranks_arr == 1).sum()) / n
    print(f"  Neuroglot→source recall@1: {recall1:.3f}  (chance={1.0/n:.3f})")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="data/dialogue_game_v7/best.pt",
                        help="Baseline dialogue checkpoint for Check 2")
    parser.add_argument("--decoder-path", default="data/models/v7/vae_decoder.pt")
    parser.add_argument("--spm-path", default="data/models/v7/spm.model")
    parser.add_argument("--embedding-store", default="data/embeddings")
    parser.add_argument("--qwen-model", default="Qwen/Qwen2.5-0.5B")
    parser.add_argument("--n-cluster", type=int, default=1000,
                        help="Number of Leipzig sentences for Check 1")
    parser.add_argument("--k-clusters", type=int, default=20)
    parser.add_argument("--samples-per-cluster", type=int, default=5)
    parser.add_argument("--n-coupling", type=int, default=20,
                        help="Number of samples for Check 2")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--skip-check1", action="store_true")
    parser.add_argument("--skip-check2", action="store_true")
    args = parser.parse_args()

    device = torch.device(args.device)

    logger.info("Loading Qwen %s (bf16)...", args.qwen_model)
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.qwen_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.qwen_model, torch_dtype=torch.bfloat16,
    ).to(device)
    model.eval()

    if not args.skip_check1:
        check_1_cluster_structure(model, tokenizer, device, args)

    if not args.skip_check2:
        check_2_input_coupling(model, tokenizer, device, args)


if __name__ == "__main__":
    main()

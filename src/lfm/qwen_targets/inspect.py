"""Sanity inspection utilities for built Qwen-latent stores.

Two cheap checks, reused across CLI and scripting:

* :func:`inspect_cluster_structure` — cluster a sample of text through
  the extractor, run k-means, and print example members of each
  cluster.  Answers "does this latent space carry usable structure
  from this corpus at all?"

* :func:`inspect_neuroglot_coupling` — generate Neuroglot documents
  from a trained dialogue-game checkpoint, feed them back through the
  LLM, and measure whether the resulting hidden states differentiate
  (relative to natural English source variance).  Answers "does the
  LLM's internal representation of Neuroglot vary per input, or does
  it collapse to a single attractor?"
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import torch

from lfm.qwen_targets.config import ExtractorConfig
from lfm.qwen_targets.corpora import CorpusSource, JSONLCorpusSource
from lfm.qwen_targets.extractor import HiddenStateExtractor

logger = logging.getLogger(__name__)


def inspect_cluster_structure(
    source: CorpusSource,
    extractor_config: ExtractorConfig,
    k: int = 20,
    samples_per_cluster: int = 5,
    max_texts: int = 1000,
    seed: int = 42,
    device: str = "cuda",
) -> dict:
    """Encode up to ``max_texts`` from a corpus, cluster, and report.

    Prints one sample per cluster to stdout and returns a dict of
    summary statistics (cluster sizes, pairwise cosine mean/std).
    """
    from sklearn.cluster import KMeans

    device_t = torch.device(device if torch.cuda.is_available() else "cpu")
    extractor = HiddenStateExtractor(extractor_config, device=device_t)

    texts: list[str] = []
    for record in source:
        texts.append(record.text)
        if len(texts) >= max_texts:
            break

    logger.info("Encoding %d texts for cluster inspection...", len(texts))
    all_embs: list[torch.Tensor] = []
    for batch in extractor.encode_stream(texts):
        all_embs.append(batch)
    embs = torch.cat(all_embs, dim=0)

    print()
    print("=" * 88)
    print("CHECK 1 — Cluster structure of LLM latent space")
    print("=" * 88)

    # Pairwise cosine summary
    sim = embs @ embs.t()
    eye = torch.eye(embs.size(0), dtype=torch.bool)
    off = sim[~eye]
    print(
        f"pairwise cosine (off-diag): mean={off.mean():.4f} "
        f"std={off.std():.4f} min={off.min():.4f} max={off.max():.4f}"
    )
    print()

    km = KMeans(n_clusters=k, random_state=seed, n_init=10)
    labels = km.fit_predict(embs.numpy())
    for c in range(k):
        members = [i for i, lb in enumerate(labels) if lb == c]
        if not members:
            continue
        show = members[:samples_per_cluster]
        print(f"[cluster {c}] size={len(members)}")
        for i in show:
            t = texts[i].strip().replace("\n", " ")
            print(f"  - {t[:120]}")
        print()

    # Release extractor before returning
    del extractor
    torch.cuda.empty_cache()

    return {
        "n_texts": int(embs.size(0)),
        "dim": int(embs.size(1)),
        "cosine_mean": float(off.mean()),
        "cosine_std": float(off.std()),
        "cosine_min": float(off.min()),
        "cosine_max": float(off.max()),
    }


def inspect_neuroglot_coupling(
    checkpoint_path: str,
    extractor_config: ExtractorConfig,
    decoder_path: str,
    spm_path: str,
    embedding_store_dir: str,
    n_coupling: int = 20,
    seed: int = 42,
    device: str = "cuda",
) -> dict:
    """Probe whether the LLM's hidden state couples to Neuroglot input.

    Generates ``n_coupling`` Neuroglot documents from the given
    dialogue-game checkpoint, feeds each through the LLM, and
    compares to the natural source-sentence hidden states.
    """
    import sentencepiece as spm_lib

    from lfm.agents.games.dialogue import DialogueGame, DialogueGameConfig
    from lfm.faculty.model import LanguageFaculty
    from lfm.embeddings.store import EmbeddingStore

    device_t = torch.device(device if torch.cuda.is_available() else "cpu")
    rng = np.random.default_rng(seed)

    store = EmbeddingStore(embedding_store_dir)
    store.load()
    cluster_pool = list(store._cluster_index.keys())
    chosen = rng.choice(cluster_pool, size=n_coupling, replace=False)
    indices = [int(store.sample_from_cluster(int(c), 1, rng=rng)[0]) for c in chosen]

    target_set = set(indices)
    passages: dict[int, str] = {}
    with open(f"{embedding_store_dir}/passages.jsonl", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i in target_set:
                passages[i] = json.loads(line)["text"]
            if len(passages) == len(target_set):
                break
    sources = [passages[i] for i in indices]
    embs = torch.tensor(
        store._embeddings[indices], dtype=torch.float32, device=device_t,
    )
    del store

    # Source encoding
    extractor = HiddenStateExtractor(extractor_config, device=device_t)
    source_qemb = torch.cat(list(extractor.encode_stream(sources)), dim=0)
    logger.info("Encoded %d source sentences via %s", len(sources), extractor_config.model_name)

    # Load dialogue game for Neuroglot generation
    ckpt = torch.load(checkpoint_path, map_location=device_t, weights_only=False)

    dim_kwargs: dict = {}
    try:
        meta = EmbeddingStore.read_metadata(embedding_store_dir)
        if "embedding_dim" in meta:
            dim_kwargs["embedding_dim"] = int(meta["embedding_dim"])
    except FileNotFoundError:
        pass

    game_cfg = DialogueGameConfig(
        decoder_path=decoder_path,
        spm_path=spm_path,
        embedding_store_dir=embedding_store_dir,
        max_phrases=ckpt.get("max_phrases", 3),
        num_turns=ckpt.get("num_turns", 4),
        device=str(device_t),
        llm_loss_weight=0.0,
        **dim_kwargs,
    )
    faculty = LanguageFaculty(game_cfg.build_faculty_config()).to(device_t)
    game = DialogueGame(game_cfg, faculty).to(device_t)
    game.load_checkpoint_state(ckpt)
    game.eval()

    # Generate Neuroglot documents for each source embedding
    from lfm.agents.decode import rerun_decoder_multiphrase_no_grad
    from lfm.translator.romanize import romanize_iso

    sp = spm_lib.SentencePieceProcessor()
    sp.Load(spm_path)
    vocab_size = sp.GetPieceSize()
    eos_id = game.gen.eos_id

    targets_batch = embs.unsqueeze(1)
    context_summaries: list = []
    documents: list[list[str]] = [[] for _ in range(embs.size(0))]

    with torch.no_grad():
        for turn_idx in range(game.config.num_turns):
            turn_emb = game.turn_embeddings[turn_idx]
            context = (
                torch.stack(context_summaries, dim=1) if context_summaries else None
            )
            conditioning = game.context_transformer(
                targets_batch, turn_emb, context, target_mask=None,
            )
            z_seq, z_weights, _ = game.z_gen(conditioning)
            tokens, gen_mask, bounds = game.phrase_decoder.decode(z_seq, z_weights)
            hidden = rerun_decoder_multiphrase_no_grad(
                game.gen, z_seq, z_weights, tokens, gen_mask, bounds,
            )
            trimmed_mask = gen_mask[:, :hidden.size(1)]
            summary = game._summarize_turn(hidden, trimmed_mask)
            context_summaries.append(summary)

            for j in range(embs.size(0)):
                ids = [
                    t.item() for t, m in zip(tokens[j], gen_mask[j])
                    if m and t.item() != eos_id and t.item() < vocab_size
                ]
                ipa = sp.decode(ids).strip()
                rom = romanize_iso(ipa).strip() if ipa else ""
                if rom:
                    documents[j].append(rom)
            del hidden, tokens, gen_mask, bounds, z_seq, z_weights

    del game
    torch.cuda.empty_cache()

    neuroglot_strs = [" ".join(t.rstrip(".") + "." for t in d) for d in documents]
    ng_qemb = torch.cat(list(extractor.encode_stream(neuroglot_strs)), dim=0)

    # Diagnostic statistics
    def _off_diag_stats(mat: torch.Tensor) -> tuple[float, float]:
        n = mat.size(0)
        off = mat[~torch.eye(n, dtype=torch.bool)]
        return float(off.mean()), float(off.std())

    ng_pair_sim = ng_qemb @ ng_qemb.t()
    src_pair_sim = source_qemb @ source_qemb.t()
    cross_sim = ng_qemb @ source_qemb.t()

    ng_mean, ng_std = _off_diag_stats(ng_pair_sim)
    src_mean, src_std = _off_diag_stats(src_pair_sim)

    print()
    print("=" * 88)
    print("CHECK 2 — Input coupling of LLM hidden state to Neuroglot")
    print("=" * 88)
    print(
        f"Neuroglot→LLM states (N={ng_qemb.size(0)}): mean={ng_mean:.4f} std={ng_std:.4f}"
    )
    print(
        f"Source→LLM states     (N={source_qemb.size(0)}): mean={src_mean:.4f} std={src_std:.4f}"
    )

    diag = cross_sim.diag()
    off_diag = float(
        (cross_sim.sum() - diag.sum()) / (cross_sim.numel() - cross_sim.size(0))
    )
    diag_mean = float(diag.mean())
    print()
    print("Cross: Neuroglot_i vs source_j in LLM space")
    print(f"  mean cos(ng_i, source_i): {diag_mean:.4f}")
    print(f"  mean cos(ng_i, source_j): {off_diag:.4f}")
    print(f"  gap (correct − random):   {diag_mean - off_diag:+.4f}")

    ranks: list[int] = []
    for i in range(cross_sim.size(0)):
        row = cross_sim[i]
        r = int((row > row[i]).sum().item()) + 1
        ranks.append(r)
    recall1 = float(sum(1 for r in ranks if r == 1)) / len(ranks)
    print(f"  Neuroglot→source recall@1: {recall1:.3f} (chance={1.0/len(ranks):.3f})")

    del extractor
    torch.cuda.empty_cache()

    return {
        "neuroglot_cos_mean": ng_mean,
        "neuroglot_cos_std": ng_std,
        "source_cos_mean": src_mean,
        "source_cos_std": src_std,
        "cross_diag_mean": diag_mean,
        "cross_off_diag_mean": off_diag,
        "recall_at_1": recall1,
    }

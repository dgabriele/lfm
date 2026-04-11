"""Test whether Qwen's interpretations of a pressured agent's output
cluster consistently within semantic neighborhoods of the input
embeddings.

The LLM-pressure hypothesis, refined: the agent's output is never a
literal translation of the source sentence.  Instead, its surface
statistical pattern shifts into a region of Qwen's latent space that
Qwen associates with a particular *semantic style*.  Qwen's
interpretation of any individual document is an abstract guess about
what kind of thing the document might describe — not a faithful
decoding.

For the hypothesis to hold empirically, we need:

1. **Cluster consistency.** Multiple embeddings from the same Leipzig
   cluster (conceptually similar source sentences) should produce
   Neuroglot documents whose Qwen interpretations land in a shared
   semantic neighborhood.  If Qwen gives "political rivalry" for one
   and "medieval warfare" for another within the same cluster, the
   signal is being mapped consistently (both are about conflict
   between parties), even if neither is literally correct.
2. **Cross-cluster separation.** Documents from *different* Leipzig
   clusters should produce Qwen interpretations from *different*
   semantic neighborhoods.  Otherwise the agent is producing a
   constant stylistic fingerprint for everything.

This script measures both by:
  - Sampling K Leipzig clusters
  - For each cluster, drawing N target embeddings
  - Generating a Neuroglot document per embedding
  - Prompting Qwen to interpret each
  - Computing within-cluster vs across-cluster interpretation
    embedding similarity (using Qwen's own hidden states as
    sentence encoder)

Within-cluster cosine similarity minus across-cluster similarity is
the signal.  Positive = clustering works, zero = no signal,
negative = interpretations are chaotic.
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


def load_dialogue_game(checkpoint_path, decoder_path, spm_path, embedding_store, device):
    from lfm.agents.games.dialogue import DialogueGame, DialogueGameConfig
    from lfm.faculty.model import LanguageFaculty

    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    game_cfg = DialogueGameConfig(
        decoder_path=decoder_path,
        spm_path=spm_path,
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


def generate_documents(game, embeddings, sp, vocab_size, eos_id):
    from lfm.agents.decode import rerun_decoder_multiphrase_no_grad
    from lfm.translator.romanize import romanize_iso

    batch = embeddings.size(0)
    num_turns = game.config.num_turns
    targets = embeddings.unsqueeze(1)
    context_summaries: list = []
    documents: list[list[str]] = [[] for _ in range(batch)]

    for turn_idx in range(num_turns):
        turn_emb = game.turn_embeddings[turn_idx]
        context = (
            torch.stack(context_summaries, dim=1) if context_summaries else None
        )
        conditioning = game.context_transformer(
            targets, turn_emb, context, target_mask=None,
        )
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
                documents[j].append(rom)

        del hidden, tokens, gen_mask, bounds, z_seq, z_weights
    return documents


def join_turns(turns: list[str]) -> str:
    return " ".join(t.rstrip(".") + "." for t in turns)


def interpret(model, tokenizer, document: str) -> str:
    messages = [
        {
            "role": "system",
            "content": (
                "You will be given a passage written in an artificial "
                "language.  Read the passage and describe, in one or "
                "two short English sentences, the general theme or "
                "kind of scenario it seems to describe.  Do not try "
                "to translate word-for-word; give the abstract gist."
            ),
        },
        {
            "role": "user",
            "content": f"{document}\n\nWhat is the general theme?",
        },
    ]
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True,
    )
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=80,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
            repetition_penalty=1.1,
            pad_token_id=tokenizer.pad_token_id,
        )
    generated = out[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(generated, skip_special_tokens=True).strip()


def encode_text(model, tokenizer, text: str, device) -> torch.Tensor:
    """Use the last hidden state (mean-pooled) as a sentence embedding."""
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=200).to(device)
    with torch.no_grad():
        out = model(**inputs, output_hidden_states=True)
    h = out.hidden_states[-1]  # (1, T, D)
    mask = inputs["attention_mask"].unsqueeze(-1).float()
    pooled = (h * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
    return F.normalize(pooled.squeeze(0), dim=-1).float().cpu()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True,
                        help="Dialogue game checkpoint to test")
    parser.add_argument("--decoder-path", default="data/models/v7/vae_decoder.pt")
    parser.add_argument("--spm-path", default="data/models/v7/spm.model")
    parser.add_argument("--embedding-store", default="data/embeddings")
    parser.add_argument("--qwen-model", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--num-clusters", type=int, default=6,
                        help="How many Leipzig clusters to sample")
    parser.add_argument("--samples-per-cluster", type=int, default=4,
                        help="Embeddings sampled per cluster")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    device = torch.device(args.device)
    rng = np.random.default_rng(args.seed)

    logger.info("Loading embedding store...")
    from lfm.embeddings.store import EmbeddingStore
    store = EmbeddingStore(args.embedding_store)
    store.load()

    chosen_clusters = rng.choice(
        list(store._cluster_index.keys()),
        size=args.num_clusters,
        replace=False,
    )
    all_indices = []
    cluster_of_index: list[int] = []
    for cid in chosen_clusters:
        idxs = store.sample_from_cluster(int(cid), args.samples_per_cluster, rng=rng)
        all_indices.extend(int(i) for i in idxs)
        cluster_of_index.extend([int(cid)] * len(idxs))

    passages = {}
    target_set = set(all_indices)
    with open(f"{args.embedding_store}/passages.jsonl") as f:
        for i, line in enumerate(f):
            if i in target_set:
                passages[i] = json.loads(line)["text"]
            if len(passages) == len(target_set):
                break

    embeddings = torch.tensor(
        store._embeddings[all_indices], dtype=torch.float32, device=device,
    )
    del store

    logger.info("Loading dialogue game from %s", args.checkpoint)
    game = load_dialogue_game(
        args.checkpoint, args.decoder_path, args.spm_path,
        args.embedding_store, device,
    )
    sp = spm_lib.SentencePieceProcessor()
    sp.Load(args.spm_path)
    vocab_size = sp.GetPieceSize()
    eos_id = game.gen.eos_id

    logger.info("Generating %d Neuroglot documents...", len(all_indices))
    with torch.no_grad():
        docs_per_sample = generate_documents(game, embeddings, sp, vocab_size, eos_id)
    docs = [join_turns(d) for d in docs_per_sample]

    del game
    torch.cuda.empty_cache()

    logger.info("Loading Qwen %s for interpretation and embedding...", args.qwen_model)
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.qwen_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.qwen_model, torch_dtype=torch.bfloat16,
    ).to(device)
    model.eval()

    def _pairwise_signal(emb_tensor: torch.Tensor, label: str) -> tuple[float, float, float]:
        sim = emb_tensor @ emb_tensor.t()
        cluster_ids = np.array(cluster_of_index)
        within = []
        across = []
        n = sim.size(0)
        for i in range(n):
            for j in range(i + 1, n):
                s = sim[i, j].item()
                if cluster_ids[i] == cluster_ids[j]:
                    within.append(s)
                else:
                    across.append(s)
        w = float(np.mean(within)) if within else 0.0
        a = float(np.mean(across)) if across else 0.0
        print(f"  {label:40s}  within={w:.4f}  across={a:.4f}  Δ={w - a:+.4f}")
        return w, a, w - a

    # Direct encoding of the raw agent output — bypasses the noisy
    # interpretation/generation step and asks "does Qwen internally
    # distinguish the clusters when reading the Neuroglot directly."
    direct_embs: list[torch.Tensor] = []
    for doc in docs:
        direct_embs.append(encode_text(model, tokenizer, doc, device))

    # Generate interpretations and encode those (noisier, biased by
    # Qwen's generation attractors).
    interpretations: list[str] = []
    interp_embs: list[torch.Tensor] = []
    for i, (doc, idx, cid) in enumerate(zip(docs, all_indices, cluster_of_index)):
        interp = interpret(model, tokenizer, doc)
        interpretations.append(interp)
        interp_embs.append(encode_text(model, tokenizer, interp, device))
        logger.info("[cluster=%d idx=%d] %s", cid, idx, interp[:120].replace("\n", " "))

    print()
    print("=" * 88)
    print("CLUSTERING SIGNAL")
    print("=" * 88)
    print("signal = within_cluster_cosine − across_cluster_cosine")
    print("positive → Qwen distinguishes clusters; near-zero → no signal")
    print()
    _pairwise_signal(
        torch.stack(direct_embs),
        "RAW Neuroglot (Qwen hidden state, direct)",
    )
    _pairwise_signal(
        torch.stack(interp_embs),
        "INTERPRETED (Qwen hidden state of generated text)",
    )
    print()
    print("=" * 88)
    print("PER-CLUSTER INTERPRETATIONS")
    print("=" * 88)
    for cid in np.unique(cluster_ids):
        members = np.where(cluster_ids == cid)[0]
        print(f"\n[cluster {cid}]")
        for m in members:
            idx = all_indices[m]
            src = passages.get(idx, "")[:120]
            interp = interpretations[m][:160].replace("\n", " ")
            print(f"  src: {src}")
            print(f"  →    {interp}")


if __name__ == "__main__":
    main()

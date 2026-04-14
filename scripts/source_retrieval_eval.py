"""Source-retrieval evaluation for LLM-pressured dialogue checkpoints.

The right question for the LLM-pressure hypothesis is not "do
interpretations cluster" but "does Qwen's interpretation of a
Neuroglot sample retrieve the *correct* source sentence from a
pool of candidates, above chance?"

Protocol:
  1. Sample N embeddings from the Leipzig store.
  2. Look up each embedding's original English source sentence.
  3. Generate a Neuroglot document per embedding from the agent
     checkpoint under test.
  4. For each Neuroglot, prompt Qwen to interpret it in English
     (abstract gist, not word-for-word).
  5. Encode every source sentence and every interpretation with a
     frozen sentence-transformer (independent of Qwen — breaks the
     "Qwen evaluating Qwen" confound).
  6. For each sample i, compute cosine(interp_i, source_j) for all j.
     Rank source_i among all candidates.  If interpretations are
     thematically faithful to sources, source_i should rank near the
     top for its own interpretation.

Metrics:
  - Recall@1: fraction of samples where the correct source is the top
    retrieval.  Chance = 1/N.
  - Mean Reciprocal Rank (MRR): 1/N * sum(1/rank_i).  Chance ≈ 1/ln(N).
  - Mean cosine gap: cosine(interp_i, source_i) minus cosine(interp_i,
    random source_j).  Positive means the correct source is on average
    closer than a random source.

With N=20 samples, chance recall@1 = 5%.  Anything meaningfully above
5% is signal the clustering test missed.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys

import numpy as np
import torch

logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stderr)
logger = logging.getLogger(__name__)


def load_dialogue_game(
    checkpoint_path,
    decoder_path,
    spm_path,
    embedding_store,
    device,
    generator_name="multilingual_vae",
    generator_vocab_size=None,
):
    from lfm.agents.games.dialogue import DialogueGame, DialogueGameConfig
    from lfm.embeddings.store import EmbeddingStore
    from lfm.faculty.model import LanguageFaculty

    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    # Auto-detect embedding_dim from the store (Qwen=896, SBERT=384, etc.)
    probe = EmbeddingStore(embedding_store)
    probe.load()
    embedding_dim = probe.embedding_dim
    del probe
    game_cfg = DialogueGameConfig(
        decoder_path=decoder_path,
        spm_path=spm_path,
        embedding_store_dir=embedding_store,
        embedding_dim=embedding_dim,
        max_phrases=ckpt.get("max_phrases", 3),
        num_turns=ckpt.get("num_turns", 4),
        device=str(device),
        llm_loss_weight=0.0,
        generator_name=generator_name,
        generator_vocab_size=generator_vocab_size,
    )
    faculty = LanguageFaculty(game_cfg.build_faculty_config()).to(device)
    game = DialogueGame(game_cfg, faculty).to(device)
    game.load_checkpoint_state(ckpt)
    game.eval()
    return game


def generate_documents(game, embeddings, vocab_size, eos_id):
    from lfm.agents.decode import rerun_decoder_multiphrase_no_grad

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

        # VAE-agnostic surface rendering — handles v7 (IPA → romanized)
        # and v8/v9 (phoneme alphabet → space/concat) the same way.
        rendered = game.gen.render_surface(
            tokens, mask=gen_mask, eos_id=eos_id, output_mode="romanized",
        )
        for j, text in enumerate(rendered):
            text = text.strip()
            if text:
                documents[j].append(text)
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
                "language.  Read it and describe, in one short English "
                "sentence, what kind of scene, topic, or situation it "
                "seems to be about.  Do not translate word-for-word; "
                "give the abstract gist."
            ),
        },
        {
            "role": "user",
            "content": f"{document}\n\nWhat is this about?",
        },
    ]
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True,
    )
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=60,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
            repetition_penalty=1.1,
            pad_token_id=tokenizer.pad_token_id,
        )
    generated = out[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(generated, skip_special_tokens=True).strip()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--decoder-path", default="data/models/v7/vae_decoder.pt")
    parser.add_argument("--spm-path", default="data/models/v7/spm.model",
                        help="SentencePiece .model (v7) or phoneme alphabet .json (v8/v9)")
    parser.add_argument("--embedding-store", default="data/embeddings")
    parser.add_argument("--generator-name", default="multilingual_vae",
                        choices=["multilingual_vae", "phoneme_vae"],
                        help="VAE generator backend; phoneme_vae for v8/v9 decoders")
    parser.add_argument("--generator-vocab-size", type=int, default=None,
                        help="Override vocab_size (v8=30, v9=5001, v9.5=30819)")
    parser.add_argument("--qwen-model", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--sbert-model", default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--num-samples", type=int, default=20)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    device = torch.device(args.device)
    rng = np.random.default_rng(args.seed)

    logger.info("Sampling %d diverse embeddings...", args.num_samples)
    from lfm.embeddings.store import EmbeddingStore
    store = EmbeddingStore(args.embedding_store)
    store.load()
    clusters = rng.choice(
        list(store._cluster_index.keys()),
        size=args.num_samples,
        replace=False,
    )
    indices = [
        int(store.sample_from_cluster(int(c), 1, rng=rng)[0])
        for c in clusters
    ]

    passages = {}
    target = set(indices)
    with open(f"{args.embedding_store}/passages.jsonl") as f:
        for i, line in enumerate(f):
            if i in target:
                passages[i] = json.loads(line)["text"]
            if len(passages) == len(target):
                break
    sources = [passages[i] for i in indices]

    embeddings = torch.tensor(
        store._embeddings[indices], dtype=torch.float32, device=device,
    )
    del store

    logger.info("Loading dialogue game from %s", args.checkpoint)
    game = load_dialogue_game(
        args.checkpoint, args.decoder_path, args.spm_path,
        args.embedding_store, device,
        generator_name=args.generator_name,
        generator_vocab_size=args.generator_vocab_size,
    )
    # vocab_size + EOS from the generator (VAE-backend-agnostic).
    vocab_size = game.gen._vocab_size  # noqa: SLF001
    eos_id = game.gen.eos_id

    logger.info("Generating Neuroglot documents...")
    with torch.no_grad():
        doc_turns = generate_documents(game, embeddings, vocab_size, eos_id)
    docs = [join_turns(d) for d in doc_turns]

    del game
    torch.cuda.empty_cache()

    logger.info("Loading Qwen %s for interpretation...", args.qwen_model)
    from transformers import AutoModelForCausalLM, AutoTokenizer
    qwen_tok = AutoTokenizer.from_pretrained(args.qwen_model)
    if qwen_tok.pad_token is None:
        qwen_tok.pad_token = qwen_tok.eos_token
    qwen = AutoModelForCausalLM.from_pretrained(
        args.qwen_model, torch_dtype=torch.bfloat16,
    ).to(device)
    qwen.eval()

    logger.info("Generating interpretations...")
    interps = []
    for doc in docs:
        interps.append(interpret(qwen, qwen_tok, doc))

    del qwen, qwen_tok
    torch.cuda.empty_cache()

    # Independent sentence encoder — NOT Qwen.  Breaks the confound.
    logger.info("Loading sentence encoder %s...", args.sbert_model)
    from transformers import AutoTokenizer as ATok
    sbert_tok = ATok.from_pretrained(args.sbert_model)
    sbert = AutoModelForCausalLM.from_pretrained  # noqa: E501
    from transformers import AutoModel
    sbert = AutoModel.from_pretrained(args.sbert_model).to(device)
    sbert.eval()

    def sbert_encode(texts: list[str]) -> torch.Tensor:
        inputs = sbert_tok(
            texts, padding=True, truncation=True,
            max_length=256, return_tensors="pt",
        ).to(device)
        with torch.no_grad():
            out = sbert(**inputs)
        # Mean pooling
        mask = inputs["attention_mask"].unsqueeze(-1).float()
        pooled = (out.last_hidden_state * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
        return torch.nn.functional.normalize(pooled, dim=-1).float().cpu()

    logger.info("Encoding %d sources and interpretations via SBERT...", len(sources))
    source_embs = sbert_encode(sources)  # (N, D)
    interp_embs = sbert_encode(interps)  # (N, D)

    # Retrieval metrics
    n = len(sources)
    sim = interp_embs @ source_embs.t()  # (N, N) — interp_i vs source_j
    # For each interp_i, rank of true source_i among all sources
    ranks = []
    for i in range(n):
        row = sim[i]
        target_score = row[i].item()
        rank = int((row > target_score).sum().item()) + 1  # 1-indexed
        ranks.append(rank)
    ranks_arr = np.array(ranks)
    recall_at_1 = float((ranks_arr == 1).sum()) / n
    recall_at_3 = float((ranks_arr <= 3).sum()) / n
    mrr = float(np.mean(1.0 / ranks_arr))

    # Mean cosine gap: correct source vs average over wrong sources
    diag = sim.diag().mean().item()
    off_diag = (sim.sum() - sim.diag().sum()).item() / (n * (n - 1))
    gap = diag - off_diag

    print()
    print("=" * 88)
    print("SOURCE-RETRIEVAL METRIC (via SBERT, independent of Qwen)")
    print("=" * 88)
    print(f"samples:          {n}")
    print(f"chance recall@1:  {1.0 / n:.4f}")
    print(f"recall@1:         {recall_at_1:.4f}  ({'+' if recall_at_1 > 1.0 / n else '−'}{recall_at_1 - 1.0 / n:+.4f} vs chance)")
    print(f"recall@3:         {recall_at_3:.4f}")
    print(f"MRR:              {mrr:.4f}")
    print()
    print(f"mean cos(interp_i, source_i):  {diag:.4f}")
    print(f"mean cos(interp_i, source_j):  {off_diag:.4f}")
    print(f"gap (correct − random):        {gap:+.4f}")
    print()
    print("Interpretation: if gap > 0 and recall@1 > chance, Qwen's")
    print("interpretation is genuinely coupled to the source content,")
    print("even if the coupling is via abstraction rather than literal")
    print("translation.  Near-zero gap = no signal.")
    print()
    print("=" * 88)
    print("PER-SAMPLE DETAIL")
    print("=" * 88)
    for i in range(min(n, 10)):
        print(f"\n[{i}] rank={ranks[i]} cos={sim[i, i].item():.3f}")
        print(f"  src:    {sources[i][:140]}")
        print(f"  interp: {interps[i][:140]}")


if __name__ == "__main__":
    main()

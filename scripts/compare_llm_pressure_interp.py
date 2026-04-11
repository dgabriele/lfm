"""Compare downstream Qwen interpretability of two dialogue checkpoints.

Takes two dialogue-game checkpoints — a baseline (no LLM pressure,
e.g. the 98.3% ``dialogue_game_v7/best.pt``) and a pressured one
(e.g. ``dialogue_game_v7_llm_pressure/latest.pt``) — and for a
shared set of random embeddings generates a 4-turn Neuroglot
document from each.  Each document is then handed to a frozen base
Qwen 2.5 Instruct model which is asked to interpret it in English.
No fine-tuning, no translation training — the only thing varying
between the two columns is whether the agent was trained under the
LLM-pressure objective.

If the pressured column's interpretations are visibly more
coherent / on-topic than the baseline column's, that is direct
evidence the pressure moved the Neuroglot into a region of Qwen's
latent space where zero-shot interpretation does something useful.

Usage::

    poetry run python scripts/compare_llm_pressure_interp.py \\
        --baseline data/dialogue_game_v7/best.pt \\
        --pressured data/dialogue_game_v7_llm_pressure/latest.pt \\
        --num-samples 10
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import sys

import numpy as np
import sentencepiece as spm_lib
import torch

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
        # LLM pressure off during generation — the scorer would otherwise
        # try to load Qwen a second time here and is not needed.
        llm_loss_weight=0.0,
    )
    faculty = LanguageFaculty(game_cfg.build_faculty_config()).to(device)
    game = DialogueGame(game_cfg, faculty).to(device)
    game.load_checkpoint_state(ckpt)
    game.eval()
    return game


def generate_documents(game, embeddings, sp, vocab_size, eos_id):
    """Generate 4-turn romanized Neuroglot documents for a batch of embeddings."""
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
            torch.stack(context_summaries, dim=1)
            if context_summaries else None
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


def compute_ppl(model, tokenizer, document: str) -> float:
    inputs = tokenizer(document, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model(**inputs, labels=inputs["input_ids"])
    return math.exp(min(out.loss.item(), 20))


def interpret(model, tokenizer, document: str) -> str:
    """Ask Qwen to interpret a Neuroglot document in English."""
    messages = [
        {
            "role": "system",
            "content": (
                "You will be given a passage written in an artificial "
                "language called Neuroglot.  Neuroglot is how an agent "
                "expresses its perception of some input — it is not "
                "a transcription of any natural human language, and "
                "you have never seen it before.  Read the passage and "
                "guess, in English, what it might be about.  It is "
                "OK to say you cannot tell."
            ),
        },
        {
            "role": "user",
            "content": f"{document}\n\nWhat is this passage about?",
        },
    ]
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True,
    )
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=200,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
            repetition_penalty=1.2,
            pad_token_id=tokenizer.pad_token_id,
        )
    generated = outputs[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(generated, skip_special_tokens=True).strip()


def join_turns(turns: list[str]) -> str:
    return " ".join(t.rstrip(".") + "." for t in turns)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", required=True,
                        help="Baseline dialogue checkpoint (no LLM pressure)")
    parser.add_argument("--pressured", required=True,
                        help="LLM-pressure trained dialogue checkpoint")
    parser.add_argument("--decoder-path", default="data/models/v7/vae_decoder.pt")
    parser.add_argument("--spm-path", default="data/models/v7/spm.model")
    parser.add_argument("--embedding-store", default="data/embeddings")
    parser.add_argument("--qwen-model", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--num-samples", type=int, default=10)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    device = torch.device(args.device)
    rng = np.random.default_rng(args.seed)

    # Pick the same diverse sample indices for both models so we can
    # compare apples to apples.
    logger.info("Sampling embeddings...")
    from lfm.embeddings.store import EmbeddingStore
    store = EmbeddingStore(args.embedding_store)
    store.load()
    clusters = rng.choice(
        list(store._cluster_index.keys()),
        size=args.num_samples, replace=False,
    )
    indices = [
        store.sample_from_cluster(int(c), 1, rng=rng)[0]
        for c in clusters
    ]
    passages = []
    with open(f"{args.embedding_store}/passages.jsonl") as f:
        target = set(indices)
        for i, line in enumerate(f):
            if i in target:
                passages.append((i, json.loads(line)["text"]))
            if len(passages) == len(target):
                break
    passages.sort(key=lambda x: indices.index(x[0]))
    passages = [p[1] for p in passages]

    embeddings = torch.tensor(
        store._embeddings[indices], dtype=torch.float32, device=device,
    )
    # Free the numpy memory-mapped store.
    del store

    # Generate from baseline
    logger.info("Loading baseline dialogue game...")
    baseline_game = load_dialogue_game(
        args.baseline, args.decoder_path, args.spm_path, args.embedding_store, device,
    )
    sp = spm_lib.SentencePieceProcessor()
    sp.Load(args.spm_path)
    vocab_size = sp.GetPieceSize()
    eos_id = baseline_game.gen.eos_id

    logger.info("Generating baseline documents...")
    with torch.no_grad():
        baseline_docs = generate_documents(baseline_game, embeddings, sp, vocab_size, eos_id)

    del baseline_game
    torch.cuda.empty_cache()

    # Generate from pressured
    logger.info("Loading LLM-pressured dialogue game...")
    pressured_game = load_dialogue_game(
        args.pressured, args.decoder_path, args.spm_path, args.embedding_store, device,
    )
    logger.info("Generating pressured documents...")
    with torch.no_grad():
        pressured_docs = generate_documents(pressured_game, embeddings, sp, vocab_size, eos_id)

    del pressured_game
    torch.cuda.empty_cache()

    # Load base Qwen-Instruct
    logger.info("Loading Qwen %s...", args.qwen_model)
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.qwen_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.qwen_model, torch_dtype=torch.bfloat16,
    ).to(device)
    model.eval()

    # Compare side-by-side
    print("=" * 88)
    print("LLM-PRESSURE DOWNSTREAM INTERPRETATION TEST")
    print("=" * 88)
    for i, (idx, passage, base_turns, pres_turns) in enumerate(
        zip(indices, passages, baseline_docs, pressured_docs),
    ):
        base_doc = join_turns(base_turns)
        pres_doc = join_turns(pres_turns)

        base_ppl = compute_ppl(model, tokenizer, base_doc)
        pres_ppl = compute_ppl(model, tokenizer, pres_doc)

        base_interp = interpret(model, tokenizer, base_doc)
        pres_interp = interpret(model, tokenizer, pres_doc)

        print(f"\n--- Sample {i+1} (embedding #{idx}) ---")
        print(f"\nORIGINAL ENGLISH:\n  {passage}")
        print(f"\nBASELINE Neuroglot (ppl={base_ppl:.1f}):\n  {base_doc[:240]}")
        print(f"\nBASELINE Qwen interpretation:\n  {base_interp}")
        print(f"\nPRESSURED Neuroglot (ppl={pres_ppl:.1f}):\n  {pres_doc[:240]}")
        print(f"\nPRESSURED Qwen interpretation:\n  {pres_interp}")
        print()

    print("=" * 88)


if __name__ == "__main__":
    main()

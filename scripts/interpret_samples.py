"""Generate IPA documents and prompt the fine-tuned LLM to interpret them.

Picks diverse embeddings, generates 4-turn Xenoglot documents through
the dialogue game, then asks the fine-tuned Qwen to interpret each one.

Usage:
    poetry run python scripts/interpret_samples.py \
        --model-path data/translator_dialogue_v7/model_latest \
        --dialogue-checkpoint data/dialogue_game_v7/best.pt \
        --num-samples 10
"""

from __future__ import annotations

import argparse
import json
import logging
import sys

import numpy as np
import sentencepiece as spm_lib
import torch

logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stderr)
logger = logging.getLogger(__name__)


def load_passages(path: str, indices: list[int]) -> list[str]:
    """Load specific passages by index from JSONL."""
    passages = {}
    target = set(indices)
    with open(path) as f:
        for i, line in enumerate(f):
            if i in target:
                passages[i] = json.loads(line)["text"]
            if len(passages) == len(target):
                break
    return [passages[i] for i in indices]


def generate_documents(
    game, embeddings: torch.Tensor, sp, vocab_size: int, eos_id: int,
) -> list[list[str]]:
    """Generate 4-turn IPA documents for a batch of embeddings."""
    from lfm.agents.decode import rerun_decoder_multiphrase_no_grad

    batch = embeddings.size(0)
    num_turns = game.config.num_turns
    targets = embeddings.unsqueeze(1)
    context_summaries = []
    documents = [[] for _ in range(batch)]

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

        # Decode to IPA
        from lfm.translator.romanize import syllable_hyphenate
        ipa_strings = []
        for j in range(batch):
            ids = [
                t.item() for t, m in zip(tokens[j], gen_mask[j])
                if m and t.item() != eos_id and t.item() < vocab_size
            ]
            ipa = sp.decode(ids).strip()
            ipa_strings.append(syllable_hyphenate(ipa) if ipa else "")
        for i, ipa in enumerate(ipa_strings):
            if ipa:
                documents[i].append(f"[T{turn_idx}] {ipa}")

        del hidden, tokens, gen_mask, bounds, z_seq, z_weights

    return documents


def compute_ppl(model, tokenizer, document: str) -> float:
    """Compute perplexity of a Xenoglot document."""
    import math
    inputs = tokenizer(document, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model(**inputs, labels=inputs["input_ids"])
    return math.exp(min(out.loss.item(), 20))


def interpret(model, tokenizer, document: str) -> str:
    """Prompt the fine-tuned LLM to interpret a Xenoglot document."""
    prompt = (
        f"[Xenoglot]\n{document}\n\n"
        f"The passage above is written in Xenoglot, a natural language "
        f"with its own vocabulary, grammar, and meaning. It is not a "
        f"phonetic transcription of any known human language. "
        f"Based on your understanding of Xenoglot, what is this passage "
        f"about? What does it mean? Interpret it in English.\n\n"
    )
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=512,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
            repetition_penalty=1.2,
            pad_token_id=tokenizer.pad_token_id,
        )
    # Decode only the generated part
    generated = outputs[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(generated, skip_special_tokens=True).strip()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", default="data/translator_dialogue_v7/model_latest")
    parser.add_argument("--dialogue-checkpoint", default="data/dialogue_game_v7/best.pt")
    parser.add_argument("--decoder-path", default="data/models/v7/vae_decoder.pt")
    parser.add_argument("--spm-path", default="data/models/v7/spm.model")
    parser.add_argument("--embedding-store", default="data/embeddings")
    parser.add_argument("--num-samples", type=int, default=10)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    device = torch.device(args.device)
    rng = np.random.default_rng(args.seed)

    # Load embedding store
    logger.info("Loading embeddings...")
    from lfm.embeddings.store import EmbeddingStore
    store = EmbeddingStore(args.embedding_store)
    store.load()
    n = store.num_passages

    # Pick diverse samples — one per cluster
    clusters = rng.choice(
        list(store._cluster_index.keys()),
        size=args.num_samples,
        replace=False,
    )
    indices = [
        store.sample_from_cluster(int(c), 1, rng=rng)[0]
        for c in clusters
    ]

    # Load original English passages
    logger.info("Loading passages...")
    passages = load_passages(f"{args.embedding_store}/passages.jsonl", indices)

    # Load dialogue game
    logger.info("Loading dialogue game...")
    from lfm.agents.games.dialogue import DialogueGame, DialogueGameConfig
    from lfm.faculty.model import LanguageFaculty

    ckpt = torch.load(args.dialogue_checkpoint, map_location=device, weights_only=False)
    game_cfg = DialogueGameConfig(
        decoder_path=args.decoder_path,
        spm_path=args.spm_path,
        embedding_store_dir=args.embedding_store,
        max_phrases=ckpt.get("max_phrases", 3),
        num_turns=ckpt.get("num_turns", 4),
        device=str(device),
    )
    faculty = LanguageFaculty(game_cfg.build_faculty_config()).to(device)
    game = DialogueGame(game_cfg, faculty).to(device)
    game.load_checkpoint_state(ckpt)
    game.eval()

    # Load SPM
    sp = spm_lib.SentencePieceProcessor()
    sp.Load(args.spm_path)
    vocab_size = sp.GetPieceSize()
    eos_id = game.gen.eos_id

    # Generate IPA documents
    logger.info("Generating Xenoglot documents...")
    embeddings = torch.tensor(
        store._embeddings[indices], dtype=torch.float32, device=device,
    )
    with torch.no_grad():
        documents = generate_documents(game, embeddings, sp, vocab_size, eos_id)

    # Free dialogue game VRAM
    del game, faculty
    torch.cuda.empty_cache()

    # Load fine-tuned LLM
    logger.info("Loading fine-tuned LLM...")
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    model = AutoModelForCausalLM.from_pretrained(args.model_path).to(device)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model.eval()

    # Run interpretations
    print("=" * 80)
    for i, (idx, passage, doc_turns) in enumerate(zip(indices, passages, documents)):
        doc_text = "\n".join(doc_turns)

        ppl = compute_ppl(model, tokenizer, doc_text)
        interpretation = interpret(model, tokenizer, doc_text)

        print(f"\n--- Sample {i+1} (embedding #{idx}, PPL={ppl:.1f}) ---")
        print(f"\nOriginal English:\n  {passage}")
        print(f"\nXenoglot:")
        for t in doc_turns:
            print(f"  {t}")
        print(f"\nLLM Interpretation:\n  {interpretation}")
        print()

    print("=" * 80)


if __name__ == "__main__":
    main()

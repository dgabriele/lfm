"""Forced-choice discrimination evaluation for agent checkpoints.

SBERT-based retrieval is blind to analogical similarity and has shown
no signal.  This script asks a different, easier question: given a
Neuroglot passage and K candidate source sentences, can a frozen LLM
pick the true source above chance?

This sidesteps the weaknesses of open-ended generation:

  - No hallucination step — the model's output is a single discrete
    choice, not a narrative interpretation.
  - No sentence-embedding blindness — the LLM reads both passages in
    its own latent space and compares them however it naturally does.
  - Small LLMs are much better at discrimination than generation.
    Qwen 0.5B might not be able to *describe* what a Neuroglot passage
    is about, but it can plausibly *distinguish* the true source from
    obvious distractors.

Two modes:
  - "direct":  prompt Qwen with (neuroglot, K candidate sources) and
    ask it to pick which source the passage is about.
  - "via-interp": first generate a Qwen interpretation of the
    Neuroglot, then prompt Qwen again with (interpretation, K
    candidate sources) to pick the most thematically related source.
    Uses Qwen's generation as a translation step before the judge.

Chance = 1/K.  Above-chance accuracy is meaningful signal.
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import re
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


def _qwen_short_reply(
    model, tokenizer, messages: list[dict], max_new: int = 16,
) -> str:
    """Greedy-decode a short reply from Qwen."""
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True,
    )
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
        )
    generated = out[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(generated, skip_special_tokens=True).strip()


def _parse_choice(text: str, k: int) -> int | None:
    """Extract a 1..K integer choice from the LLM's reply."""
    m = re.search(r"\b(\d+)\b", text)
    if m is None:
        return None
    val = int(m.group(1))
    if 1 <= val <= k:
        return val - 1  # 0-indexed
    return None


def direct_forced_choice(
    model, tokenizer, neuroglot: str, candidates: list[str], rng: random.Random,
) -> tuple[int | None, str]:
    """Ask Qwen which candidate source the Neuroglot passage is about.

    Returns (chosen_idx, raw_reply).
    """
    numbered = "\n".join(
        f"{i+1}. {c[:200]}" for i, c in enumerate(candidates)
    )
    messages = [
        {
            "role": "system",
            "content": (
                "You will be given a passage in an unfamiliar language and "
                "a numbered list of possible source sentences.  Even though "
                "the passage is not a direct translation, exactly one of "
                "the sentences reflects what the passage is really about "
                "— thematically, structurally, or analogically.  Your job "
                "is to pick which one.  Answer with the number only."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Passage:\n{neuroglot}\n\n"
                f"Candidate sources:\n{numbered}\n\n"
                f"Which source sentence best matches the passage?  "
                f"Answer with only a number from 1 to {len(candidates)}."
            ),
        },
    ]
    reply = _qwen_short_reply(model, tokenizer, messages, max_new=8)
    idx = _parse_choice(reply, len(candidates))
    return idx, reply


def interpret(model, tokenizer, document: str) -> str:
    messages = [
        {
            "role": "system",
            "content": (
                "Read the unfamiliar-language passage and describe in a "
                "single short English sentence what kind of scene, theme, "
                "or situation it seems to convey.  Be abstract — do not "
                "translate word-for-word."
            ),
        },
        {
            "role": "user",
            "content": f"{document}\n\nWhat is this passage about?",
        },
    ]
    return _qwen_short_reply(model, tokenizer, messages, max_new=60)


def via_interp_forced_choice(
    model, tokenizer, interpretation: str, candidates: list[str],
) -> tuple[int | None, str]:
    numbered = "\n".join(
        f"{i+1}. {c[:200]}" for i, c in enumerate(candidates)
    )
    messages = [
        {
            "role": "system",
            "content": (
                "You will be given a short description and a numbered "
                "list of candidate sentences.  Pick the candidate that "
                "is most thematically or analogically related to the "
                "description, even if the literal topics differ.  "
                "Answer with the number only."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Description: {interpretation}\n\n"
                f"Candidates:\n{numbered}\n\n"
                f"Which candidate is most thematically related?  "
                f"Answer with only a number from 1 to {len(candidates)}."
            ),
        },
    ]
    reply = _qwen_short_reply(model, tokenizer, messages, max_new=8)
    idx = _parse_choice(reply, len(candidates))
    return idx, reply


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--decoder-path", default="data/models/v7/vae_decoder.pt")
    parser.add_argument("--spm-path", default="data/models/v7/spm.model")
    parser.add_argument("--embedding-store", default="data/embeddings")
    parser.add_argument("--qwen-model", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--load-4bit", action="store_true",
                        help="Quantize the interpreter model to 4-bit via bitsandbytes")
    parser.add_argument("--num-samples", type=int, default=50)
    parser.add_argument("--k", type=int, default=10,
                        help="Number of candidates per forced-choice item")
    parser.add_argument("--mode", choices=["direct", "via-interp", "both"],
                        default="both")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    device = torch.device(args.device)
    rng = np.random.default_rng(args.seed)
    py_rng = random.Random(args.seed)

    logger.info("Sampling %d diverse embeddings...", args.num_samples)
    from lfm.embeddings.store import EmbeddingStore
    store = EmbeddingStore(args.embedding_store)
    store.load()
    cluster_pool = list(store._cluster_index.keys())
    chosen_clusters = rng.choice(cluster_pool, size=args.num_samples, replace=False)
    indices = [
        int(store.sample_from_cluster(int(c), 1, rng=rng)[0])
        for c in chosen_clusters
    ]

    # Also pull a large pool of distractor sentences from OTHER clusters
    # so each forced-choice item has fresh distractors.
    distractor_clusters = rng.choice(
        [c for c in cluster_pool if c not in set(chosen_clusters)],
        size=args.num_samples * (args.k - 1),
        replace=True,
    )
    distractor_indices = [
        int(store.sample_from_cluster(int(c), 1, rng=rng)[0])
        for c in distractor_clusters
    ]

    target = set(indices) | set(distractor_indices)
    passages = {}
    with open(f"{args.embedding_store}/passages.jsonl") as f:
        for i, line in enumerate(f):
            if i in target:
                passages[i] = json.loads(line)["text"]
            if len(passages) == len(target):
                break

    sources = [passages[i] for i in indices]
    distractor_texts = [passages[i] for i in distractor_indices]

    embeddings = torch.tensor(
        store._embeddings[indices], dtype=torch.float32, device=device,
    )
    del store

    logger.info("Loading dialogue game...")
    game = load_dialogue_game(
        args.checkpoint, args.decoder_path, args.spm_path,
        args.embedding_store, device,
    )
    sp = spm_lib.SentencePieceProcessor()
    sp.Load(args.spm_path)
    vocab_size = sp.GetPieceSize()
    eos_id = game.gen.eos_id

    logger.info("Generating Neuroglot documents...")
    with torch.no_grad():
        doc_turns = generate_documents(game, embeddings, sp, vocab_size, eos_id)
    docs = [join_turns(d) for d in doc_turns]

    del game
    torch.cuda.empty_cache()

    logger.info(
        "Loading Qwen %s (%s)...",
        args.qwen_model, "4-bit" if args.load_4bit else "bf16",
    )
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.qwen_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    if args.load_4bit:
        from transformers import BitsAndBytesConfig
        bnb = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )
        model = AutoModelForCausalLM.from_pretrained(
            args.qwen_model,
            quantization_config=bnb,
            device_map={"": 0},
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            args.qwen_model, torch_dtype=torch.bfloat16,
        ).to(device)
    model.eval()

    n = args.num_samples
    k = args.k

    results: dict[str, list[int | None]] = {"direct": [], "via-interp": []}
    interp_samples: list[str] = []

    for i in range(n):
        # Build the K candidate list: true source + K-1 distractors.
        distr_slice = distractor_texts[i * (k - 1):(i + 1) * (k - 1)]
        cand_texts = [sources[i]] + distr_slice
        # Shuffle so the true index is unknown to the model
        order = list(range(k))
        py_rng.shuffle(order)
        shuffled = [cand_texts[j] for j in order]
        true_idx = order.index(0)

        if args.mode in ("direct", "both"):
            idx, _ = direct_forced_choice(model, tokenizer, docs[i], shuffled, py_rng)
            results["direct"].append(idx == true_idx if idx is not None else None)

        if args.mode in ("via-interp", "both"):
            interp = interpret(model, tokenizer, docs[i])
            idx, _ = via_interp_forced_choice(model, tokenizer, interp, shuffled)
            results["via-interp"].append(idx == true_idx if idx is not None else None)
            if i < 8:
                interp_samples.append(
                    f"[{i}] src: {sources[i][:120]}\n   doc: {docs[i][:160]}\n"
                    f"   interp: {interp[:160]}\n"
                    f"   true_idx={true_idx+1}  chosen={idx+1 if idx is not None else '?'}  "
                    f"correct={idx == true_idx}"
                )

    print()
    print("=" * 88)
    print("FORCED-CHOICE DISCRIMINATION")
    print("=" * 88)
    print(f"samples: {n}    K (candidates per item): {k}    chance: {1.0 / k:.4f}")
    print()
    for mode in ["direct", "via-interp"]:
        r = [x for x in results[mode] if x is not None]
        if not r:
            continue
        acc = float(sum(r)) / len(r)
        n_valid = len(r)
        chance = 1.0 / k
        # Binomial test against chance
        from math import comb
        correct = sum(r)
        # P(X >= correct) under Bin(n_valid, chance)
        pval = sum(
            comb(n_valid, j) * (chance ** j) * ((1 - chance) ** (n_valid - j))
            for j in range(correct, n_valid + 1)
        )
        delta = acc - chance
        marker = ""
        if pval < 0.01:
            marker = "  ** (p<0.01)"
        elif pval < 0.05:
            marker = "  * (p<0.05)"
        print(
            f"  {mode:10s}  correct={correct}/{n_valid}  "
            f"acc={acc:.3f}  (chance={chance:.3f}  Δ={delta:+.3f}  p={pval:.4f}){marker}"
        )
    print()
    if interp_samples:
        print("=" * 88)
        print("SAMPLE INTERPRETATIONS (via-interp mode)")
        print("=" * 88)
        for s in interp_samples:
            print(s)
            print()


if __name__ == "__main__":
    main()

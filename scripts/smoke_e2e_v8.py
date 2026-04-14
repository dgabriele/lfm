#!/usr/bin/env python
"""End-to-end smoke test of the v8 pipeline.

Validates the full Neuroglot interpretation loop:

    Qwen-latent embedding
       │
       ▼  (v8 dialogue game generates Neuroglot from latent)
    Neuroglot text (phoneme surface)
       │
       ▼  (Qwen-Instruct interprets, possibly with FT)
    English description

Run this once the dialogue game has been trained (configs/dialogue_v8_phase1.yaml)
and we have a checkpoint at ``data/dialogue_game_v8/best.pt``.  Optionally
provide an FT'd Qwen checkpoint via ``--qwen-checkpoint`` to test the
trained interpreter; otherwise zero-shot Qwen-Instruct is used (sanity
check that the loop runs end-to-end; quality won't be meaningful without FT).

Usage:
    poetry run python scripts/smoke_e2e_v8.py \\
        [--n-samples 8] \\
        [--dialogue-checkpoint data/dialogue_game_v8/best.pt] \\
        [--alphabet data/phoneme_alphabet_multi.json] \\
        [--decoder data/models/v8/vae_decoder.pt] \\
        [--embedding-store data/embeddings] \\
        [--qwen-checkpoint <path>]   # optional FT'd Qwen
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import torch

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger(__name__)


def _load_dialogue_game(args):
    """Load the v8 dialogue game from a checkpoint, returning (game, device)."""
    from lfm.agents.games.dialogue import DialogueGame, DialogueGameConfig
    from lfm.embeddings.store import EmbeddingStore
    from lfm.faculty.model import LanguageFaculty

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"loading dialogue checkpoint from {args.dialogue_checkpoint}")
    ckpt = torch.load(
        args.dialogue_checkpoint, map_location=device, weights_only=False,
    )

    # Auto-detect embedding_dim from the store the dialogue was trained on.
    dim_kwargs: dict = {}
    try:
        meta = EmbeddingStore.read_metadata(args.embedding_store)
        if "embedding_dim" in meta:
            dim_kwargs["embedding_dim"] = int(meta["embedding_dim"])
    except FileNotFoundError:
        pass

    cfg = DialogueGameConfig(
        decoder_path=args.decoder,
        spm_path=args.alphabet,            # phoneme alphabet JSON for v8
        embedding_store_dir=args.embedding_store,
        max_phrases=ckpt.get("max_phrases", ckpt.get("max_segments", 4)),
        num_turns=ckpt.get("num_turns", 4),
        device=str(device),
        contrastive_scoring=bool(ckpt.get("contrastive_scoring", False)),
        **dim_kwargs,
    )
    faculty = LanguageFaculty(cfg.build_faculty_config()).to(device)
    game = DialogueGame(cfg, faculty).to(device)
    game.load_checkpoint_state(ckpt)
    game.eval()
    return game, device


def _sample_embeddings(args, n: int, device) -> torch.Tensor:
    """Sample N random embeddings from the store."""
    from lfm.embeddings.store import EmbeddingStore

    store = EmbeddingStore(args.embedding_store)
    store.load()
    arr = np.asarray(store._embeddings)
    rng = np.random.default_rng(args.seed)
    idx = rng.choice(arr.shape[0], size=n, replace=False)
    sample = arr[idx]
    return torch.tensor(sample, dtype=torch.float32, device=device)


def _generate_neuroglot(game, embeddings: torch.Tensor) -> list[str]:
    """Run dialogue game on a batch of embeddings, returning Neuroglot strings.

    Uses VAE-agnostic ``game.gen.render_surface`` so this works for v7
    (IPA) and v8 (phoneme) decoders identically.
    """
    targets = embeddings.unsqueeze(1)  # (B, 1, dim) — single-target case
    num_turns = game.config.num_turns

    context_summaries: list[torch.Tensor] = []
    surfaces: list[list[str]] = [[] for _ in range(embeddings.size(0))]

    with torch.no_grad():
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
            tokens, gen_mask, phrase_bounds = game.phrase_decoder.decode(
                z_seq, z_weights,
            )
            from lfm.agents.decode import rerun_decoder_multiphrase_no_grad
            hidden = rerun_decoder_multiphrase_no_grad(
                game.gen, z_seq, z_weights, tokens, gen_mask, phrase_bounds,
            )
            trimmed_mask = gen_mask[:, :hidden.size(1)]
            summary = game._summarize_turn(hidden, trimmed_mask)
            context_summaries.append(summary)

            # VAE-agnostic surface render — works identically for v7 IPA
            # decoder and v8 phoneme decoder.
            turn_surfaces = game.gen.render_surface(
                tokens, mask=gen_mask,
            )
            for b, s in enumerate(turn_surfaces):
                if s:
                    surfaces[b].append(f"[T{turn_idx}] {s}")

    return ["\n".join(t) for t in surfaces]


def _interpret_with_qwen(neuroglot_texts: list[str], qwen_checkpoint: str | None) -> list[str]:
    """Feed Neuroglot expressions to Qwen-Instruct → English descriptions."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model_name = qwen_checkpoint or "Qwen/Qwen2.5-0.5B-Instruct"
    logger.info(f"loading Qwen model: {model_name}")
    tok = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.float16,
    ).to("cuda" if torch.cuda.is_available() else "cpu").eval()

    descriptions: list[str] = []
    for ng in neuroglot_texts:
        prompt = (
            "The following is an utterance in 'Neuroglot', an emergent "
            "language produced by a neural network to describe a perceived "
            "concept.  Interpret what the utterance is about in plain "
            "English (one sentence).\n\n"
            f"Neuroglot:\n{ng}\n\n"
            "English interpretation:"
        )
        messages = [{"role": "user", "content": prompt}]
        chat = tok.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
        )
        input_ids = tok(chat, return_tensors="pt").input_ids.to(model.device)
        with torch.no_grad():
            out = model.generate(
                input_ids, max_new_tokens=80, do_sample=False, temperature=1.0,
            )
        gen = tok.decode(out[0][input_ids.size(1):], skip_special_tokens=True)
        descriptions.append(gen.strip())
    return descriptions


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dialogue-checkpoint",
                        default="data/dialogue_game_v8/best.pt")
    parser.add_argument("--alphabet",
                        default="data/phoneme_alphabet_multi.json")
    parser.add_argument("--decoder",
                        default="data/models/v8/vae_decoder.pt")
    parser.add_argument("--embedding-store", default="data/embeddings")
    parser.add_argument("--qwen-checkpoint", default=None,
                        help="FT'd Qwen checkpoint; default is base Qwen-Instruct")
    parser.add_argument("--n-samples", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    for required in (args.dialogue_checkpoint, args.alphabet, args.decoder):
        if not Path(required).exists():
            raise FileNotFoundError(f"required path missing: {required}")

    game, device = _load_dialogue_game(args)
    embeddings = _sample_embeddings(args, args.n_samples, device)

    logger.info(f"generating Neuroglot for {args.n_samples} embeddings...")
    neuroglot_texts = _generate_neuroglot(game, embeddings)

    logger.info(f"running Qwen interpretation...")
    descriptions = _interpret_with_qwen(neuroglot_texts, args.qwen_checkpoint)

    print("\n" + "=" * 80)
    for i, (ng, desc) in enumerate(zip(neuroglot_texts, descriptions)):
        print(f"\n[{i}] Neuroglot:\n{ng}")
        print(f"    English (Qwen):\n    {desc}")
        print("-" * 80)


if __name__ == "__main__":
    main()

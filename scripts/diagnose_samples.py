"""Standalone diagnostic: load a Phase 1 checkpoint and produce N samples
under argmax + n-gram blocking + GT-length-cap, so we can inspect the LM's
underlying distribution without sampling drift artefacts.

Non-disruptive: runs against any checkpoint file; doesn't touch a running
training process.

Usage:
    poetry run python scripts/diagnose_samples.py CONFIG_PATH --checkpoint X.pt --n 8
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
import yaml
from transformers import PreTrainedTokenizerFast

from lfm.synth.backend import CausalDecoderBackend
from lfm.synth.cipher import WordCipher
from lfm.synth.config import SynthConfig
from lfm.synth.model import SynthLM
from lfm.synth.vocab import AlienVocab


def block_ngram_repeats(logits: torch.Tensor, history: list[int], n: int) -> None:
    if len(history) < n:
        return
    prefix = tuple(history[-(n - 1):])
    for i in range(len(history) - (n - 1)):
        if tuple(history[i : i + n - 1]) == prefix:
            banned = history[i + n - 1]
            logits[banned] = float("-inf")


@torch.no_grad()
def generate_argmax(
    model: SynthLM, alien_tok: PreTrainedTokenizerFast, cipher: WordCipher,
    sentence: str, length_cap: int, device: torch.device, block_n: tuple = (3, 4),
) -> tuple[str, str]:
    eos_id = alien_tok.eos_token_id
    gt_lower = cipher.encode_for_tokenizer(sentence)
    gt_ids = alien_tok(gt_lower, return_tensors="pt")["input_ids"][0]
    gt_text = cipher.encode_sentence(sentence)
    seed = gt_ids[:1].unsqueeze(0).to(device)

    context = model.backend.embed_alien(seed)
    generated = [int(seed[0, 0])]
    cap = min(int(gt_ids.numel() * 1.3) + 5, length_cap)
    for _ in range(cap):
        hidden = model.backend.forward_hidden(context)
        logits = model.backend.alien_logits(hidden[:, -1:]).squeeze(1).squeeze(0)
        for n in block_n:
            block_ngram_repeats(logits, generated, n)
        nxt = int(logits.argmax().item())
        generated.append(nxt)
        if nxt == eos_id:
            break
        nxt_t = torch.tensor([[nxt]], device=device)
        context = torch.cat([context, model.backend.embed_alien(nxt_t)], dim=1)
    return gt_text, alien_tok.decode(generated, skip_special_tokens=True)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("config")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--n", type=int, default=8)
    p.add_argument("--device", default="cuda")
    p.add_argument("--max-len", type=int, default=128)
    args = p.parse_args()

    cfg = SynthConfig(**yaml.safe_load(Path(args.config).read_text()))
    out_dir = Path(cfg.output_dir)
    vocab = AlienVocab.load(out_dir)
    cipher = WordCipher.from_dirs(vocab, out_dir)
    alien_tok = PreTrainedTokenizerFast.from_pretrained(str(out_dir / "alien_tokenizer"))

    print("loading backend + checkpoint...")
    backend = CausalDecoderBackend(
        cfg.base_model_name, alien_vocab_size=len(alien_tok), with_reference_body=False,
    )
    model = SynthLM(backend, cfg)
    raw = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    state = raw["model"] if isinstance(raw, dict) and "model" in raw else raw
    model.load_phase1_state(state)
    device = torch.device(args.device)
    model.to(device).eval()

    # Pick N sentences spanning the diagnostic distribution
    import json, random
    dataset_path = Path(cfg.phase1_dataset_dir)
    if dataset_path.suffix == ".jsonl":
        all_lines = [json.loads(l)["text"] for l in dataset_path.read_text().splitlines() if l.strip()]
    else:
        all_lines = [l.strip() for l in dataset_path.read_text().splitlines() if l.strip()]
    rng = random.Random(0)
    rng.shuffle(all_lines)
    samples = all_lines[: args.n]

    print(f"\n=== argmax + 3,4-gram blocking + GT-length cap ===\n")
    for i, sent in enumerate(samples):
        gt, gen = generate_argmax(model, alien_tok, cipher, sent, args.max_len, device)
        print(f"[{i+1}] EN:  {sent[:140]}")
        print(f"    GT:  {gt[:200]}")
        print(f"    GEN: {gen[:200]}")
        print()


if __name__ == "__main__":
    main()

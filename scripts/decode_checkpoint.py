"""Offline decode test against a v13 VAE checkpoint.

Loads checkpoint + SPM, encodes a few tagged prompts, and decodes
them greedily (argmax) and via the diagnostics.py nucleus sampler.
Used to confirm the tag-handling fix (ccc470f) works against the
checkpoint the still-running training process produced.

Usage::

    poetry run python scripts/decode_checkpoint.py \\
        --checkpoint data/models/v13_english_ortho/vae_resume.pt \\
        --config     data/models/v13_english_ortho/config.yaml \\
        --spm        data/datasets/english-constituents-v13/spm.model
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
import yaml
import sentencepiece as spm
import torch.nn.functional as F
from torch import nn

from lfm.generator.pretrain.config import VAEPretrainConfig
from lfm.generator.pretrain.model import build_model
from lfm.generator.pretrain.diagnostics import sample_decode, encode_text
from lfm.generator.layers import PhraseDecoder, multiscale_causal_mask


def _specials(sp: spm.SentencePieceProcessor) -> set[int]:
    return {
        tid
        for tid in [sp.unk_id(), sp.bos_id(), sp.eos_id(), sp.pad_id()]
        if tid >= 0
    }


@torch.no_grad()
def greedy_decode(
    z: torch.Tensor,
    *,
    modules: dict,
    cfg: VAEPretrainConfig,
    vocab_size: int,
    bos_id: int,
    eos_id: int,
    device: torch.device,
    sp: spm.SentencePieceProcessor,
    max_len: int | None = None,
) -> list[str]:
    _dec = modules["decoder"]
    _is_ling = isinstance(_dec, PhraseDecoder)
    n = z.size(0)
    _n_mem = getattr(cfg, "num_memory_tokens", 1)
    mem = modules["latent_to_decoder"](z).reshape(n, _n_mem, -1)
    ids = torch.full((n, 1), bos_id, dtype=torch.long, device=device)
    max_len = max_len or cfg.max_seq_len
    specials = _specials(sp)
    finished = torch.zeros(n, dtype=torch.bool, device=device)

    for _ in range(max_len - 1):
        if _is_ling:
            tgt = modules["dec_token_embedding"](ids)
            if not isinstance(modules["dec_pos_embedding"], nn.Identity):
                p = torch.arange(ids.size(1), device=device).unsqueeze(0)
                tgt = tgt + modules["dec_pos_embedding"](p)
            cm = multiscale_causal_mask(
                ids.size(1),
                num_heads=cfg.decoder_num_heads,
                head_windows=cfg.attention_head_windows,
                global_every=cfg.attention_global_every,
                device=device,
            )
            out = _dec(tgt, mem, tgt_mask=cm, rope_freqs=modules.get("_rope_freqs"))
        else:
            p = torch.arange(ids.size(1), device=device).unsqueeze(0)
            tgt = modules["dec_token_embedding"](ids) + modules["dec_pos_embedding"](p)
            cm = nn.Transformer.generate_square_subsequent_mask(
                ids.size(1), device=device
            )
            out = _dec(tgt=tgt, memory=mem, tgt_mask=cm)

        logits = modules["output_head"](out[:, -1])
        for tid in specials:
            logits[:, tid] = float("-inf")
        logits[:, bos_id] = float("-inf")
        nxt = logits.argmax(dim=-1, keepdim=True)
        ids = torch.cat([ids, nxt], dim=1)
        finished |= nxt.squeeze(-1).eq(eos_id)
        if finished.all():
            break

    texts = []
    filt = specials | {bos_id, eos_id}
    for j in range(n):
        toks = ids[j, 1:].cpu().tolist()
        if eos_id in toks:
            toks = toks[: toks.index(eos_id)]
        toks = [x for x in toks if x < vocab_size and x not in filt]
        texts.append(sp.decode(toks))
    return texts


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True, type=Path)
    ap.add_argument("--config", required=True, type=Path)
    ap.add_argument("--spm", required=True, type=Path)
    ap.add_argument(
        "--prompts",
        nargs="+",
        default=[
            "<NP> seeds grapes peaches barrels or bottles </NP>",
            "<NP> the convergence on small angular scales </NP>",
            "<S> she entered a popular restaurant </S>",
            "<VP> jumped over the lazy dog </VP>",
            "<PP> in the atlantic </PP>",
        ],
    )
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    device = torch.device(args.device)
    sp = spm.SentencePieceProcessor(model_file=str(args.spm))
    vocab_size = sp.vocab_size()
    bos_id = vocab_size
    eos_id = vocab_size + 1
    full_vocab = vocab_size + 2

    cfg_dict = yaml.unsafe_load(args.config.read_text())
    # Auto-scaling may have bumped max_seq_len mid-run; the checkpoint
    # carries the final value.  Use it so pos-embedding tables line up.
    ckpt_peek = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if "max_seq_len" in ckpt_peek:
        cfg_dict["max_seq_len"] = int(ckpt_peek["max_seq_len"])
    del ckpt_peek
    cfg = VAEPretrainConfig(**cfg_dict)

    modules = build_model(cfg, cfg.decoder_hidden_dim, full_vocab, device)
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    module_states = ckpt.get("modules", ckpt)
    for name, mod in modules.items():
        if name.startswith("_"):
            continue
        state = module_states.get(name)
        if state is None:
            print(f"[warn] no state for '{name}'")
            continue
        mod.load_state_dict(state)
        mod.eval()

    print(f"checkpoint: epoch={ckpt.get('epoch', '?')} step={ckpt.get('global_step', '?')}")
    print(f"SPM: vocab={vocab_size} unk={sp.unk_id()} bos={sp.bos_id()} eos={sp.eos_id()} pad={sp.pad_id()}")
    print(f"specials suppressed: {sorted(_specials(sp))}")

    # Encode prompts
    specials = _specials(sp)
    batch = []
    for p in args.prompts:
        ids = sp.encode(p, out_type=int)
        ids = [x for x in ids if x not in specials]
        batch.append(ids)
    lengths = [len(b) for b in batch]
    B = len(batch)
    L = max(lengths)
    src = torch.zeros((B, L), dtype=torch.long, device=device)
    for i, b in enumerate(batch):
        src[i, : len(b)] = torch.tensor(b, device=device)

    # Encode via the canonical encode_text path from diagnostics.py.
    z = encode_text(
        src,
        torch.tensor(lengths, device=device),
        modules=modules,
        cfg=cfg,
        device=device,
    )

    print("\n=== GREEDY (argmax) ===\n")
    greedy = greedy_decode(
        z, modules=modules, cfg=cfg, vocab_size=vocab_size,
        bos_id=bos_id, eos_id=eos_id, device=device, sp=sp,
    )
    for p, g in zip(args.prompts, greedy):
        print(f"orig: {p}")
        print(f" dec: {g}")
        print()

    print("\n=== NUCLEUS (fixed diagnostics.sample_decode) ===\n")
    sampled = sample_decode(
        z, modules=modules, cfg=cfg, vocab_size=vocab_size,
        bos_id=bos_id, eos_id=eos_id, device=device, sp=sp,
    )
    for p, s in zip(args.prompts, sampled):
        print(f"orig: {p}")
        print(f" dec: {s}")
        print()

    # --- Interpolation between first two prompts ---
    if z.size(0) >= 2:
        print("\n=== INTERPOLATION (z[0] → z[1]) ===\n")
        alphas = [0.0, 0.25, 0.5, 0.75, 1.0]
        z_interp = torch.stack(
            [(1 - a) * z[0] + a * z[1] for a in alphas], dim=0
        )
        decoded = sample_decode(
            z_interp, modules=modules, cfg=cfg, vocab_size=vocab_size,
            bos_id=bos_id, eos_id=eos_id, device=device, sp=sp,
        )
        print(f"  A: {args.prompts[0]}")
        print(f"  B: {args.prompts[1]}")
        print()
        for a, d in zip(alphas, decoded):
            print(f"  α={a:.2f}: {d}")

    # --- Perturbation around z[0] ---
    if z.size(0) >= 1:
        print("\n=== PERTURBATION (around z[0]) ===\n")
        sigmas = [0.0, 0.25, 0.5, 1.0, 2.0]
        gen = torch.Generator(device=device).manual_seed(42)
        z_pert = torch.stack(
            [z[0] + s * torch.randn(z[0].shape, device=device, generator=gen)
             for s in sigmas], dim=0,
        )
        decoded = sample_decode(
            z_pert, modules=modules, cfg=cfg, vocab_size=vocab_size,
            bos_id=bos_id, eos_id=eos_id, device=device, sp=sp,
        )
        print(f"  anchor: {args.prompts[0]}")
        print()
        for s, d in zip(sigmas, decoded):
            print(f"  σ={s:.2f}: {d}")


if __name__ == "__main__":
    main()

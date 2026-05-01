"""Generate monolingual alien-language corpus from a trained projector.

Source-agnostic: any embedding store matching the projector's expected shape
(N, n_source_positions, source_dim) plugs in. No English source / metadata
in the output — pure alien text, one document per line, suitable for UNMT
continue-pretraining of a target LLM (Qwen 7B-Instruct, etc).

Usage:
    poetry run python scripts/generate_synth_corpus.py CONFIG_PATH \\
        --checkpoint data/synth_contrastive_local/step5900_200k_ref.pt \\
        --embeddings data/embeddings_qwen_subset/embeddings.npy \\
        --n-docs 100 --out data/synth_corpus_smoke/v1.txt
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import torch
import yaml
from transformers import PreTrainedTokenizerFast

from lfm.agents.games.synth_contrastive import (
    SynthContrastiveGame, SynthContrastiveGameConfig,
)
from lfm.synth.backend import CausalDecoderBackend
from lfm.synth.config import SynthConfig
from lfm.synth.model import SynthLM


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("config")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--embeddings", required=True,
                   help="path to embeddings.npy memmap of (N, n_pos, dim) float16")
    p.add_argument("--n-docs", type=int, default=100)
    p.add_argument("--max-gen-len", type=int, default=None,
                   help="override config max_gen_len for longer documents")
    p.add_argument("--temperature", type=float, default=None,
                   help="override config generation_temperature")
    p.add_argument("--batch-size", type=int, default=1024,
                   help="generation is no_grad with KV cache; batch=1024 saturates "
                        "an 8GB GPU at max_gen_len=256 (~5GB peak, ~150 docs/sec)")
    p.add_argument("--out", required=True)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    log = logging.getLogger(__name__)

    game_cfg = SynthContrastiveGameConfig(**yaml.safe_load(Path(args.config).read_text()))
    if args.max_gen_len is not None:
        object.__setattr__(game_cfg, "max_gen_len", args.max_gen_len)
    if args.temperature is not None:
        object.__setattr__(game_cfg, "generation_temperature", args.temperature)

    synth_cfg = SynthConfig(**yaml.safe_load(Path(game_cfg.synth_config).read_text()))
    alien_tok = PreTrainedTokenizerFast.from_pretrained(
        str(Path(synth_cfg.output_dir) / "alien_tokenizer"),
    )
    backend = CausalDecoderBackend(
        synth_cfg.base_model_name, alien_vocab_size=len(alien_tok),
        with_reference_body=False,
    )
    synth_lm = SynthLM(backend, synth_cfg)
    raw = torch.load(game_cfg.phase1_checkpoint, map_location="cpu", weights_only=True)
    state = raw["model"] if isinstance(raw, dict) and "model" in raw else raw
    synth_lm.load_phase1_state(state)

    device = torch.device(game_cfg.device)
    synth_lm.to(device).eval()
    game = SynthContrastiveGame(game_cfg, synth_lm, synth_cfg).to(device).eval()

    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    game.load_checkpoint_state(ckpt)
    log.info("loaded projector from %s (step %s)",
             args.checkpoint, ckpt.get("step", "?"))
    log.info("max_gen_len=%d  temperature=%.2f  ngram_block=%s",
             game_cfg.max_gen_len, game_cfg.generation_temperature,
             game_cfg.ngram_block)

    emb_path = Path(args.embeddings)
    # Detect format: standard .npy or raw memmap. Both are supported so this
    # script keeps working pre- and post-build_synth_clusters.py conversion.
    with emb_path.open("rb") as fh:
        is_standard = fh.read(6).startswith(b"\x93NUMPY")
    if is_standard:
        embeddings = np.load(str(emb_path), mmap_mode="r")
    else:
        n_pos, dim = game_cfg.n_source_positions, game_cfg.source_dim
        bytes_per = n_pos * dim * 2  # float16
        if emb_path.stat().st_size % bytes_per:
            raise ValueError(
                f"{emb_path} size {emb_path.stat().st_size} not divisible by "
                f"per-sample bytes {bytes_per}"
            )
        n_total = emb_path.stat().st_size // bytes_per
        embeddings = np.memmap(emb_path, dtype=np.float16, mode="r",
                               shape=(n_total, n_pos, dim))
    n_total = int(embeddings.shape[0])
    log.info("embeddings: shape=%s n_total=%d  format=%s",
             embeddings.shape, n_total,
             "npy" if is_standard else "raw_memmap")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(args.seed)
    n_written = 0
    n_batches = (args.n_docs + args.batch_size - 1) // args.batch_size
    with torch.no_grad(), open(out_path, "w") as f:
        for batch_i in range(n_batches):
            rem = args.n_docs - n_written
            bs = min(args.batch_size, rem)
            if bs <= 0:
                break
            batch_idx = rng.integers(0, n_total, size=bs)
            anchor = torch.tensor(embeddings[batch_idx].astype(np.float32)).to(device)
            # Multi-paragraph generation: K alien expressions per anchor,
            # concatenated into one document (one line of corpus.txt).
            # K=1 reduces to legacy single-paragraph behavior.
            n_para = max(1, getattr(game_cfg, "n_paragraphs", 1))
            para_texts: list[list[str]] = [[] for _ in range(bs)]
            for k in range(n_para):
                tok_ids, valid = game.generate(anchor, paragraph_idx=k)
                token_ids = tok_ids.cpu().numpy()
                valid_mask = valid.cpu().numpy()
                for i in range(bs):
                    ids = token_ids[i][valid_mask[i]]
                    text = alien_tok.decode(ids.tolist(), skip_special_tokens=True)
                    para_texts[i].append(text.replace("\n", " ").strip())
            for i in range(bs):
                doc = " ".join(p for p in para_texts[i] if p)
                f.write(doc + "\n")
                n_written += 1
            log.info("batch %d/%d  written %d / %d",
                     batch_i + 1, n_batches, n_written, args.n_docs)

    log.info("done -> %s (%d docs)", out_path, n_written)


if __name__ == "__main__":
    main()

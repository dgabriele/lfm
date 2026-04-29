#!/usr/bin/env python3
"""Generate Qwen2.5-0.5B sentence embeddings from Leipzig English corpora.

Reads raw Leipzig tab-separated sentence files, deduplicates, and encodes
with Qwen2.5-0.5B mean-pool embeddings.

Three-stage pipeline keeps GPU saturated:
  1. Reader thread   : batches sentences from source files → text_q
  2. Tokenizer thread: tokenizes batches → tok_q (CPU tensors)
  3. GPU main loop   : forward pass + mean-pool → write_q
  4. Writer thread   : pre-allocated float16 memmap + passages.jsonl

Usage:
    poetry run python scripts/generate_qwen_embeddings.py
    poetry run python scripts/generate_qwen_embeddings.py --n-samples 1000000
"""

from __future__ import annotations

import argparse
import json
import logging
import queue
import threading
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger(__name__)

_SENTINEL = object()

DEFAULT_SOURCES = [
    "data/leipzig/eng_news_2023_1M/eng_news_2023_1M-sentences.txt",
    "data/leipzig/eng_wikipedia_2016_1M/eng_wikipedia_2016_1M-sentences.txt",
    "data/leipzig/eng_news_2020_1M/eng_news_2020_1M-sentences.txt",
    "data/leipzig/eng_news_2020_300K/eng_news_2020_300K-sentences.txt",
    "data/leipzig/eng-simple_wikipedia_2021_100K/eng-simple_wikipedia_2021_100K-sentences.txt",
]


# ── data loading ──────────────────────────────────────────────────────────────

def load_sentences(paths: list[str], n_samples: int) -> list[str]:
    """Load, deduplicate, and cap sentences from Leipzig tab-separated files.

    Reads sources in order, stopping once n_samples unique sentences are collected.
    Leipzig format: ``id<TAB>sentence text``
    """
    seen: set[str] = set()
    texts: list[str] = []
    for path in paths:
        if len(texts) >= n_samples:
            break
        before = len(texts)
        with open(path, encoding="utf-8", errors="replace") as f:
            for line in f:
                if len(texts) >= n_samples:
                    break
                parts = line.split("\t", 1)
                if len(parts) != 2:
                    continue
                text = parts[1].strip()
                if text and text not in seen:
                    seen.add(text)
                    texts.append(text)
        logger.info("loaded %s — added %d (total %d)", Path(path).parent.name, len(texts) - before, len(texts))
    return texts


# ── pooling strategies ───────────────────────────────────────────────────────

def _mean_pool(hidden: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    """(B, T, D) → (B, D)"""
    mask = attention_mask.unsqueeze(-1).to(hidden.dtype)
    return (hidden * mask).sum(1) / mask.sum(1).clamp(min=1e-9)


def _last_k_concat(
    hidden: torch.Tensor, attention_mask: torch.Tensor, k: int
) -> torch.Tensor:
    """(B, T, D) → (B, K, D). Take the last K non-pad positions per sequence.
    For sequences shorter than K, the missing leading positions are padded
    with zeros so the output retains a fixed (K, D) shape."""
    B, T, D = hidden.shape
    seq_lens = attention_mask.sum(dim=1).long()
    out = torch.zeros(B, k, D, device=hidden.device, dtype=hidden.dtype)
    for i in range(B):
        L = int(seq_lens[i])
        kk = min(k, L)
        out[i, k - kk : k] = hidden[i, L - kk : L]
    return out


# ── pipeline threads ──────────────────────────────────────────────────────────

def _reader(texts: list[str], batch_size: int, text_q: queue.Queue) -> None:
    for start in range(0, len(texts), batch_size):
        text_q.put(texts[start : start + batch_size])
    text_q.put(_SENTINEL)


def _tokenizer(model_name: str, max_len: int, text_q: queue.Queue, tok_q: queue.Queue) -> None:
    tok = AutoTokenizer.from_pretrained(model_name)
    while True:
        item = text_q.get()
        if item is _SENTINEL:
            tok_q.put(_SENTINEL)
            return
        enc = tok(item, padding=True, truncation=True, max_length=max_len, return_tensors="pt")
        tok_q.put((item, enc["input_ids"], enc["attention_mask"]))


def _writer(
    out_dir: Path, n_samples: int, embed_shape: tuple, write_q: queue.Queue
) -> None:
    """embed_shape is the per-sample shape (e.g. (D,) for mean or (K, D) for last_k_concat)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    full_shape = (n_samples, *embed_shape)
    emb_map = np.memmap(out_dir / "embeddings.npy", dtype=np.float16, mode="w+", shape=full_shape)
    written = 0
    with (out_dir / "passages.jsonl").open("w", buffering=1 << 20) as jf:
        while True:
            item = write_q.get()
            if item is _SENTINEL:
                break
            texts, embeddings = item
            end = written + len(texts)
            emb_map[written:end] = embeddings
            for t in texts:
                jf.write(json.dumps({"text": t}) + "\n")
            written = end
            if written % 50_000 == 0:
                logger.info("written %d / %d", written, n_samples)
    emb_map.flush()
    logger.info("embeddings → %s  shape=%s", out_dir / "embeddings.npy", full_shape)
    logger.info("passages   → %s  (%d lines)", out_dir / "passages.jsonl", written)


# ── main ──────────────────────────────────────────────────────────────────────

def main(args: argparse.Namespace) -> None:
    device = torch.device(args.device)

    texts = load_sentences(args.sources, args.n_samples)
    n_samples = len(texts)
    logger.info("encoding %d sentences with %s", n_samples, args.model_name)

    model = AutoModel.from_pretrained(args.model_name, torch_dtype=torch.float16).to(device).eval()
    d_model = model.config.hidden_size
    if args.pool == "mean":
        embed_shape: tuple = (d_model,)
    elif args.pool == "last_k_concat":
        embed_shape = (args.last_k, d_model)
    else:
        raise ValueError(f"unknown pool: {args.pool}")
    logger.info("pool=%s  per_sample_shape=%s  device=%s", args.pool, embed_shape, device)

    out_dir = Path(args.output_dir)
    text_q:  queue.Queue = queue.Queue(maxsize=8)
    tok_q:   queue.Queue = queue.Queue(maxsize=4)
    write_q: queue.Queue = queue.Queue(maxsize=16)

    threading.Thread(target=_reader,    args=(texts, args.batch_size, text_q),               daemon=True).start()
    threading.Thread(target=_tokenizer, args=(args.model_name, args.max_len, text_q, tok_q), daemon=True).start()
    writer_t = threading.Thread(target=_writer, args=(out_dir, n_samples, embed_shape, write_q), daemon=True)
    writer_t.start()

    processed = 0
    with torch.no_grad():
        while True:
            item = tok_q.get()
            if item is _SENTINEL:
                break
            texts_batch, input_ids, attention_mask = item
            mask = attention_mask.to(device)
            hidden = model(
                input_ids=input_ids.to(device),
                attention_mask=mask,
            ).last_hidden_state
            if args.pool == "mean":
                emb = _mean_pool(hidden, mask)
                if args.normalize:
                    emb = torch.nn.functional.normalize(emb, dim=-1)
            else:  # last_k_concat
                emb = _last_k_concat(hidden, mask, args.last_k)  # (B, K, D)
                if args.normalize:
                    # Normalise the flattened (K*D,) vector so cosine in projector
                    # input space corresponds to cosine on the concatenated source.
                    flat = emb.reshape(emb.size(0), -1)
                    flat = torch.nn.functional.normalize(flat, dim=-1)
                    emb = flat.view_as(emb)
            write_q.put((texts_batch, emb.cpu().to(torch.float16).numpy()))
            processed += len(texts_batch)

    write_q.put(_SENTINEL)
    writer_t.join()
    logger.info("done — %d embeddings written to %s", processed, out_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sources",    nargs="+", default=DEFAULT_SOURCES)
    parser.add_argument("--output-dir", default="data/embeddings_qwen")
    parser.add_argument("--model-name", default="Qwen/Qwen2.5-0.5B")
    parser.add_argument("--n-samples",  type=int, default=1_000_000)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--max-len",    type=int, default=128)
    parser.add_argument("--device",     default="cuda")
    parser.add_argument("--normalize",  action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--pool",       choices=["mean", "last_k_concat"], default="last_k_concat",
                        help="Source pooling: mean-pool (legacy) or last K hidden states concatenated.")
    parser.add_argument("--last-k",     type=int, default=8,
                        help="K for last_k_concat pooling — number of trailing positions to keep.")
    main(parser.parse_args())

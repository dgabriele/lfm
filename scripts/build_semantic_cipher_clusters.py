"""Build a semantic-similarity-aware cluster map for the WordCipher.

Each unique English word in the corpus is mapped to a cluster ID via k-means
on Qwen input-embedding vectors (mean of subword embeddings for multi-piece
words). The cluster map is saved as JSON and consumed by WordCipher to bias
the first alien syllable of each word toward a cluster-specific subset of the
syllable vocabulary, so that semantically similar English words produce alien
words that share statistical first-syllable structure.

Usage:
    poetry run python scripts/build_semantic_cipher_clusters.py \\
        --corpus data/embeddings_qwen/passages.jsonl \\
        --vocab-dir data/synth_qwen_local \\
        --output data/synth_qwen_local/word_clusters.json \\
        --n-clusters 256 --device cuda
"""

from __future__ import annotations

import argparse
import json
import logging
import re
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from sklearn.cluster import MiniBatchKMeans
from transformers import AutoModelForCausalLM, AutoTokenizer

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger(__name__)

_WORD_RE = re.compile(r"[A-Za-zÀ-ɏ]+", re.UNICODE)


def collect_vocabulary(corpus_path: Path, min_count: int) -> list[str]:
    """Extract all unique words from the corpus, lowercased, ≥min_count occurrences."""
    counts: Counter = Counter()
    if corpus_path.suffix == ".jsonl":
        for line in corpus_path.read_text().splitlines():
            if not line.strip():
                continue
            text = json.loads(line)["text"]
            for w in _WORD_RE.findall(text):
                counts[w.lower()] += 1
    else:
        for line in corpus_path.read_text().splitlines():
            for w in _WORD_RE.findall(line):
                counts[w.lower()] += 1
    return [w for w, c in counts.most_common() if c >= min_count]


@torch.no_grad()
def embed_words(words: list[str], model_name: str, device: torch.device, batch_size: int) -> np.ndarray:
    """Embed each word via the contextual last-hidden-state of the last token of
    a simple template sentence ('The {word}.'). Contextual embeddings carry far
    richer semantic structure than raw input embedding lookups."""
    tok = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.bfloat16
    ).to(device).eval()
    d = model.config.hidden_size
    out = np.zeros((len(words), d), dtype=np.float32)
    for i in range(0, len(words), batch_size):
        chunk = words[i : i + batch_size]
        prompts = [f"The {w}" for w in chunk]
        enc = tok(prompts, padding=True, return_tensors="pt").to(device)
        h = model(**enc, output_hidden_states=True).hidden_states[-1]   # (B, T, D)
        # Take the last non-pad position per sequence (the {word} subword tail).
        last_idx = (enc["attention_mask"].sum(dim=1) - 1).long()
        vecs = h[torch.arange(h.size(0), device=device), last_idx]
        out[i : i + len(chunk)] = vecs.float().cpu().numpy()
        if (i // batch_size) % 50 == 0:
            logger.info("embedded %d / %d words", min(i + batch_size, len(words)), len(words))
    return out


def cluster_words(embeddings: np.ndarray, n_clusters: int, seed: int) -> np.ndarray:
    """K-means cluster word embeddings; return cluster_id per word."""
    logger.info("k-means: n_words=%d, n_clusters=%d", embeddings.shape[0], n_clusters)
    km = MiniBatchKMeans(
        n_clusters=n_clusters, random_state=seed, batch_size=4096, n_init=5, max_iter=200,
    )
    labels = km.fit_predict(embeddings)
    sizes = Counter(labels.tolist())
    logger.info("cluster size: min=%d  median=%d  max=%d  empty=%d",
                min(sizes.values()), int(np.median(list(sizes.values()))),
                max(sizes.values()), n_clusters - len(sizes))
    return labels


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--corpus", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--model-name", default="Qwen/Qwen2.5-0.5B")
    p.add_argument("--n-clusters", type=int, default=256)
    p.add_argument("--min-count", type=int, default=2,
                   help="Drop words appearing fewer than min-count times in the corpus.")
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--device", default="cuda")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    logger.info("collecting vocabulary from %s (min_count=%d)", args.corpus, args.min_count)
    words = collect_vocabulary(args.corpus, args.min_count)
    logger.info("%d unique words ≥%d occurrences", len(words), args.min_count)

    device = torch.device(args.device)
    logger.info("embedding words via %s input embeddings on %s", args.model_name, device)
    embs = embed_words(words, args.model_name, device, args.batch_size)

    labels = cluster_words(embs, args.n_clusters, args.seed)
    word_to_cluster = {w: int(c) for w, c in zip(words, labels)}

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({
        "n_clusters": args.n_clusters,
        "model_name": args.model_name,
        "n_words": len(words),
        "seed": args.seed,
        "word_to_cluster": word_to_cluster,
    }))
    logger.info("wrote cluster map → %s  (%d words, %d clusters)",
                args.output, len(words), args.n_clusters)


if __name__ == "__main__":
    main()

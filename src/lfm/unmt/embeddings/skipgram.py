"""Skip-gram with negative sampling for monolingual BPE embeddings.

One instance of this trainer runs per language.  Unlike classical
word2vec on raw words, we train directly on the BPE unit ids produced
by the per-language sentencepiece tokenizer — the tokenizer already
handles subword decomposition, so every embedded token is exactly the
kind of unit that will appear in the Stage 3 seq2seq model's
vocabulary.

The trained embedding matrix has shape ``(V, D)`` where ``V`` is the
per-language BPE vocab size and ``D`` is ``model_dim``.  These matrices
become the inputs to the MUSE alignment step in
:mod:`lfm.unmt.embeddings.muse_align`.

Implementation notes
--------------------

* Random sampling of target/context pairs rather than a full pass
  over the corpus — the Neuroglot corpus has ~100M BPE tokens, a full
  pass is overkill for ~50M update steps.
* Unigram subsampling of frequent tokens (the Mikolov 2013 trick) to
  reduce the influence of high-frequency function words.
* Negative sampling uses a unigram distribution raised to the 3/4
  power, matching the original word2vec recipe.
* Training is GPU-first; all tensors live on the compute device.
"""

from __future__ import annotations

import logging
import math
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import sentencepiece as spm
import torch
import torch.nn as nn
import torch.nn.functional as F

from lfm.unmt.config import UNMTConfig
from lfm.unmt.tokenizer import (
    BilingualTokenizer,
    load_tokenizer,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class EmbeddingArtifacts:
    """Paths to trained per-language embedding matrices."""

    neuroglot_embeddings: Path
    english_embeddings: Path


class SkipGramModel(nn.Module):
    """Classic word2vec skip-gram with separate target + context matrices.

    At inference time the ``target`` matrix is the "the" embedding
    matrix — that is what downstream code (MUSE alignment, the Stage 3
    transformer) uses.  The ``context`` matrix only matters during
    training.
    """

    def __init__(self, vocab_size: int, embed_dim: int) -> None:
        super().__init__()
        init_scale = 0.5 / embed_dim
        self.target = nn.Embedding(vocab_size, embed_dim)
        self.context = nn.Embedding(vocab_size, embed_dim)
        nn.init.uniform_(self.target.weight, -init_scale, init_scale)
        nn.init.zeros_(self.context.weight)

    def forward(
        self,
        target_ids: torch.Tensor,
        pos_ids: torch.Tensor,
        neg_ids: torch.Tensor,
    ) -> torch.Tensor:
        """Negative sampling loss.

        Args:
            target_ids: ``(B,)`` target token ids.
            pos_ids: ``(B,)`` positive-context token ids (within window).
            neg_ids: ``(B, K)`` negative-context token ids sampled from
                the unigram distribution.
        """
        t = self.target(target_ids)                 # (B, D)
        p = self.context(pos_ids)                   # (B, D)
        n = self.context(neg_ids)                   # (B, K, D)

        pos_score = (t * p).sum(-1)                 # (B,)
        neg_score = torch.bmm(n, t.unsqueeze(-1)).squeeze(-1)  # (B, K)

        pos_loss = -F.logsigmoid(pos_score).mean()
        neg_loss = -F.logsigmoid(-neg_score).mean()
        return pos_loss + neg_loss


def _tokenize_corpus_to_ids(
    lines: list[str],
    sp: spm.SentencePieceProcessor,
) -> list[list[int]]:
    """Tokenize a corpus into per-line BPE id lists (local ids).

    Returns local sentencepiece ids, not global vocabulary offsets —
    skipgram trains on the per-language vocab directly.
    """
    return [sp.EncodeAsIds(line) for line in lines]


def _build_unigram_tables(
    sequences: list[list[int]],
    vocab_size: int,
    subsample_t: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute unigram stats for sampling and subsampling.

    Returns:
        freqs: raw token frequencies ``(V,)``.
        keep_probs: Mikolov subsampling keep-probability per token
            ``(V,)``.  Tokens with low frequency have ``keep_prob = 1``;
            frequent tokens have ``keep_prob < 1``.
    """
    freqs = np.zeros(vocab_size, dtype=np.int64)
    for seq in sequences:
        for tid in seq:
            if 0 <= tid < vocab_size:
                freqs[tid] += 1

    total = freqs.sum()
    if total == 0:
        raise RuntimeError("Empty corpus — no tokens to train on")

    relative = freqs / total
    # Avoid division by zero for unseen tokens.
    with np.errstate(divide="ignore", invalid="ignore"):
        keep_probs = np.where(
            relative > 0,
            np.minimum(1.0, np.sqrt(subsample_t / np.maximum(relative, 1e-12))),
            1.0,
        )
    return freqs, keep_probs


def _sample_targets_and_pos(
    sequences: list[list[int]],
    keep_probs: np.ndarray,
    window: int,
    batch_size: int,
    rng: random.Random,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build a training batch of (target, positive) pairs.

    Randomly selects sentences, skip-gram-samples target positions
    within each, applies subsampling, then picks a context token at a
    random offset in ``[-window, +window] \\ {0}``.
    """
    target_ids: list[int] = []
    pos_ids: list[int] = []

    while len(target_ids) < batch_size:
        seq = sequences[rng.randrange(len(sequences))]
        if len(seq) < 2:
            continue
        i = rng.randrange(len(seq))
        tid = seq[i]
        if rng.random() > keep_probs[tid]:
            continue
        # Choose an effective window size in [1, window] — classic trick
        w = rng.randint(1, window)
        lo = max(0, i - w)
        hi = min(len(seq) - 1, i + w)
        if lo == hi:
            continue
        # Pick a context position != i
        while True:
            j = rng.randint(lo, hi)
            if j != i:
                break
        pid = seq[j]
        if rng.random() > keep_probs[pid]:
            continue
        target_ids.append(tid)
        pos_ids.append(pid)

    return (
        torch.tensor(target_ids, dtype=torch.long),
        torch.tensor(pos_ids, dtype=torch.long),
    )


def _sample_negatives(
    unigram_weights: np.ndarray,
    count: int,
    rng: np.random.Generator,
) -> torch.Tensor:
    """Draw ``count`` token ids from the unigram**0.75 distribution."""
    sampled = rng.choice(len(unigram_weights), size=count, p=unigram_weights)
    return torch.tensor(sampled, dtype=torch.long)


def _train_one_language(
    sequences: list[list[int]],
    vocab_size: int,
    embed_dim: int,
    window: int,
    neg_samples: int,
    subsample_t: float,
    batch_size: int,
    total_steps: int,
    lr: float,
    device: torch.device,
    seed: int,
) -> torch.Tensor:
    """Train skip-gram on one language's tokenized corpus.

    Returns the target embedding matrix ``(V, D)`` on CPU.
    """
    py_rng = random.Random(seed)
    np_rng = np.random.default_rng(seed)

    freqs, keep_probs = _build_unigram_tables(sequences, vocab_size, subsample_t)
    # Unigram distribution raised to 3/4 for negative sampling.
    unigram = freqs.astype(np.float64) ** 0.75
    unigram_sum = unigram.sum()
    if unigram_sum <= 0:
        raise RuntimeError("Unigram distribution collapsed to zero")
    unigram /= unigram_sum

    model = SkipGramModel(vocab_size, embed_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    logger.info(
        "Skipgram: vocab=%d dim=%d window=%d neg=%d batch=%d steps=%d device=%s",
        vocab_size, embed_dim, window, neg_samples, batch_size,
        total_steps, device,
    )

    model.train()
    running_loss = 0.0
    log_every = max(1, total_steps // 50)
    import time
    start_time = time.time()
    for step in range(1, total_steps + 1):
        target_ids, pos_ids = _sample_targets_and_pos(
            sequences, keep_probs, window, batch_size, py_rng,
        )
        neg_flat = _sample_negatives(
            unigram, batch_size * neg_samples, np_rng,
        ).view(batch_size, neg_samples)

        target_ids = target_ids.to(device)
        pos_ids = pos_ids.to(device)
        neg_ids = neg_flat.to(device)

        loss = model(target_ids, pos_ids, neg_ids)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        if step % log_every == 0 or step == total_steps:
            elapsed = time.time() - start_time
            rate = step / max(elapsed, 1e-6)
            remaining = (total_steps - step) / max(rate, 1e-6)
            logger.info(
                "  step=%d/%d avg_loss=%.4f rate=%.0f steps/s ETA=%ds",
                step, total_steps,
                running_loss / log_every,
                rate, int(remaining),
            )
            running_loss = 0.0

    return model.target.weight.detach().cpu()


def train_embeddings(
    config: UNMTConfig,
    embed_dim: int | None = None,
    window: int = 5,
    neg_samples: int = 5,
    subsample_t: float = 1e-4,
    batch_size: int = 8192,
    total_steps: int = 50_000,
    lr: float = 1e-3,
    max_lines_per_language: int = 200_000,
) -> EmbeddingArtifacts:
    """Train monolingual skip-gram embeddings for both languages.

    Writes two ``.pt`` files under ``config.output_dir`` and returns
    their paths.  Each file contains a dict ``{"weights": Tensor (V, D),
    "vocab_size": int, "embed_dim": int, "lang": str}``.

    Training cost on the defaults is small enough that both languages
    complete in a few minutes on the local GPU.  For a full vast.ai
    run, bump ``total_steps`` and ``max_lines_per_language`` in the
    training config or pass larger values to this function.

    Existing embedding files are reused — rerun with ``force=True``
    semantics by deleting the file first.
    """
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    ng_path = output_dir / "embed_neuroglot.pt"
    en_path = output_dir / "embed_english.pt"

    if ng_path.exists() and en_path.exists():
        logger.info("Embedding matrices already trained:")
        logger.info("  %s", ng_path)
        logger.info("  %s", en_path)
        return EmbeddingArtifacts(
            neuroglot_embeddings=ng_path, english_embeddings=en_path,
        )

    if embed_dim is None:
        embed_dim = config.model_dim
    device = torch.device(config.device if torch.cuda.is_available() else "cpu")

    tokenizer = load_tokenizer(config)

    # Import of the raw corpus line reader to avoid pulling in the full
    # Dataset object (we want lists of ints, not DAE pair tensors).
    from lfm.unmt.data.monolingual import _read_corpus_lines

    for lang, corpus_path, sp, vocab_size, out_path in [
        (
            "ng",
            Path(config.neuroglot_corpus),
            tokenizer._ng,
            tokenizer._ng_vocab,
            ng_path,
        ),
        (
            "en",
            Path(config.english_corpus),
            tokenizer._en,
            tokenizer._en_vocab,
            en_path,
        ),
    ]:
        if out_path.exists():
            logger.info("  %s embeddings already exist at %s", lang, out_path)
            continue

        logger.info("Reading %s corpus from %s", lang, corpus_path)
        lines = _read_corpus_lines(corpus_path)
        if len(lines) > max_lines_per_language:
            # Deterministic slice, not a shuffle — the tokenizer already
            # used the head of the corpus, stay consistent.
            lines = lines[:max_lines_per_language]
        logger.info("  %d lines", len(lines))

        logger.info("Tokenizing %s corpus into BPE ids", lang)
        sequences = _tokenize_corpus_to_ids(lines, sp)
        total_tokens = sum(len(s) for s in sequences)
        logger.info("  %d BPE tokens total", total_tokens)

        weights = _train_one_language(
            sequences=sequences,
            vocab_size=vocab_size,
            embed_dim=embed_dim,
            window=window,
            neg_samples=neg_samples,
            subsample_t=subsample_t,
            batch_size=batch_size,
            total_steps=total_steps,
            lr=lr,
            device=device,
            seed=config.seed,
        )

        torch.save(
            {
                "weights": weights,
                "vocab_size": vocab_size,
                "embed_dim": embed_dim,
                "lang": lang,
            },
            out_path,
        )
        logger.info("Saved %s embeddings → %s", lang, out_path)

    return EmbeddingArtifacts(
        neuroglot_embeddings=ng_path, english_embeddings=en_path,
    )


def load_embeddings(path: Path) -> tuple[torch.Tensor, dict]:
    """Load a ``.pt`` embedding artifact and return ``(weights, metadata)``."""
    blob = torch.load(path, map_location="cpu", weights_only=False)
    metadata = {
        k: v for k, v in blob.items() if k != "weights"
    }
    return blob["weights"], metadata

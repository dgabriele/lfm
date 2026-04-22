"""Dataset builder for completeness scorer training.

Generates positive (real sentences) and negative (corrupted) examples.
Crucially, sentences with alien/replaced content words are POSITIVE —
the scorer must learn structural completeness, not vocabulary recognition.

Corruption types target the actual failure modes of diffusion decoders:
repetition loops, fragments, word salad, topic splicing.
"""

from __future__ import annotations

import json
import logging
import random
from pathlib import Path

import numpy as np
import sentencepiece as spm
import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset

logger = logging.getLogger(__name__)


def _is_content_token(token_id: int, spm_size: int) -> bool:
    """Heuristic: content tokens are SPM tokens that aren't very short pieces."""
    return 0 < token_id < spm_size


def _corrupt_shuffle(tokens: list[int], rng: random.Random) -> list[int]:
    """Shuffle word order (preserving token groups)."""
    result = list(tokens)
    if len(result) > 3:
        n_swaps = rng.randint(2, max(2, len(result) // 2))
        for _ in range(n_swaps):
            i, j = rng.sample(range(len(result)), 2)
            result[i], result[j] = result[j], result[i]
    return result


def _corrupt_repeat(tokens: list[int], rng: random.Random) -> list[int]:
    """Insert repetition loops — the main diffusion failure mode."""
    if len(tokens) < 3:
        return tokens
    # Pick a span and repeat it
    start = rng.randint(0, len(tokens) - 2)
    span_len = rng.randint(1, min(3, len(tokens) - start))
    span = tokens[start:start + span_len]
    n_repeats = rng.randint(3, 8)
    return tokens[:start] + span * n_repeats + tokens[start + span_len:]


def _corrupt_truncate(tokens: list[int], rng: random.Random) -> list[int]:
    """Truncate at a random point (incomplete idea)."""
    if len(tokens) < 4:
        return tokens
    cut = rng.randint(len(tokens) // 3, len(tokens) - 1)
    return tokens[:cut]


def _corrupt_splice(tokens_a: list[int], tokens_b: list[int], rng: random.Random) -> list[int]:
    """Splice two unrelated sentences together (topic incoherence)."""
    cut_a = rng.randint(len(tokens_a) // 3, 2 * len(tokens_a) // 3) if len(tokens_a) > 3 else len(tokens_a)
    cut_b = rng.randint(len(tokens_b) // 3, 2 * len(tokens_b) // 3) if len(tokens_b) > 3 else 0
    return tokens_a[:cut_a] + tokens_b[cut_b:]


def _corrupt_drop_function_words(
    tokens: list[int], sp: spm.SentencePieceProcessor, rng: random.Random,
) -> list[int]:
    """Drop short tokens (likely function words) — breaks grammatical glue."""
    result = []
    for t in tokens:
        piece = sp.IdToPiece(t) if 0 < t < sp.GetPieceSize() else ""
        # Short pieces (1-2 chars, excluding ▁) are likely function words
        clean = piece.replace("▁", "")
        if len(clean) <= 2 and rng.random() < 0.7:
            continue  # drop
        result.append(t)
    return result if len(result) > 2 else tokens


def _alienize_content(
    tokens: list[int], sp: spm.SentencePieceProcessor,
    rng: random.Random, alien_rate: float = 0.5,
) -> list[int]:
    """Replace content tokens with random SPM tokens (alien vocabulary).

    The resulting sentence has the same structure but unfamiliar words.
    This is a POSITIVE example — the scorer should accept it.
    """
    spm_size = sp.GetPieceSize()
    result = []
    for t in tokens:
        piece = sp.IdToPiece(t) if 0 < t < spm_size else ""
        clean = piece.replace("▁", "")
        # Replace longer tokens (content words) with random tokens
        if len(clean) > 2 and rng.random() < alien_rate:
            result.append(rng.randint(1, spm_size - 1))
        else:
            result.append(t)
    return result


def build_scorer_dataset(
    jsonl_path: str | Path,
    spm_path: str,
    output_path: str | Path,
    max_samples: int = 500_000,
    max_seq_len: int = 100,
    seed: int = 42,
) -> None:
    """Build training dataset for the completeness scorer.

    For each real sentence, generates:
    - 1 positive (original)
    - 1 positive with alien vocabulary (structural completeness preserved)
    - 2-3 negatives (various corruption types)

    Saves as a single .pt file with tokens, lengths, labels.
    """
    rng = random.Random(seed)
    sp = spm.SentencePieceProcessor()
    sp.Load(spm_path)
    spm_size = sp.GetPieceSize()

    all_tokens: list[list[int]] = []
    all_labels: list[int] = []

    # Load sentences
    logger.info("Loading sentences from %s...", jsonl_path)
    sentences: list[list[int]] = []
    with open(jsonl_path) as f:
        for i, line in enumerate(f):
            if max_samples and len(sentences) >= max_samples:
                break
            rec = json.loads(line)
            ipa = rec["ipa"]
            tids = sp.Encode(ipa, out_type=int)
            if 3 <= len(tids) <= max_seq_len:
                sentences.append(tids)

    logger.info("  %d sentences loaded", len(sentences))

    for idx, tids in enumerate(sentences):
        # Positive: original sentence
        all_tokens.append(tids)
        all_labels.append(1)

        # Positive: alien vocabulary version
        alien = _alienize_content(tids, sp, rng)
        all_tokens.append(alien)
        all_labels.append(1)

        # Negative: shuffled word order
        all_tokens.append(_corrupt_shuffle(tids, rng))
        all_labels.append(0)

        # Negative: repetition loop
        all_tokens.append(_corrupt_repeat(tids, rng))
        all_labels.append(0)

        # Negative: random truncation (50% chance)
        if rng.random() < 0.5:
            all_tokens.append(_corrupt_truncate(tids, rng))
            all_labels.append(0)

        # Negative: cross-sentence splice (50% chance)
        if rng.random() < 0.5 and idx > 0:
            other = sentences[rng.randint(0, len(sentences) - 1)]
            all_tokens.append(_corrupt_splice(tids, other, rng))
            all_labels.append(0)

        # Negative: function word dropout (50% chance)
        if rng.random() < 0.5:
            all_tokens.append(_corrupt_drop_function_words(tids, sp, rng))
            all_labels.append(0)

        if (idx + 1) % 100_000 == 0:
            logger.info("  %dK processed...", (idx + 1) // 1000)

    # Pad and save
    max_len = min(max(len(t) for t in all_tokens), max_seq_len)
    n = len(all_tokens)
    token_tensor = torch.zeros(n, max_len, dtype=torch.long)
    length_tensor = torch.zeros(n, dtype=torch.long)
    label_tensor = torch.tensor(all_labels, dtype=torch.float)

    for i, tids in enumerate(all_tokens):
        tids = tids[:max_len]
        token_tensor[i, :len(tids)] = torch.tensor(tids)
        length_tensor[i] = len(tids)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "tokens": token_tensor,
        "lengths": length_tensor,
        "labels": label_tensor,
    }, output_path)

    n_pos = label_tensor.sum().long().item()
    n_neg = n - n_pos
    logger.info(
        "Scorer dataset: %d samples (%d pos, %d neg), max_len=%d → %s",
        n, n_pos, n_neg, max_len, output_path,
    )


class ScorerDataset(Dataset):
    """Simple dataset from the saved .pt file."""

    def __init__(self, path: str | Path) -> None:
        data = torch.load(path, weights_only=True)
        self.tokens = data["tokens"]
        self.lengths = data["lengths"]
        self.labels = data["labels"]

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> dict:
        return {
            "tokens": self.tokens[idx],
            "lengths": self.lengths[idx],
            "labels": self.labels[idx],
        }

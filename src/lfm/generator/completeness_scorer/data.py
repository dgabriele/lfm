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


def _tokenize_words(
    ipa: str, sp: spm.SentencePieceProcessor,
) -> list[list[int]]:
    """Tokenize IPA text into word groups (list of token-id lists)."""
    return [sp.Encode(w, out_type=int) for w in ipa.split() if w]


def _flatten_words(word_groups: list[list[int]]) -> list[int]:
    """Flatten word groups back to a flat token list."""
    return [t for group in word_groups for t in group]


def _corrupt_word_shuffle(word_groups: list[list[int]], rng: random.Random) -> list[int]:
    """Shuffle whole words, keeping each word's tokens intact.

    This is the key corruption for detecting word salad —
    individual words are valid but their arrangement isn't.
    """
    groups = list(word_groups)
    rng.shuffle(groups)
    return _flatten_words(groups)


def _corrupt_repeat(tokens: list[int], rng: random.Random) -> list[int]:
    """Insert repetition loops — the main diffusion failure mode."""
    if len(tokens) < 3:
        return tokens
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


def _corrupt_clause_swap(word_groups: list[list[int]], rng: random.Random) -> list[int]:
    """Swap first and second half of the sentence at a word boundary."""
    if len(word_groups) < 4:
        return _flatten_words(word_groups)
    mid = len(word_groups) // 2
    jitter = rng.randint(-1, 1)
    cut = max(1, min(len(word_groups) - 1, mid + jitter))
    return _flatten_words(word_groups[cut:] + word_groups[:cut])


def _corrupt_word_insert(
    word_groups: list[list[int]], donor_groups: list[list[int]],
    rng: random.Random,
) -> list[int]:
    """Insert random words from another sentence into this one."""
    result = list(word_groups)
    n_insert = rng.randint(2, max(2, len(result) // 3))
    for _ in range(n_insert):
        if donor_groups:
            word = rng.choice(donor_groups)
            pos = rng.randint(0, len(result))
            result.insert(pos, word)
    return _flatten_words(result)


def _corrupt_splice(
    words_a: list[list[int]], words_b: list[list[int]], rng: random.Random,
) -> list[int]:
    """Splice two unrelated sentences at word boundaries (topic incoherence)."""
    cut_a = rng.randint(len(words_a) // 3, 2 * len(words_a) // 3) if len(words_a) > 3 else len(words_a)
    cut_b = rng.randint(len(words_b) // 3, 2 * len(words_b) // 3) if len(words_b) > 3 else 0
    return _flatten_words(words_a[:cut_a] + words_b[cut_b:])


def _corrupt_drop_function_words(
    word_groups: list[list[int]], sp: spm.SentencePieceProcessor,
    rng: random.Random,
) -> list[int]:
    """Drop short words (likely function words) — breaks grammatical glue."""
    result = []
    for group in word_groups:
        text = sp.Decode(group)
        if len(text.strip()) <= 3 and rng.random() < 0.7:
            continue
        result.append(group)
    return _flatten_words(result) if len(result) > 2 else _flatten_words(word_groups)


def _alienize_content(
    word_groups: list[list[int]], sp: spm.SentencePieceProcessor,
    rng: random.Random, alien_rate: float = 0.5,
) -> list[int]:
    """Replace content words with random token sequences (alien vocabulary).

    Preserves function words and word boundaries. The resulting sentence
    has the same structure but unfamiliar content words.
    This is a POSITIVE example — the scorer should accept it.
    """
    spm_size = sp.GetPieceSize()
    result = []
    for group in word_groups:
        text = sp.Decode(group)
        if len(text.strip()) > 3 and rng.random() < alien_rate:
            # Replace with random tokens of similar length
            n = max(1, len(group))
            result.append([rng.randint(1, spm_size - 1) for _ in range(n)])
        else:
            result.append(group)
    return _flatten_words(result)


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

    # Load sentences as word groups — pure IPA, word-level operations.
    logger.info("Loading IPA sentences from %s...", jsonl_path)
    sentences: list[tuple[list[int], list[list[int]]]] = []  # (flat_tokens, word_groups)
    with open(jsonl_path) as f:
        for i, line in enumerate(f):
            if max_samples and len(sentences) >= max_samples:
                break
            rec = json.loads(line)
            ipa = rec["ipa"]
            words = _tokenize_words(ipa, sp)
            flat = _flatten_words(words)
            if 3 <= len(flat) <= max_seq_len and len(words) >= 3:
                sentences.append((flat, words))

    logger.info("  %d sentences loaded", len(sentences))

    for idx, (tids, words) in enumerate(sentences):
        # Positive: original sentence
        all_tokens.append(tids)
        all_labels.append(1)

        # Positive: alien vocabulary (structure preserved, content replaced)
        all_tokens.append(_alienize_content(words, sp, rng))
        all_labels.append(1)

        # Negative: word-order shuffle (intact words, broken arrangement)
        all_tokens.append(_corrupt_word_shuffle(words, rng))
        all_labels.append(0)

        # Negative: repetition loop
        all_tokens.append(_corrupt_repeat(tids, rng))
        all_labels.append(0)

        # Negative: clause swap
        all_tokens.append(_corrupt_clause_swap(words, rng))
        all_labels.append(0)

        # Negative: random truncation (50% chance)
        if rng.random() < 0.5:
            all_tokens.append(_corrupt_truncate(tids, rng))
            all_labels.append(0)

        # Negative: word insertion from another sentence (50% chance)
        if rng.random() < 0.5 and idx > 0:
            donor = sentences[rng.randint(0, len(sentences) - 1)][1]
            all_tokens.append(_corrupt_word_insert(words, donor, rng))
            all_labels.append(0)

        # Negative: cross-sentence splice (50% chance)
        if rng.random() < 0.5 and idx > 0:
            other_words = sentences[rng.randint(0, len(sentences) - 1)][1]
            all_tokens.append(_corrupt_splice(words, other_words, rng))
            all_labels.append(0)

        # Negative: function word dropout (50% chance)
        if rng.random() < 0.5:
            all_tokens.append(_corrupt_drop_function_words(words, sp, rng))
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

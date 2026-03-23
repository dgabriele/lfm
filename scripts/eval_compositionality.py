"""Compositionality measurement for LFM's linguistic bottleneck.

Three complementary metrics:

1. **Topographic similarity (topsim)** — Spearman correlation between
   pairwise distances in meaning space and signal space (Brighton & Kirby,
   2006).  The gold-standard compositionality measure in emergent
   communication.

2. **Positional disentanglement** — For each input dimension, measure how
   much its variation is captured by a localized region of the output.
   High disentanglement means the mapping is compositional: one input
   feature → one output feature.

3. **Diagnostic probe** — Train a linear probe on generated messages to
   predict input embedding dimensions.  If the probe succeeds, the
   message retains structured input information.

Usage::

    python scripts/eval_compositionality.py
"""

from __future__ import annotations

import logging

import numpy as np
import torch
from scipy import stats
from torch import Tensor, nn

from lfm.embeddings.store import EmbeddingStore
from lfm.faculty.config import FacultyConfig
from lfm.faculty.model import LanguageFaculty
from lfm.generator.config import GeneratorConfig

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(message)s",
)
logger = logging.getLogger(__name__)


# ── Generation helpers ──────────────────────────────────────────────


def generate_messages(
    faculty: LanguageFaculty,
    embeddings: np.ndarray,
    batch_size: int,
    device: torch.device,
) -> tuple[list[list[int]], np.ndarray]:
    """Generate messages for a batch of embeddings.

    Returns:
        Tuple of (token_id_lists, hidden_state_matrix) where
        hidden_state_matrix is (N, hidden_dim) mean-pooled.
    """
    n = len(embeddings)
    all_tokens: list[list[int]] = []
    all_hidden: list[np.ndarray] = []

    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        batch = torch.tensor(
            embeddings[start:end], dtype=torch.float32, device=device,
        )

        with torch.no_grad():
            out = faculty(batch)

        tokens = out["generator.tokens"]
        mask = out["generator.mask"]
        hidden = out["generator.embeddings"]

        for i in range(end - start):
            length = int(mask[i].sum().item())
            all_tokens.append(tokens[i, :length].cpu().tolist())
            h = hidden[i, :length].mean(dim=0).cpu().numpy()
            all_hidden.append(h)

    return all_tokens, np.stack(all_hidden)


# ── Metric 1: Topographic similarity ───────────────────────────────


def topographic_similarity(
    input_embeddings: np.ndarray,
    message_tokens: list[list[int]],
    message_hidden: np.ndarray,
    num_pairs: int = 5000,
    seed: int = 42,
) -> dict[str, float]:
    """Compute topsim: correlation between meaning and signal distances.

    Uses both edit distance on token sequences and cosine distance on
    hidden states as signal-space metrics.

    Args:
        input_embeddings: (N, input_dim) embeddings.
        message_tokens: List of token id lists per sample.
        message_hidden: (N, hidden_dim) mean-pooled decoder states.
        num_pairs: Number of random pairs to sample.
        seed: Random seed.

    Returns:
        Dictionary with topsim values.
    """
    rng = np.random.default_rng(seed)
    n = len(input_embeddings)

    idx_a = rng.integers(0, n, size=num_pairs)
    idx_b = rng.integers(0, n, size=num_pairs)

    # Meaning-space distances (cosine)
    meaning_dists = np.array([
        1.0 - float(
            np.dot(input_embeddings[i], input_embeddings[j])
            / (np.linalg.norm(input_embeddings[i])
               * np.linalg.norm(input_embeddings[j]) + 1e-8)
        )
        for i, j in zip(idx_a, idx_b)
    ])

    # Signal-space distances: edit distance on token sequences
    def _edit_dist(a: list[int], b: list[int]) -> float:
        """Normalized edit distance on integer sequences."""
        if not a and not b:
            return 0.0
        la, lb = len(a), len(b)
        if la < lb:
            a, b = b, a
            la, lb = lb, la
        prev = list(range(lb + 1))
        for i in range(la):
            curr = [i + 1]
            for j in range(lb):
                cost = 0 if a[i] == b[j] else 1
                curr.append(min(curr[j] + 1, prev[j + 1] + 1, prev[j] + cost))
            prev = curr
        return prev[-1] / max(la, lb)

    token_dists = np.array([
        _edit_dist(message_tokens[i], message_tokens[j])
        for i, j in zip(idx_a, idx_b)
    ])

    # Signal-space distances: cosine on hidden states
    hidden_dists = np.array([
        1.0 - float(
            np.dot(message_hidden[i], message_hidden[j])
            / (np.linalg.norm(message_hidden[i])
               * np.linalg.norm(message_hidden[j]) + 1e-8)
        )
        for i, j in zip(idx_a, idx_b)
    ])

    # Topsim = Spearman correlation
    rho_token, p_token = stats.spearmanr(meaning_dists, token_dists)
    rho_hidden, p_hidden = stats.spearmanr(meaning_dists, hidden_dists)

    results = {
        "topsim_token_edit": float(rho_token),
        "topsim_token_edit_p": float(p_token),
        "topsim_hidden_cosine": float(rho_hidden),
        "topsim_hidden_cosine_p": float(p_hidden),
    }

    logger.info(
        "Topsim (token edit): %.4f (p=%.2e)",
        rho_token, p_token,
    )
    logger.info(
        "Topsim (hidden cosine): %.4f (p=%.2e)",
        rho_hidden, p_hidden,
    )

    return results


# ── Metric 2: Positional disentanglement ───────────────────────────


def positional_disentanglement(
    input_embeddings: np.ndarray,
    message_hidden: np.ndarray,
    num_dims: int = 20,
    seed: int = 42,
) -> dict[str, float]:
    """Measure whether individual input dimensions map to localized output regions.

    For each of the top-variance input dimensions, compute the mutual
    information (via correlation) with each output dimension.  If a single
    input dimension's information is concentrated in few output dimensions,
    the mapping is disentangled.

    Uses entropy of the squared correlation profile as a proxy: lower
    entropy = more concentrated = more disentangled.

    Args:
        input_embeddings: (N, input_dim).
        message_hidden: (N, hidden_dim) mean-pooled.
        num_dims: Number of top-variance input dims to analyze.
        seed: Random seed.

    Returns:
        Dictionary with disentanglement scores.
    """
    # Select top-variance input dimensions
    variances = input_embeddings.var(axis=0)
    top_dims = np.argsort(variances)[-num_dims:]

    entropies = []

    for d in top_dims:
        x = input_embeddings[:, d]

        # Correlation of this input dim with each hidden dim
        correlations = np.array([
            abs(float(stats.pearsonr(x, message_hidden[:, j])[0]))
            for j in range(message_hidden.shape[1])
        ])

        # Normalize to probability distribution
        total = correlations.sum()
        if total < 1e-8:
            entropies.append(np.log(message_hidden.shape[1]))  # max entropy
            continue
        probs = correlations / total

        # Shannon entropy of the correlation profile
        # Lower = more concentrated = more disentangled
        entropy = -np.sum(probs * np.log(probs + 1e-10))
        entropies.append(entropy)

    max_entropy = np.log(message_hidden.shape[1])
    mean_entropy = np.mean(entropies)

    # Disentanglement score: 1 - (mean_entropy / max_entropy)
    # 1.0 = perfectly disentangled, 0.0 = maximally entangled
    disentanglement = 1.0 - mean_entropy / max_entropy

    results = {
        "positional_disentanglement": float(disentanglement),
        "mean_correlation_entropy": float(mean_entropy),
        "max_possible_entropy": float(max_entropy),
    }

    logger.info(
        "Positional disentanglement: %.4f (entropy: %.4f / %.4f)",
        disentanglement, mean_entropy, max_entropy,
    )

    return results


# ── Metric 3: Diagnostic probe ─────────────────────────────────────


class LinearProbe(nn.Module):
    """Linear probe for predicting input features from message embeddings."""

    def __init__(self, input_dim: int, output_dim: int) -> None:
        super().__init__()
        self.linear = nn.Linear(input_dim, output_dim)

    def forward(self, x: Tensor) -> Tensor:
        return self.linear(x)


def diagnostic_probe(
    input_embeddings: np.ndarray,
    message_hidden: np.ndarray,
    train_fraction: float = 0.8,
    probe_dims: int = 50,
    epochs: int = 200,
    lr: float = 1e-3,
    batch_size: int = 256,
    device: str = "cuda",
) -> dict[str, float]:
    """Train a linear probe to predict input dimensions from message hidden states.

    If the probe achieves low MSE / high R^2, the message retains structured
    information about the input.

    Args:
        input_embeddings: (N, input_dim).
        message_hidden: (N, hidden_dim).
        train_fraction: Fraction used for training.
        probe_dims: Number of top-variance input dims to predict.
        epochs: Training epochs.
        lr: Learning rate.
        batch_size: Training batch size.
        device: Compute device.

    Returns:
        Dictionary with probe accuracy metrics.
    """
    torch_device = torch.device(device)
    n = len(input_embeddings)
    n_train = int(n * train_fraction)

    # Select top-variance target dims
    variances = input_embeddings.var(axis=0)
    target_dims = np.argsort(variances)[-probe_dims:]
    targets = input_embeddings[:, target_dims]

    # Standardize targets for meaningful MSE
    target_mean = targets[:n_train].mean(axis=0)
    target_std = targets[:n_train].std(axis=0) + 1e-8
    targets = (targets - target_mean) / target_std

    # Split
    x_train = torch.tensor(message_hidden[:n_train], dtype=torch.float32, device=torch_device)
    y_train = torch.tensor(targets[:n_train], dtype=torch.float32, device=torch_device)
    x_test = torch.tensor(message_hidden[n_train:], dtype=torch.float32, device=torch_device)
    y_test = torch.tensor(targets[n_train:], dtype=torch.float32, device=torch_device)

    # Train
    hidden_dim = message_hidden.shape[1]
    probe = LinearProbe(hidden_dim, probe_dims).to(torch_device)
    optimizer = torch.optim.Adam(probe.parameters(), lr=lr)

    for epoch in range(epochs):
        probe.train()
        perm = torch.randperm(n_train, device=torch_device)
        total_loss = 0.0
        n_batches = 0

        for start in range(0, n_train, batch_size):
            end = min(start + batch_size, n_train)
            idx = perm[start:end]
            pred = probe(x_train[idx])
            loss = ((pred - y_train[idx]) ** 2).mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            n_batches += 1

        if epoch % 50 == 0:
            logger.info("  Probe epoch %d: train_mse=%.4f", epoch, total_loss / n_batches)

    # Evaluate
    probe.eval()
    with torch.no_grad():
        pred_test = probe(x_test)
        mse = ((pred_test - y_test) ** 2).mean().item()

        # R^2 per dimension
        ss_res = ((pred_test - y_test) ** 2).sum(dim=0)
        ss_tot = ((y_test - y_test.mean(dim=0)) ** 2).sum(dim=0)
        r2_per_dim = (1.0 - ss_res / (ss_tot + 1e-8)).cpu().numpy()

        mean_r2 = float(r2_per_dim.mean())
        median_r2 = float(np.median(r2_per_dim))

    # Baseline: predicting mean
    baseline_mse = float(y_test.var().item())

    results = {
        "probe_test_mse": mse,
        "probe_baseline_mse": baseline_mse,
        "probe_mean_r2": mean_r2,
        "probe_median_r2": median_r2,
        "probe_r2_above_0": float((r2_per_dim > 0).mean()),
        "probe_r2_above_0.5": float((r2_per_dim > 0.5).mean()),
    }

    logger.info(
        "Probe: test_mse=%.4f (baseline=%.4f), mean_R²=%.4f, median_R²=%.4f",
        mse, baseline_mse, mean_r2, median_r2,
    )
    logger.info(
        "  Dims with R²>0: %.1f%%, R²>0.5: %.1f%%",
        results["probe_r2_above_0"] * 100,
        results["probe_r2_above_0.5"] * 100,
    )

    return results


# ── Main ────────────────────────────────────────────────────────────


def main(
    store_dir: str = "data/embeddings",
    decoder_path: str = "data/vae_decoder.pt",
    spm_path: str = "data/spm.model",
    num_samples: int = 2000,
    batch_size: int = 64,
    embedding_dim: int = 384,
    device: str = "cuda",
    seed: int = 42,
) -> dict[str, float]:
    """Run all compositionality metrics.

    Args:
        store_dir: Path to the embedding store.
        decoder_path: Path to pretrained VAE decoder.
        spm_path: Path to sentencepiece model.
        num_samples: Number of embeddings to generate messages for.
        batch_size: Batch size for generation.
        embedding_dim: Embedding dimensionality.
        device: Compute device.
        seed: Random seed.

    Returns:
        Combined results dictionary.
    """
    torch_device = torch.device(device)
    rng = np.random.default_rng(seed)

    # ── Load embeddings ──────────────────────────────────────────
    store = EmbeddingStore(store_dir)
    store.load()
    n = store.num_passages
    logger.info("Store: %d passages, dim=%d", n, store.embedding_dim)

    # Sample a fixed subset
    sample_idx = rng.choice(n, size=min(num_samples, n), replace=False)
    input_embeddings = store.get_embeddings(sample_idx)  # (S, dim)

    # ── Build faculty ────────────────────────────────────────────
    faculty_config = FacultyConfig(
        dim=embedding_dim,
        generator=GeneratorConfig(
            pretrained_decoder_path=decoder_path,
            spm_model_path=spm_path,
            freeze_decoder=True,
            max_output_len=32,
        ),
    )
    faculty = LanguageFaculty(faculty_config).to(torch_device)
    faculty.generator.eval()

    # Trigger lazy init
    with torch.no_grad():
        faculty(torch.randn(1, embedding_dim, device=torch_device))

    # ── Generate messages ────────────────────────────────────────
    logger.info("Generating messages for %d embeddings...", len(input_embeddings))
    message_tokens, message_hidden = generate_messages(
        faculty, input_embeddings, batch_size, torch_device,
    )
    logger.info(
        "Generated: mean_len=%.1f, hidden_dim=%d",
        np.mean([len(t) for t in message_tokens]),
        message_hidden.shape[1],
    )

    # ── Run metrics ──────────────────────────────────────────────
    results: dict[str, float] = {}

    logger.info("\n=== 1. Topographic Similarity ===")
    topsim = topographic_similarity(
        input_embeddings, message_tokens, message_hidden, seed=seed,
    )
    results.update(topsim)

    logger.info("\n=== 2. Positional Disentanglement ===")
    disent = positional_disentanglement(
        input_embeddings, message_hidden, seed=seed,
    )
    results.update(disent)

    logger.info("\n=== 3. Diagnostic Probe ===")
    probe = diagnostic_probe(
        input_embeddings, message_hidden, device=device,
    )
    results.update(probe)

    # ── Summary ──────────────────────────────────────────────────
    logger.info("\n=== Final Results ===")
    for k, v in sorted(results.items()):
        logger.info("  %s: %.6f", k, v)

    return results


if __name__ == "__main__":
    main()

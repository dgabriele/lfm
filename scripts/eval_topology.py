"""Semantic topology preservation test.

Measures whether semantically similar inputs produce linguistically similar
outputs through the frozen decoder.  High correlation between input-space
distances and message-space distances means the bottleneck preserves
semantic topology, not just identity — a key result that end-to-end
emergent codes typically fail to achieve (Chaabouni et al., NeurIPS 2019).

Three signal-space distance metrics:
1. **Edit distance** on decoded IPA strings (surface form)
2. **Phonetic distance** via token-level Jaccard (subword overlap)
3. **Hidden state cosine distance** (decoder representation)

Reports Spearman and Pearson correlations for each.

Usage::

    python scripts/eval_topology.py
"""

from __future__ import annotations

import logging

import numpy as np
import torch
from scipy import stats

from lfm.embeddings.store import EmbeddingStore
from lfm.faculty.config import FacultyConfig
from lfm.faculty.model import LanguageFaculty
from lfm.generator.config import GeneratorConfig

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(message)s",
)
logger = logging.getLogger(__name__)


# ── Distance metrics ────────────────────────────────────────────────


def levenshtein(a: str, b: str) -> int:
    """Compute Levenshtein edit distance between two strings."""
    if len(a) < len(b):
        return levenshtein(b, a)
    if not b:
        return len(a)

    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a):
        curr = [i + 1]
        for j, cb in enumerate(b):
            cost = 0 if ca == cb else 1
            curr.append(min(curr[j] + 1, prev[j + 1] + 1, prev[j] + cost))
        prev = curr
    return prev[-1]


def normalized_edit_distance(a: str, b: str) -> float:
    """Edit distance normalized by max length (0=identical, 1=maximally different)."""
    if not a and not b:
        return 0.0
    return levenshtein(a, b) / max(len(a), len(b))


def token_jaccard_distance(tokens_a: list[int], tokens_b: list[int]) -> float:
    """1 - Jaccard similarity on token multisets."""
    set_a = set(tokens_a)
    set_b = set(tokens_b)
    if not set_a and not set_b:
        return 0.0
    intersection = len(set_a & set_b)
    union = len(set_a | set_b)
    return 1.0 - intersection / union


def cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    """1 - cosine similarity."""
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 1.0
    return 1.0 - float(np.dot(a, b) / (norm_a * norm_b))


# ── Main ────────────────────────────────────────────────────────────


def main(
    store_dir: str = "data/embeddings",
    decoder_path: str = "data/vae_decoder.pt",
    spm_path: str = "data/spm.model",
    input_proj_path: str | None = None,
    num_pairs: int = 2000,
    batch_size: int = 64,
    embedding_dim: int = 384,
    device: str = "cuda",
    seed: int = 42,
) -> dict[str, float]:
    """Run the semantic topology preservation test.

    Samples random pairs of embeddings, generates messages for each,
    and measures correlation between input-space and signal-space distances.

    Args:
        store_dir: Path to the embedding store.
        decoder_path: Path to pretrained VAE decoder.
        spm_path: Path to sentencepiece model.
        input_proj_path: Path to trained input projection checkpoint.
            If None, uses random initialization (baseline).
        num_pairs: Number of embedding pairs to evaluate.
        batch_size: Batch size for generation.
        embedding_dim: Embedding dimensionality.
        device: Compute device.
        seed: Random seed.

    Returns:
        Dictionary of correlation results.
    """
    torch_device = torch.device(device)
    rng = np.random.default_rng(seed)

    # ── Load embeddings ──────────────────────────────────────────
    store = EmbeddingStore(store_dir)
    store.load()
    n = store.num_passages
    logger.info("Store: %d passages, dim=%d", n, store.embedding_dim)

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

    # Load trained input projection if provided
    if input_proj_path is not None:
        checkpoint = torch.load(input_proj_path, map_location=torch_device, weights_only=True)
        faculty.generator._input_proj.load_state_dict(checkpoint["input_proj"])
        logger.info("Loaded trained input projection from %s", input_proj_path)
    else:
        logger.info("Using random (untrained) input projection — baseline mode")

    # ── Sample pairs ─────────────────────────────────────────────
    idx_a = rng.integers(0, n, size=num_pairs)
    idx_b = rng.integers(0, n, size=num_pairs)

    emb_a = store.get_embeddings(idx_a)  # (P, dim)
    emb_b = store.get_embeddings(idx_b)  # (P, dim)

    # Input-space cosine distances
    input_distances = np.array([
        cosine_distance(emb_a[i], emb_b[i]) for i in range(num_pairs)
    ])
    logger.info(
        "Input distances: mean=%.4f, std=%.4f, min=%.4f, max=%.4f",
        input_distances.mean(), input_distances.std(),
        input_distances.min(), input_distances.max(),
    )

    # ── Generate messages in batches ─────────────────────────────
    all_ipa_a: list[str] = []
    all_ipa_b: list[str] = []
    all_tokens_a: list[list[int]] = []
    all_tokens_b: list[list[int]] = []
    all_hidden_a: list[np.ndarray] = []
    all_hidden_b: list[np.ndarray] = []

    for start in range(0, num_pairs, batch_size):
        end = min(start + batch_size, num_pairs)
        bs = end - start

        batch_a = torch.tensor(emb_a[start:end], dtype=torch.float32, device=torch_device)
        batch_b = torch.tensor(emb_b[start:end], dtype=torch.float32, device=torch_device)

        with torch.no_grad():
            out_a = faculty(batch_a)
            out_b = faculty(batch_b)

        # Decode to IPA text
        tokens_a = out_a["generator.tokens"]  # (bs, S)
        tokens_b = out_b["generator.tokens"]
        ipa_a = faculty.generator.decode_to_text(tokens_a)
        ipa_b = faculty.generator.decode_to_text(tokens_b)
        all_ipa_a.extend(ipa_a)
        all_ipa_b.extend(ipa_b)

        # Token ids (for Jaccard)
        mask_a = out_a["generator.mask"]
        mask_b = out_b["generator.mask"]
        for i in range(bs):
            len_a = int(mask_a[i].sum().item())
            len_b = int(mask_b[i].sum().item())
            all_tokens_a.append(tokens_a[i, :len_a].cpu().tolist())
            all_tokens_b.append(tokens_b[i, :len_b].cpu().tolist())

        # Mean-pooled hidden states (for cosine distance)
        hidden_a = out_a["generator.embeddings"]  # (bs, S, H)
        hidden_b = out_b["generator.embeddings"]
        for i in range(bs):
            len_a = int(mask_a[i].sum().item())
            len_b = int(mask_b[i].sum().item())
            h_a = hidden_a[i, :len_a].mean(dim=0).cpu().numpy()
            h_b = hidden_b[i, :len_b].mean(dim=0).cpu().numpy()
            all_hidden_a.append(h_a)
            all_hidden_b.append(h_b)

        if (start // batch_size) % 5 == 0:
            logger.info("Generated %d / %d pairs", end, num_pairs)

    # ── Compute signal-space distances ───────────────────────────
    edit_distances = np.array([
        normalized_edit_distance(all_ipa_a[i], all_ipa_b[i])
        for i in range(num_pairs)
    ])
    jaccard_distances = np.array([
        token_jaccard_distance(all_tokens_a[i], all_tokens_b[i])
        for i in range(num_pairs)
    ])
    hidden_distances = np.array([
        cosine_distance(all_hidden_a[i], all_hidden_b[i])
        for i in range(num_pairs)
    ])

    # ── Correlations ─────────────────────────────────────────────
    results: dict[str, float] = {}

    for name, signal_dist in [
        ("edit_distance", edit_distances),
        ("token_jaccard", jaccard_distances),
        ("hidden_cosine", hidden_distances),
    ]:
        # Spearman (rank correlation — more robust)
        rho, p_rho = stats.spearmanr(input_distances, signal_dist)
        # Pearson (linear correlation)
        r, p_r = stats.pearsonr(input_distances, signal_dist)

        results[f"{name}.spearman_rho"] = float(rho)
        results[f"{name}.spearman_p"] = float(p_rho)
        results[f"{name}.pearson_r"] = float(r)
        results[f"{name}.pearson_p"] = float(p_r)

        logger.info(
            "%s: spearman=%.4f (p=%.2e), pearson=%.4f (p=%.2e)",
            name, rho, p_rho, r, p_r,
        )

    # ── Summary statistics ───────────────────────────────────────
    for name, arr in [
        ("edit_distance", edit_distances),
        ("token_jaccard", jaccard_distances),
        ("hidden_cosine", hidden_distances),
    ]:
        results[f"{name}.mean"] = float(arr.mean())
        results[f"{name}.std"] = float(arr.std())

    # ── Example pairs ────────────────────────────────────────────
    # Show closest and farthest input pairs with their messages
    sorted_idx = np.argsort(input_distances)

    logger.info("\n=== Closest input pairs (most similar) ===")
    for rank in range(min(5, num_pairs)):
        i = sorted_idx[rank]
        logger.info(
            "  [d=%.4f] A: '%s'  B: '%s'  (edit=%.3f, hidden_cos=%.4f)",
            input_distances[i],
            all_ipa_a[i][:60], all_ipa_b[i][:60],
            edit_distances[i], hidden_distances[i],
        )

    logger.info("\n=== Farthest input pairs (most different) ===")
    for rank in range(min(5, num_pairs)):
        i = sorted_idx[-(rank + 1)]
        logger.info(
            "  [d=%.4f] A: '%s'  B: '%s'  (edit=%.3f, hidden_cos=%.4f)",
            input_distances[i],
            all_ipa_a[i][:60], all_ipa_b[i][:60],
            edit_distances[i], hidden_distances[i],
        )

    # ── Message length vs input norm ─────────────────────────────
    # Test if z-norm scaling produces correlated length variation
    all_emb = np.concatenate([emb_a, emb_b], axis=0)
    all_tokens = all_tokens_a + all_tokens_b
    norms = np.linalg.norm(all_emb, axis=1)
    lengths = np.array([len(t) for t in all_tokens])

    rho_len, p_len = stats.spearmanr(norms, lengths)
    results["norm_length.spearman_rho"] = float(rho_len)
    results["norm_length.spearman_p"] = float(p_len)
    logger.info(
        "Input norm vs message length: spearman=%.4f (p=%.2e)",
        rho_len, p_len,
    )

    logger.info("\n=== Final Results ===")
    for k, v in sorted(results.items()):
        logger.info("  %s: %.6f", k, v)

    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Semantic topology preservation test")
    parser.add_argument("--store_dir", default="data/embeddings")
    parser.add_argument("--decoder_path", default="data/vae_decoder.pt")
    parser.add_argument("--spm_path", default="data/spm.model")
    parser.add_argument("--input_proj", default=None, help="Trained input projection checkpoint")
    parser.add_argument("--num_pairs", type=int, default=2000)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    main(
        store_dir=args.store_dir,
        decoder_path=args.decoder_path,
        spm_path=args.spm_path,
        input_proj_path=args.input_proj,
        num_pairs=args.num_pairs,
        batch_size=args.batch_size,
        device=args.device,
        seed=args.seed,
    )

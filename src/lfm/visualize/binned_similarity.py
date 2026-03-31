"""Binned similarity analysis for agent game linguistic encodings.

Bins embedding store samples, encodes each through a trained agent
checkpoint, and computes pairwise similarity between bins using IPA
token representations.  Renders dendrogram + heatmap dashboard.

Metrics:
  - **Jaccard**: Set-based overlap of IPA token vocabularies per bin.
    Measures whether nearby embeddings use similar phonotactic inventories.
  - **Cosine (hidden)**: Mean cosine similarity of decoder hidden state
    centroids per bin.  Measures whether the decoder's internal
    representation preserves neighborhood structure.
  - **Edit distance**: Normalized Levenshtein distance on IPA token
    sequences.  Measures surface-form similarity of the linguistic output.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import torch

logger = logging.getLogger(__name__)


def encode_embeddings(
    faculty,
    embeddings: np.ndarray,
    device: torch.device,
    batch_size: int = 128,
) -> tuple[np.ndarray, np.ndarray]:
    """Encode all embeddings through the faculty, return tokens and hidden centroids.

    Args:
        faculty: LanguageFaculty with loaded checkpoint.
        embeddings: ``(N, dim)`` embedding array.
        device: Torch device.
        batch_size: Encoding batch size.

    Returns:
        tokens: ``(N, max_len)`` int32 token IDs (padded).
        centroids: ``(N, hidden_dim)`` mean-pooled decoder hidden states.
    """
    gen = faculty.generator
    gen.eval()

    n = embeddings.shape[0]
    all_tokens = []
    all_centroids = []

    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        batch = torch.tensor(
            embeddings[start:end], dtype=torch.float32,
        ).to(device)

        with torch.no_grad():
            out = faculty(batch)

        tok = out["generator.tokens"].cpu().numpy()
        mask = out["generator.mask"].cpu().float()
        states = out["generator.embeddings"]

        # Mean-pool hidden states per sample
        centroid = (states * mask.unsqueeze(-1).to(device)).sum(1)
        centroid = centroid / mask.sum(1, keepdim=True).to(device).clamp(min=1)
        all_centroids.append(centroid.cpu().numpy())
        all_tokens.append(tok)

        if start % (batch_size * 10) == 0 and start > 0:
            logger.info("  encoded %d / %d", start, n)

    return np.concatenate(all_tokens), np.concatenate(all_centroids)


def compute_binned_similarity(
    tokens: np.ndarray,
    centroids: np.ndarray,
    n_bins: int,
    metric: str,
) -> tuple[np.ndarray, list[tuple[int, int]]]:
    """Compute pairwise similarity matrix between consecutive bins.

    Args:
        tokens: ``(N, max_len)`` token ID array.
        centroids: ``(N, hidden_dim)`` hidden state centroids.
        n_bins: Target number of bins.
        metric: ``'jaccard'``, ``'cosine'``, or ``'edit'``.

    Returns:
        ``(n_bins, n_bins)`` similarity matrix, list of (start, end) tuples.
    """
    n = tokens.shape[0]
    bin_size = max(1, n // n_bins)
    bin_indices = []

    if metric == "jaccard":
        n_codes = int(tokens.max()) + 1
        actual_bins = (n + bin_size - 1) // bin_size
        indicators = np.zeros((actual_bins, n_codes), dtype=np.uint8)

        for b, start in enumerate(range(0, n, bin_size)):
            end = min(start + bin_size, n)
            chunk = tokens[start:end].ravel()
            indicators[b, np.unique(chunk[chunk > 0])] = 1
            bin_indices.append((start, end))

        ind_f = indicators.astype(np.float32)
        dot = ind_f @ ind_f.T
        sizes = indicators.sum(axis=1, dtype=np.float32)
        union = sizes[:, None] + sizes[None, :] - dot
        sim = np.where(union > 0, dot / union, 0.0).astype(np.float32)
        np.fill_diagonal(sim, 1.0)

    elif metric == "cosine":
        actual_bins = (n + bin_size - 1) // bin_size
        bin_centroids = np.zeros(
            (actual_bins, centroids.shape[1]), dtype=np.float32,
        )

        for b, start in enumerate(range(0, n, bin_size)):
            end = min(start + bin_size, n)
            bin_centroids[b] = centroids[start:end].mean(axis=0)
            bin_indices.append((start, end))

        # Normalize and compute cosine similarity
        norms = np.linalg.norm(bin_centroids, axis=1, keepdims=True)
        norms = np.clip(norms, 1e-8, None)
        normed = bin_centroids / norms
        sim = (normed @ normed.T).astype(np.float32)
        np.fill_diagonal(sim, 1.0)

    elif metric == "edit":
        import editdistance

        actual_bins = (n + bin_size - 1) // bin_size
        # Representative sequence per bin: median-length sample
        representatives = []
        rep_lengths = np.zeros(actual_bins, dtype=np.float32)
        for b, start in enumerate(range(0, n, bin_size)):
            end = min(start + bin_size, n)
            chunk = tokens[start:end]
            lengths = (chunk > 0).sum(axis=1)
            median_idx = np.argsort(lengths)[len(lengths) // 2]
            seq = chunk[median_idx]
            trimmed = seq[seq > 0].tolist()
            representatives.append(trimmed)
            rep_lengths[b] = max(len(trimmed), 1)
            bin_indices.append((start, end))

        # Pairwise normalized edit distance via C extension
        max_lens = np.maximum(rep_lengths[:, None], rep_lengths[None, :])
        sim = np.eye(actual_bins, dtype=np.float32)
        for i in range(actual_bins):
            for j in range(i + 1, actual_bins):
                d = editdistance.eval(representatives[i], representatives[j])
                s = 1.0 - d / max_lens[i, j]
                sim[i, j] = s
                sim[j, i] = s

    else:
        raise ValueError(f"Unknown metric: {metric!r}")

    return sim, bin_indices


_SIMILARITY_PALETTE = [
    "#08000a", "#1a0060", "#0058b8", "#008880",
    "#00b060", "#60d000", "#d8d800", "#f0c000",
    "#f08000", "#f03000", "#d00060", "#a000a0",
    "#c060d0", "#e0a8c0", "#f0e0e0", "#fffff0",
]


def _dendrogram_tier_boundaries(
    linkage_matrix: np.ndarray,
    vmin: float,
    vmax: float,
    max_levels: int = 64,
) -> np.ndarray:
    """Extract non-uniform color boundaries from dendrogram merge heights.

    Each discrete color band maps to a real merge level in the dendrogram.
    """
    merge_sims = np.sort(1.0 - linkage_matrix[:, 2])
    visible = merge_sims[(merge_sims >= vmin) & (merge_sims <= vmax)]

    if len(visible) < 2:
        return np.linspace(vmin, vmax, 9)

    span = vmax - vmin
    tol = span * 0.005
    tier_centers = [visible[0]]
    for s in visible[1:]:
        if s - tier_centers[-1] > tol:
            tier_centers.append(s)

    if len(tier_centers) > max_levels:
        indices = np.linspace(0, len(tier_centers) - 1, max_levels).astype(int)
        tier_centers = [tier_centers[i] for i in indices]

    centers = np.array(tier_centers)
    midpoints = (centers[:-1] + centers[1:]) / 2.0
    return np.concatenate([[vmin], midpoints, [vmax]])


def _build_discrete_cmap(boundaries: np.ndarray):
    """Build discrete ListedColormap + BoundaryNorm from dendrogram tiers."""
    from matplotlib.colors import BoundaryNorm, LinearSegmentedColormap, ListedColormap

    n_colors = len(boundaries) - 1
    base = _SIMILARITY_PALETTE

    if n_colors <= len(base):
        indices = np.linspace(0, len(base) - 1, n_colors).astype(int)
        colors = [base[i] for i in indices]
    else:
        continuous = LinearSegmentedColormap.from_list("_sim_cont", base, N=256)
        colors = [continuous(i / (n_colors - 1)) for i in range(n_colors)]

    cmap = ListedColormap(colors)
    norm = BoundaryNorm(boundaries, cmap.N)
    return cmap, norm


def render_binned_similarity(
    sim_matrix: np.ndarray,
    bin_indices: list[tuple[int, int]],
    metric_name: str,
    output_path: Path,
    dpi: int = 150,
) -> None:
    """Render dendrogram + reordered heatmap dashboard.

    Color levels correspond to dendrogram merge heights, normalized
    to the data range.  Uses the same palette as spinlock's binned
    similarity dashboard.
    """
    import matplotlib.pyplot as plt
    from scipy.cluster.hierarchy import dendrogram, linkage

    n = len(sim_matrix)
    distance = np.clip(1.0 - sim_matrix, 0.0, None)
    np.fill_diagonal(distance, 0)

    condensed = distance[np.triu_indices(n, k=1)]
    Z = linkage(condensed, method="ward")

    # Dendrogram for leaf ordering
    dendro = dendrogram(Z, no_plot=True)
    order = dendro["leaves"]
    reordered = sim_matrix[np.ix_(order, order)]

    # Data-driven range (percentile clipping)
    vmin = float(np.percentile(reordered, 1))
    vmax = float(np.percentile(reordered, 99))

    # Dendrogram-derived color boundaries
    boundaries = _dendrogram_tier_boundaries(Z, vmin, vmax)
    n_levels = len(boundaries) - 1
    cmap, norm = _build_discrete_cmap(boundaries)

    fig = plt.figure(figsize=(20, 16))
    ax_dendro = plt.subplot2grid((1, 6), (0, 0), rowspan=1)
    ax_heatmap = plt.subplot2grid((1, 6), (0, 1), colspan=5)

    # Dendrogram
    dendrogram(
        Z, orientation="left", ax=ax_dendro, no_labels=True,
        color_threshold=0, link_color_func=lambda k: "black",
    )
    total_samples = bin_indices[-1][1] if bin_indices else n
    ax_dendro.set_xlabel(f"{metric_name} Distance", fontsize=10)
    ax_dendro.set_title(
        f"Hierarchical Clustering\n({n:,} bins, {total_samples:,} samples)",
        fontsize=10, fontweight="bold", pad=10,
    )
    for line in ax_dendro.collections:
        line.set_linewidth(0.5)
        line.set_alpha(0.66)

    # Heatmap
    im = ax_heatmap.imshow(
        reordered, cmap=cmap, norm=norm,
        aspect="auto", interpolation="nearest",
    )
    ax_heatmap.set_xlabel("Bin index (reordered by clustering)", fontsize=10)
    ax_heatmap.set_ylabel("Bin index (reordered by clustering)", fontsize=10)

    # Colorbar with subset of ticks
    max_ticks = 12
    if len(boundaries) > max_ticks:
        tick_idx = np.linspace(0, len(boundaries) - 1, max_ticks).astype(int)
        tick_vals = boundaries[tick_idx]
    else:
        tick_vals = boundaries
    cbar = plt.colorbar(
        im, ax=ax_heatmap, fraction=0.046, pad=0.04,
        spacing="proportional", ticks=tick_vals,
    )
    cbar.ax.set_yticklabels([f"{b:.3f}" for b in tick_vals], fontsize=8)
    cbar.set_label(f"{metric_name} Similarity", fontsize=10)

    avg_sim = float(np.mean(sim_matrix[np.triu_indices(n, k=1)]))
    ax_heatmap.set_title(
        f"{metric_name} Similarity  |  avg={avg_sim:.3f}  |  "
        f"range=[{vmin:.3f}, {vmax:.3f}]  |  {n_levels} dendrogram levels",
        fontsize=10, fontweight="bold", pad=10,
    )

    # Footer
    stats = [
        f"Metric: {metric_name}",
        f"Avg Similarity: {avg_sim:.3f}",
        f"Samples: {total_samples:,}",
        f"Bins: {n:,}",
        f"~{total_samples // max(n, 1)} samples/bin",
    ]
    fig.text(
        0.5, 0.01, " | ".join(stats),
        ha="center", fontsize=11,
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8),
    )

    plt.tight_layout(rect=[0, 0.04, 1, 1])

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(output_path), dpi=dpi, bbox_inches="tight")
    logger.info("Saved %s: %s", metric_name, output_path)
    plt.close(fig)


def generate_binned_similarity(
    checkpoint_path: str,
    embedding_store_dir: str = "data/embeddings",
    decoder_path: str = "data/vae_decoder.pt",
    spm_path: str = "data/spm.model",
    num_memory_tokens: int = 8,
    output_dir: str = "output/viz",
    n_bins: int = 10000,
    metrics: list[str] | None = None,
    batch_size: int = 128,
    device: str = "cuda",
) -> list[Path]:
    """Full pipeline: load, encode, compute, render for all metrics.

    Args:
        checkpoint_path: Path to trained agent game checkpoint.
        embedding_store_dir: Path to embedding store directory.
        decoder_path: Path to pretrained decoder.
        spm_path: Path to sentencepiece model.
        num_memory_tokens: Must match decoder checkpoint.
        output_dir: Output directory for figures.
        n_bins: Number of bins.
        metrics: List of metrics to compute (default: all three).
        batch_size: Encoding batch size.
        device: Torch device.

    Returns:
        List of saved file paths.
    """
    if metrics is None:
        metrics = ["jaccard", "cosine", "edit"]

    torch_device = torch.device(device)

    # Load faculty + checkpoint
    from lfm.embeddings.store import EmbeddingStore
    from lfm.faculty.config import FacultyConfig
    from lfm.faculty.model import LanguageFaculty
    from lfm.generator.config import GeneratorConfig

    faculty = LanguageFaculty(FacultyConfig(
        dim=384,
        generator=GeneratorConfig(
            pretrained_decoder_path=decoder_path,
            spm_model_path=spm_path,
            freeze_decoder=True,
            num_memory_tokens=num_memory_tokens,
        ),
    )).to(torch_device)

    gen = faculty.generator
    gen.eval()

    # Trigger lazy init
    with torch.no_grad():
        faculty(torch.randn(1, 384, device=torch_device))

    # Load checkpoint
    ckpt = torch.load(checkpoint_path, map_location=torch_device, weights_only=False)
    if "input_proj" in ckpt:
        gen._input_proj.load_state_dict(ckpt["input_proj"])
        gen._input_refine.load_state_dict(ckpt["input_refine"])
    logger.info(
        "Loaded checkpoint (step=%s, acc=%s)",
        ckpt.get("step"), ckpt.get("accuracy"),
    )

    # Load embeddings
    store = EmbeddingStore(embedding_store_dir)
    store.load()
    logger.info("Encoding %d embeddings...", store.num_passages)

    tokens, centroids = encode_embeddings(
        faculty, store._embeddings, torch_device, batch_size,
    )
    logger.info("Encoded: tokens %s, centroids %s", tokens.shape, centroids.shape)

    # Compute + render for each metric
    metric_labels = {"jaccard": "Jaccard", "cosine": "Cosine (Hidden)", "edit": "Edit Distance"}
    out_dir = Path(output_dir)
    saved = []

    for metric in metrics:
        label = metric_labels.get(metric, metric)
        logger.info("Computing %s similarity (%d bins)...", label, n_bins)
        sim, bins = compute_binned_similarity(tokens, centroids, n_bins, metric)
        out_path = out_dir / f"binned_similarity_{metric}.webp"
        render_binned_similarity(sim, bins, label, out_path)
        saved.append(out_path)

    return saved

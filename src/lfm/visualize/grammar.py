"""Grammar analysis and visualization for emergent Neuroglot corpus.

Applies distributional analysis to discover latent grammatical structure:
word categories via PPMI + SVD + KMeans clustering, transition probabilities,
positional preferences, and compositionality metrics.

Each plot function takes a ``GrammarAnalysis`` dataclass and saves a PNG.
"""

from __future__ import annotations

import logging
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import pdist
from scipy.stats import entropy as scipy_entropy
from sklearn.cluster import KMeans
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import normalize

from lfm.visualize.style import (
    FIGSIZE_SINGLE,
    FIGSIZE_WIDE,
    apply_style,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class GrammarAnalysis:
    """All computed statistics from a grammar analysis run."""

    # Corpus
    sentences: list[list[str]]
    vocab: list[str]
    word_to_idx: dict[str, int]
    word_counts: Counter

    # Distributional vectors
    ppmi_svd_vectors: np.ndarray  # (vocab_size, n_components)

    # Clustering
    num_categories: int
    labels: np.ndarray  # per-word category label
    cluster_members: dict[int, list[str]]  # cat_id -> word list

    # Transition matrix: (num_categories, num_categories)
    transition_probs: np.ndarray
    transition_counts: np.ndarray
    category_names: list[str]  # e.g. ["C0", "C1", ...]

    # Per-category stats
    category_token_counts: np.ndarray  # total tokens per category
    category_type_counts: np.ndarray  # unique types per category

    # Positional distribution: (num_categories, 5) for quintiles
    position_distribution: np.ndarray

    # Sentence lengths
    sentence_lengths: np.ndarray

    # Mutual information (compositionality)
    position_category_mi: float


# ---------------------------------------------------------------------------
# Analyzer
# ---------------------------------------------------------------------------


class GrammarAnalyzer:
    """Distributional grammar analysis for an emergent language corpus.

    Discovers latent word categories via PPMI co-occurrence vectors reduced
    with SVD and clustered with KMeans. Computes transition probabilities,
    positional distributions, and compositionality metrics.
    """

    def __init__(
        self,
        corpus_path: str | Path,
        num_samples: int = 10_000,
        num_categories: int = 24,
        min_freq: int = 5,
        n_svd_components: int = 50,
        context_window: int = 2,
        seed: int = 42,
    ) -> None:
        self.corpus_path = Path(corpus_path)
        self.num_samples = num_samples
        self.num_categories = num_categories
        self.min_freq = min_freq
        self.n_svd_components = n_svd_components
        self.context_window = context_window
        self.seed = seed

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    def load_sentences(self) -> list[list[str]]:
        """Load corpus and return tokenized sentences.

        Each paragraph is split into sentences on '.', then each sentence
        is lowercased and split on whitespace into word tokens.
        Reads line-by-line to bound memory usage.
        """
        sentences: list[list[str]] = []
        n_para = 0
        with open(self.corpus_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                n_para += 1
                if self.num_samples and n_para > self.num_samples:
                    break
                for sent in line.split("."):
                    sent = sent.strip()
                    if not sent:
                        continue
                    tokens = sent.lower().split()
                    if tokens:
                        sentences.append(tokens)

        logger.info(
            "Loaded %d sentences from %d paragraphs", len(sentences), n_para
        )
        return sentences

    # ------------------------------------------------------------------
    # Vocabulary
    # ------------------------------------------------------------------

    def _build_vocab(
        self, sentences: list[list[str]]
    ) -> tuple[list[str], dict[str, int], Counter]:
        """Build vocabulary filtered by minimum frequency."""
        counts: Counter = Counter()
        for sent in sentences:
            counts.update(sent)

        vocab = sorted(w for w, c in counts.items() if c >= self.min_freq)
        word_to_idx = {w: i for i, w in enumerate(vocab)}
        logger.info(
            "Vocabulary: %d words (min_freq=%d, total types=%d)",
            len(vocab),
            self.min_freq,
            len(counts),
        )
        return vocab, word_to_idx, counts

    # ------------------------------------------------------------------
    # PPMI + SVD
    # ------------------------------------------------------------------

    def build_word_vectors(
        self,
        sentences: list[list[str]],
        vocab: list[str],
        word_to_idx: dict[str, int],
    ) -> np.ndarray:
        """Build PPMI co-occurrence matrix and reduce with SVD.

        Uses scipy sparse matrix to bound memory. Returns an
        (n_vocab, n_components) matrix of word vectors.
        """
        from scipy.sparse import lil_matrix

        V = len(vocab)
        cooccur = lil_matrix((V, V), dtype=np.float64)

        # Count co-occurrences within a symmetric window
        for sent in sentences:
            indices = [word_to_idx[w] for w in sent if w in word_to_idx]
            for i, wi in enumerate(indices):
                for j in range(max(0, i - self.context_window), min(len(indices), i + self.context_window + 1)):
                    if i != j:
                        cooccur[wi, indices[j]] += 1.0

        cooccur = cooccur.tocsr()

        # PPMI transformation
        total = cooccur.sum()
        if total == 0:
            logger.warning("Empty co-occurrence matrix; returning zero vectors")
            return np.zeros((V, min(self.n_svd_components, V)))

        row_sums = np.array(cooccur.sum(axis=1)).flatten()
        col_sums = np.array(cooccur.sum(axis=0)).flatten()

        # Compute PPMI on non-zero entries only (memory efficient)
        cx = cooccur.tocoo()
        data = np.array(cx.data, dtype=np.float64)
        rows, cols = cx.row, cx.col
        pmi_data = np.log2(
            (data * total) / (row_sums[rows] * col_sums[cols] + 1e-16) + 1e-16
        )
        pmi_data = np.maximum(pmi_data, 0)  # PPMI

        from scipy.sparse import csr_matrix
        ppmi = csr_matrix((pmi_data, (rows, cols)), shape=(V, V))

        # SVD reduction (works directly on sparse matrix)
        n_components = min(self.n_svd_components, V - 1)
        svd = TruncatedSVD(n_components=n_components, random_state=self.seed)
        vectors = svd.fit_transform(ppmi)
        vectors = normalize(vectors)

        explained = svd.explained_variance_ratio_.sum()
        logger.info(
            "SVD: %d components, %.1f%% variance explained",
            n_components,
            explained * 100,
        )
        return vectors

    # ------------------------------------------------------------------
    # Clustering
    # ------------------------------------------------------------------

    def cluster_words(
        self, vectors: np.ndarray, vocab: list[str]
    ) -> tuple[np.ndarray, dict[int, list[str]]]:
        """Cluster word vectors into categories with KMeans.

        Returns (labels, cluster_members) where labels is per-word and
        cluster_members maps category id to sorted word list.
        """
        n_clusters = min(self.num_categories, len(vocab))
        km = KMeans(n_clusters=n_clusters, random_state=self.seed, n_init=10)
        labels = km.fit_predict(vectors)

        cluster_members: dict[int, list[str]] = {}
        for i, word in enumerate(vocab):
            cat = int(labels[i])
            cluster_members.setdefault(cat, []).append(word)

        for cat in cluster_members:
            cluster_members[cat].sort()

        logger.info("Clustered %d words into %d categories", len(vocab), n_clusters)
        return labels, cluster_members

    # ------------------------------------------------------------------
    # Transition matrix
    # ------------------------------------------------------------------

    def _compute_transitions(
        self,
        sentences: list[list[str]],
        word_to_idx: dict[str, int],
        labels: np.ndarray,
        num_categories: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Compute bigram transition counts and probabilities between categories."""
        counts = np.zeros((num_categories, num_categories), dtype=np.float64)

        for sent in sentences:
            cats = [
                int(labels[word_to_idx[w]])
                for w in sent
                if w in word_to_idx
            ]
            for i in range(len(cats) - 1):
                counts[cats[i], cats[i + 1]] += 1.0

        row_sums = counts.sum(axis=1, keepdims=True)
        probs = np.divide(
            counts, row_sums, out=np.zeros_like(counts), where=row_sums > 0
        )
        return probs, counts

    # ------------------------------------------------------------------
    # Positional distribution
    # ------------------------------------------------------------------

    def _compute_position_distribution(
        self,
        sentences: list[list[str]],
        word_to_idx: dict[str, int],
        labels: np.ndarray,
        num_categories: int,
    ) -> np.ndarray:
        """Compute category frequency by sentence position quintile.

        Returns (num_categories, 5) array normalized per quintile.
        """
        dist = np.zeros((num_categories, 5), dtype=np.float64)

        for sent in sentences:
            n = len(sent)
            if n == 0:
                continue
            for i, w in enumerate(sent):
                if w not in word_to_idx:
                    continue
                cat = int(labels[word_to_idx[w]])
                quintile = min(int(i / n * 5), 4)
                dist[cat, quintile] += 1.0

        # Normalize per quintile (column-wise)
        col_sums = dist.sum(axis=0, keepdims=True)
        dist = np.divide(
            dist, col_sums, out=np.zeros_like(dist), where=col_sums > 0
        )
        return dist

    # ------------------------------------------------------------------
    # Compositionality (mutual information)
    # ------------------------------------------------------------------

    def _compute_position_category_mi(
        self,
        sentences: list[list[str]],
        word_to_idx: dict[str, int],
        labels: np.ndarray,
        num_categories: int,
    ) -> float:
        """Compute mutual information between position quintile and category.

        Higher MI indicates more rigid word order (positional constraints).
        """
        joint = np.zeros((5, num_categories), dtype=np.float64)

        for sent in sentences:
            n = len(sent)
            if n == 0:
                continue
            for i, w in enumerate(sent):
                if w not in word_to_idx:
                    continue
                cat = int(labels[word_to_idx[w]])
                quintile = min(int(i / n * 5), 4)
                joint[quintile, cat] += 1.0

        total = joint.sum()
        if total == 0:
            return 0.0

        p_joint = joint / total
        p_pos = p_joint.sum(axis=1)
        p_cat = p_joint.sum(axis=0)

        mi = 0.0
        for q in range(5):
            for c in range(num_categories):
                if p_joint[q, c] > 0:
                    mi += p_joint[q, c] * np.log2(
                        p_joint[q, c] / (p_pos[q] * p_cat[c] + 1e-16)
                    )
        return float(mi)

    # ------------------------------------------------------------------
    # Main
    # ------------------------------------------------------------------

    def analyze(self) -> GrammarAnalysis:
        """Run the full analysis pipeline and return results."""
        sentences = self.load_sentences()
        vocab, word_to_idx, word_counts = self._build_vocab(sentences)

        if len(vocab) < 2:
            raise ValueError(
                f"Only {len(vocab)} words meet min_freq={self.min_freq}. "
                "Lower min_freq or provide a larger corpus."
            )

        vectors = self.build_word_vectors(sentences, vocab, word_to_idx)
        labels, cluster_members = self.cluster_words(vectors, vocab)

        num_cats = int(labels.max()) + 1
        category_names = [f"C{i}" for i in range(num_cats)]

        # Transitions
        transition_probs, transition_counts = self._compute_transitions(
            sentences, word_to_idx, labels, num_cats
        )

        # Per-category type/token counts
        cat_token = np.zeros(num_cats, dtype=np.float64)
        cat_type = np.zeros(num_cats, dtype=np.float64)
        for i, word in enumerate(vocab):
            cat = int(labels[i])
            cat_token[cat] += word_counts[word]
            cat_type[cat] += 1

        # Positional distribution
        pos_dist = self._compute_position_distribution(
            sentences, word_to_idx, labels, num_cats
        )

        # Sentence lengths
        sent_lengths = np.array([len(s) for s in sentences])

        # MI
        mi = self._compute_position_category_mi(
            sentences, word_to_idx, labels, num_cats
        )

        logger.info(
            "Analysis complete: %d sentences, %d vocab, %d categories, MI=%.4f",
            len(sentences),
            len(vocab),
            num_cats,
            mi,
        )

        return GrammarAnalysis(
            sentences=sentences,
            vocab=vocab,
            word_to_idx=word_to_idx,
            word_counts=word_counts,
            ppmi_svd_vectors=vectors,
            num_categories=num_cats,
            labels=labels,
            cluster_members=cluster_members,
            transition_probs=transition_probs,
            transition_counts=transition_counts,
            category_names=category_names,
            category_token_counts=cat_token,
            category_type_counts=cat_type,
            position_distribution=pos_dist,
            sentence_lengths=sent_lengths,
            position_category_mi=mi,
        )


# ---------------------------------------------------------------------------
# Plot functions
# ---------------------------------------------------------------------------

_DPI = 150
_BBOX = "tight"


def _savefig(fig: plt.Figure, output_dir: Path, name: str) -> Path:
    """Save figure and close it. Returns the saved path."""
    path = output_dir / f"{name}.png"
    fig.savefig(path, dpi=_DPI, bbox_inches=_BBOX, facecolor="white")
    plt.close(fig)
    logger.info("Saved %s", path)
    return path


def plot_category_distribution(
    analysis: GrammarAnalysis, output_dir: Path
) -> Path:
    """Bar chart of word category frequencies (token counts)."""
    apply_style()
    fig, ax = plt.subplots(figsize=FIGSIZE_WIDE)

    order = np.argsort(-analysis.category_token_counts)
    names = [analysis.category_names[i] for i in order]
    counts = analysis.category_token_counts[order]

    ax.bar(range(len(names)), counts, color="steelblue", edgecolor="white")
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=45, ha="right", fontsize=8)
    ax.set_xlabel("Category")
    ax.set_ylabel("Token Count")
    ax.set_title("Word Category Frequency Distribution")

    return _savefig(fig, output_dir, "category_distribution")


def plot_transition_heatmap(
    analysis: GrammarAnalysis, output_dir: Path
) -> Path:
    """Category-to-category transition probability heatmap."""
    apply_style()
    n = analysis.num_categories
    fig, ax = plt.subplots(figsize=(max(8, n * 0.45), max(7, n * 0.4)))

    sns.heatmap(
        analysis.transition_probs,
        xticklabels=analysis.category_names,
        yticklabels=analysis.category_names,
        cmap="YlOrRd",
        ax=ax,
        square=True,
        linewidths=0.3,
        cbar_kws={"label": "P(next | current)"},
    )
    ax.set_xlabel("Next Category")
    ax.set_ylabel("Current Category")
    ax.set_title("Category Transition Probabilities")
    ax.tick_params(axis="both", labelsize=7)

    return _savefig(fig, output_dir, "transition_heatmap")


def plot_transition_entropy(
    analysis: GrammarAnalysis, output_dir: Path
) -> Path:
    """Horizontal bar chart: transition entropy per category.

    Low entropy categories are positionally constrained (function-word-like).
    High entropy categories have free distribution (content-word-like).
    """
    apply_style()
    fig, ax = plt.subplots(figsize=FIGSIZE_WIDE)

    entropies = np.array([
        scipy_entropy(row + 1e-16, base=2)
        for row in analysis.transition_probs
    ])
    order = np.argsort(entropies)
    names = [analysis.category_names[i] for i in order]
    ent_sorted = entropies[order]

    colors = plt.cm.RdYlGn(ent_sorted / (ent_sorted.max() + 1e-8))
    ax.barh(range(len(names)), ent_sorted, color=colors, edgecolor="white")
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=8)
    ax.set_xlabel("Transition Entropy (bits)")
    ax.set_title("Category Transition Entropy (low = constrained, high = free)")

    return _savefig(fig, output_dir, "transition_entropy")


def plot_sentence_position(
    analysis: GrammarAnalysis, output_dir: Path
) -> Path:
    """Heatmap of category frequency by sentence position quintile."""
    apply_style()
    n = analysis.num_categories
    fig, ax = plt.subplots(figsize=(8, max(6, n * 0.35)))

    quintile_labels = ["Initial", "Early", "Middle", "Late", "Final"]
    sns.heatmap(
        analysis.position_distribution,
        xticklabels=quintile_labels,
        yticklabels=analysis.category_names,
        cmap="viridis",
        ax=ax,
        linewidths=0.3,
        cbar_kws={"label": "Proportion"},
    )
    ax.set_xlabel("Sentence Position")
    ax.set_ylabel("Category")
    ax.set_title("Category Distribution by Sentence Position")
    ax.tick_params(axis="y", labelsize=7)

    return _savefig(fig, output_dir, "sentence_position")


def plot_category_dendrogram(
    analysis: GrammarAnalysis, output_dir: Path
) -> Path:
    """Hierarchical clustering of categories by transition similarity.

    Categories with similar transition profiles cluster together,
    revealing phrase-level groupings.
    """
    apply_style()
    fig, ax = plt.subplots(figsize=FIGSIZE_WIDE)

    # Use both outgoing and incoming transition profiles
    features = np.hstack([
        analysis.transition_probs,
        analysis.transition_probs.T,
    ])
    dist = pdist(features, metric="cosine")
    # Replace NaN distances (from zero rows) with max distance
    dist = np.nan_to_num(dist, nan=1.0)
    Z = linkage(dist, method="ward")

    dendrogram(
        Z,
        labels=analysis.category_names,
        ax=ax,
        leaf_rotation=45,
        leaf_font_size=9,
        color_threshold=0.7 * Z[-1, 2],
    )
    ax.set_ylabel("Distance")
    ax.set_title("Category Dendrogram (distributional similarity)")

    return _savefig(fig, output_dir, "category_dendrogram")


def plot_productivity(
    analysis: GrammarAnalysis, output_dir: Path
) -> Path:
    """Scatter plot of type count vs token count per category.

    Open classes: high type count, high token count.
    Closed classes: low type count, high token count.
    Annotated with type-token ratio (TTR).
    """
    apply_style()
    fig, ax = plt.subplots(figsize=FIGSIZE_SINGLE)

    types = analysis.category_type_counts
    tokens = analysis.category_token_counts
    ttr = np.divide(types, tokens, out=np.zeros_like(types), where=tokens > 0)

    scatter = ax.scatter(
        tokens, types, c=ttr, cmap="coolwarm", s=80, edgecolors="black", linewidth=0.5
    )
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label("Type-Token Ratio")

    for i, name in enumerate(analysis.category_names):
        ax.annotate(
            name,
            (tokens[i], types[i]),
            textcoords="offset points",
            xytext=(5, 5),
            fontsize=7,
        )

    ax.set_xlabel("Token Count")
    ax.set_ylabel("Type Count (unique words)")
    ax.set_title("Category Productivity (open vs closed classes)")
    ax.set_xscale("log")
    ax.set_yscale("log")

    return _savefig(fig, output_dir, "productivity")


def plot_sentence_length(
    analysis: GrammarAnalysis, output_dir: Path
) -> Path:
    """Histogram of sentence lengths with mean/median lines."""
    apply_style()
    fig, ax = plt.subplots(figsize=FIGSIZE_WIDE)

    lengths = analysis.sentence_lengths
    ax.hist(lengths, bins=50, color="steelblue", alpha=0.7, edgecolor="white")
    mean_len = np.mean(lengths)
    median_len = np.median(lengths)
    ax.axvline(mean_len, color="red", linestyle="--", label=f"Mean = {mean_len:.1f}")
    ax.axvline(
        median_len, color="orange", linestyle="--", label=f"Median = {median_len:.1f}"
    )
    ax.set_xlabel("Sentence Length (words)")
    ax.set_ylabel("Count")
    ax.set_title(f"Sentence Length Distribution (n = {len(lengths):,})")
    ax.legend()

    return _savefig(fig, output_dir, "sentence_length")


def plot_zipf(analysis: GrammarAnalysis, output_dir: Path) -> Path:
    """Log-log rank-frequency plot for the Neuroglot vocabulary."""
    apply_style()
    fig, ax = plt.subplots(figsize=FIGSIZE_WIDE)

    freqs_sorted = sorted(analysis.word_counts.values(), reverse=True)
    ranks = np.arange(1, len(freqs_sorted) + 1, dtype=np.float64)
    freqs = np.array(freqs_sorted, dtype=np.float64)

    ax.loglog(ranks, freqs, ".", markersize=3, color="steelblue", alpha=0.7)

    # Zipf reference (1/r)
    ref = freqs[0] / ranks
    ax.loglog(ranks, ref, "--", color="red", alpha=0.5, linewidth=1, label="Zipf (1/r)")

    # Fit exponent on top-500
    k = min(500, len(ranks))
    slope, _ = np.polyfit(np.log(ranks[:k]), np.log(freqs[:k]), 1)
    ax.set_xlabel("Rank")
    ax.set_ylabel("Frequency")
    ax.set_title(
        f"Zipf Distribution ({len(freqs_sorted):,} types, "
        f"exponent s = {-slope:.2f})"
    )
    ax.legend()

    return _savefig(fig, output_dir, "zipf")


def plot_category_network(
    analysis: GrammarAnalysis, output_dir: Path
) -> Path | None:
    """Network graph of category transitions.

    Nodes sized by frequency, edges by transition probability.
    Skipped gracefully if networkx is not available.
    """
    try:
        import networkx as nx
    except ImportError:
        logger.warning("networkx not available; skipping category network plot")
        return None

    apply_style()
    fig, ax = plt.subplots(figsize=(10, 10))

    G = nx.DiGraph()
    n = analysis.num_categories

    # Add nodes
    for i in range(n):
        G.add_node(analysis.category_names[i])

    # Add edges for significant transitions (above uniform baseline)
    threshold = 1.5 / n  # 1.5x uniform probability
    for i in range(n):
        for j in range(n):
            p = analysis.transition_probs[i, j]
            if p > threshold:
                G.add_edge(
                    analysis.category_names[i],
                    analysis.category_names[j],
                    weight=p,
                )

    if len(G.edges) == 0:
        logger.warning("No significant transitions; skipping network plot")
        plt.close(fig)
        return None

    pos = nx.spring_layout(G, seed=42, k=2.0 / np.sqrt(n))

    # Node sizes proportional to token frequency
    max_tokens = analysis.category_token_counts.max()
    node_sizes = [
        300 + 2000 * (analysis.category_token_counts[i] / max_tokens)
        for i in range(n)
    ]

    # Edge widths proportional to transition probability
    edge_weights = [G[u][v]["weight"] for u, v in G.edges()]
    max_w = max(edge_weights) if edge_weights else 1.0
    edge_widths = [0.5 + 4.0 * (w / max_w) for w in edge_weights]

    # Community coloring via greedy modularity
    try:
        undirected = G.to_undirected()
        communities = nx.community.greedy_modularity_communities(undirected)
        node_community: dict[str, int] = {}
        for ci, comm in enumerate(communities):
            for node in comm:
                node_community[node] = ci
        n_communities = len(communities)
        cmap = plt.cm.Set3
        node_colors = [
            cmap(node_community.get(analysis.category_names[i], 0) / max(n_communities, 1))
            for i in range(n)
        ]
    except Exception:
        node_colors = "steelblue"

    nx.draw_networkx_nodes(
        G, pos, ax=ax, node_size=node_sizes, node_color=node_colors,
        edgecolors="black", linewidths=0.5,
    )
    nx.draw_networkx_labels(G, pos, ax=ax, font_size=8, font_weight="bold")
    nx.draw_networkx_edges(
        G, pos, ax=ax, width=edge_widths, alpha=0.5,
        edge_color="gray", arrows=True, arrowsize=12,
        connectionstyle="arc3,rad=0.1",
    )

    ax.set_title("Category Transition Network")
    ax.axis("off")

    return _savefig(fig, output_dir, "category_network")


def plot_compositionality(
    analysis: GrammarAnalysis, output_dir: Path
) -> Path:
    """Mutual information between position quintile and category.

    Displays the global MI value and a per-category breakdown showing
    each category's contribution to the total MI.
    """
    apply_style()
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Left: global MI as annotated bar
    ax_left = axes[0]
    ax_left.bar(
        ["Position-Category MI"],
        [analysis.position_category_mi],
        color="steelblue",
        width=0.4,
    )
    ax_left.set_ylabel("Mutual Information (bits)")
    ax_left.set_title("Word Order Rigidity")
    ax_left.text(
        0,
        analysis.position_category_mi + 0.01,
        f"{analysis.position_category_mi:.4f}",
        ha="center",
        va="bottom",
        fontweight="bold",
        fontsize=12,
    )

    # Right: per-category positional entropy (how constrained each category is)
    ax_right = axes[1]
    pos_entropy = np.array([
        scipy_entropy(analysis.position_distribution[i] + 1e-16, base=2)
        for i in range(analysis.num_categories)
    ])
    max_entropy = np.log2(5)  # uniform over 5 quintiles
    normalized = pos_entropy / max_entropy

    order = np.argsort(normalized)
    names = [analysis.category_names[i] for i in order]
    vals = normalized[order]

    colors = plt.cm.RdYlGn(vals)
    ax_right.barh(range(len(names)), vals, color=colors, edgecolor="white")
    ax_right.set_yticks(range(len(names)))
    ax_right.set_yticklabels(names, fontsize=8)
    ax_right.set_xlabel("Positional Entropy / Max (0 = rigid, 1 = free)")
    ax_right.set_title("Per-Category Positional Freedom")
    ax_right.axvline(1.0, color="gray", linestyle="--", alpha=0.5)

    fig.suptitle(
        f"Compositionality Analysis (MI = {analysis.position_category_mi:.4f} bits)",
        fontsize=13,
        fontweight="bold",
    )
    fig.tight_layout()

    return _savefig(fig, output_dir, "compositionality")


# ---------------------------------------------------------------------------
# Convenience
# ---------------------------------------------------------------------------

_ALL_PLOT_FUNCS = [
    plot_category_distribution,
    plot_transition_heatmap,
    plot_transition_entropy,
    plot_sentence_position,
    plot_category_dendrogram,
    plot_productivity,
    plot_sentence_length,
    plot_zipf,
    plot_category_network,
    plot_compositionality,
]


def plot_all(analysis: GrammarAnalysis, output_dir: Path) -> list[Path]:
    """Run all grammar visualization plots and return saved paths."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    paths: list[Path] = []
    for func in _ALL_PLOT_FUNCS:
        try:
            result = func(analysis, output_dir)
            if result is not None:
                paths.append(result)
        except Exception:
            logger.exception("Failed to generate %s", func.__name__)

    return paths

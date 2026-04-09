"""Induce a PCFG from the Xenoglot corpus via distributional clustering.

1. Build word co-occurrence vectors (distributional semantics)
2. Cluster words into POS-like categories via k-means
3. Extract category sequences from sentences
4. Induce phrase structure rules from recurring category patterns
5. Estimate rule probabilities from counts

Usage:
    poetry run python scripts/induce_pcfg.py \
        --corpus data/translator/dialogue_corpus_v7_natural.txt \
        --num-samples 10000 --num-categories 24
"""

from __future__ import annotations

import argparse
import logging
import sys
from collections import Counter, defaultdict

import numpy as np
from sklearn.cluster import KMeans

logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stderr)
logger = logging.getLogger(__name__)


def load_sentences(path: str, n: int) -> list[list[str]]:
    """Load and tokenize sentences from the corpus."""
    sentences = []
    with open(path) as f:
        for i, line in enumerate(f):
            if i >= n:
                break
            line = line.strip()
            if not line:
                continue
            # Split paragraph into sentences (period-separated)
            for sent in line.split(". "):
                sent = sent.strip().rstrip(".")
                words = sent.split()
                if len(words) >= 3:
                    sentences.append(words)
    logger.info("Loaded %d sentences from %d paragraphs", len(sentences), min(i + 1, n))
    return sentences


def build_word_vectors(
    sentences: list[list[str]], min_freq: int = 5, window: int = 2,
) -> tuple[list[str], np.ndarray]:
    """Build distributional word vectors from co-occurrence counts.

    Returns (vocab, vectors) where vectors is (vocab_size, vocab_size).
    """
    # Count word frequencies
    freq = Counter()
    for sent in sentences:
        freq.update(sent)

    # Filter to frequent words
    vocab = [w for w, c in freq.most_common() if c >= min_freq]
    word2idx = {w: i for i, w in enumerate(vocab)}
    V = len(vocab)
    logger.info("Vocabulary: %d words (min_freq=%d)", V, min_freq)

    # Build co-occurrence matrix
    cooc = np.zeros((V, V), dtype=np.float32)
    for sent in sentences:
        indices = [word2idx[w] for w in sent if w in word2idx]
        for i, idx in enumerate(indices):
            for j in range(max(0, i - window), min(len(indices), i + window + 1)):
                if i != j:
                    cooc[idx, indices[j]] += 1

    # PPMI transform
    total = cooc.sum()
    row_sums = cooc.sum(axis=1, keepdims=True)
    col_sums = cooc.sum(axis=0, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        pmi = np.log2(cooc * total / (row_sums * col_sums + 1e-10) + 1e-10)
    pmi = np.maximum(pmi, 0)  # PPMI

    # SVD dimensionality reduction
    from sklearn.decomposition import TruncatedSVD
    n_dims = min(50, V - 1)
    svd = TruncatedSVD(n_components=n_dims, random_state=42)
    vectors = svd.fit_transform(pmi)
    logger.info("Word vectors: %s (SVD %d dims, %.1f%% variance)",
                vectors.shape, n_dims, svd.explained_variance_ratio_.sum() * 100)

    return vocab, vectors


def cluster_words(
    vocab: list[str], vectors: np.ndarray, n_clusters: int,
) -> tuple[np.ndarray, dict[int, list[str]]]:
    """Cluster words into POS-like categories."""
    km = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = km.fit_predict(vectors)

    clusters: dict[int, list[str]] = defaultdict(list)
    for word, label in zip(vocab, labels):
        clusters[label].append(word)

    # Sort clusters by size
    logger.info("\nWord categories (top members):")
    for cid in sorted(clusters, key=lambda c: -len(clusters[c])):
        members = clusters[cid]
        top = " ".join(members[:8])
        logger.info("  C%02d (%3d words): %s%s",
                     cid, len(members), top,
                     "..." if len(members) > 8 else "")

    return labels, clusters


def induce_rules(
    sentences: list[list[str]],
    word2cat: dict[str, int],
    n_categories: int,
) -> dict[str, list[tuple[str, float]]]:
    """Induce phrase structure rules from category sequences.

    Uses simple chunking: find recurring bigram/trigram category patterns,
    treat them as phrase rules.
    """
    # Convert sentences to category sequences
    cat_sentences = []
    for sent in sentences:
        cats = [word2cat.get(w, -1) for w in sent]
        cats = [c for c in cats if c >= 0]
        if len(cats) >= 2:
            cat_sentences.append(cats)

    # Count category bigrams and trigrams
    bigrams = Counter()
    trigrams = Counter()
    unigrams = Counter()
    for cats in cat_sentences:
        for c in cats:
            unigrams[c] += 1
        for i in range(len(cats) - 1):
            bigrams[(cats[i], cats[i + 1])] += 1
        for i in range(len(cats) - 2):
            trigrams[(cats[i], cats[i + 1], cats[i + 2])] += 1

    # Sentence-initial and sentence-final distributions
    initial = Counter(cats[0] for cats in cat_sentences)
    final = Counter(cats[-1] for cats in cat_sentences)

    total_initial = sum(initial.values())
    total_final = sum(final.values())

    # Print grammar summary
    print("\n" + "=" * 60)
    print("INDUCED GRAMMAR SUMMARY")
    print("=" * 60)

    print(f"\nSentences analyzed: {len(cat_sentences)}")
    print(f"Categories: {n_categories}")

    print("\n--- Sentence-initial categories (S → C...) ---")
    for c, count in initial.most_common(10):
        print(f"  S → C{c:02d} ...  p={count/total_initial:.3f}  ({count})")

    print("\n--- Sentence-final categories (... → ...C) ---")
    for c, count in final.most_common(10):
        print(f"  ... C{c:02d}  p={count/total_final:.3f}  ({count})")

    print("\n--- Top bigram rules (NP/VP-like phrases) ---")
    total_bi = sum(bigrams.values())
    for (a, b), count in bigrams.most_common(20):
        print(f"  C{a:02d} → C{b:02d}  p={count/total_bi:.4f}  ({count})")

    print("\n--- Top trigram rules ---")
    total_tri = sum(trigrams.values())
    for (a, b, c), count in trigrams.most_common(15):
        print(f"  C{a:02d} C{b:02d} C{c:02d}  p={count/total_tri:.4f}  ({count})")

    # Category transition matrix
    print("\n--- Category transition entropy ---")
    for c in range(n_categories):
        successors = [bigrams.get((c, c2), 0) for c2 in range(n_categories)]
        total = sum(successors)
        if total < 10:
            continue
        probs = [s / total for s in successors if s > 0]
        entropy = -sum(p * np.log2(p) for p in probs)
        print(f"  C{c:02d}: H={entropy:.2f} bits  (occurs {unigrams[c]}x)")

    # Example parsed sentences
    print("\n--- Example category sequences ---")
    for sent, cats in zip(sentences[:10], cat_sentences[:10]):
        cat_str = " ".join(f"C{c:02d}" for c in cats)
        word_str = " ".join(sent[:12])
        if len(sent) > 12:
            word_str += "..."
        print(f"  {cat_str}")
        print(f"  {word_str}")
        print()

    return {}


def main():
    parser = argparse.ArgumentParser(description="Induce PCFG from Xenoglot corpus")
    parser.add_argument("--corpus", default="data/translator/dialogue_corpus_v7_natural.txt")
    parser.add_argument("--num-samples", type=int, default=10000)
    parser.add_argument("--num-categories", type=int, default=24)
    parser.add_argument("--min-freq", type=int, default=5)
    args = parser.parse_args()

    sentences = load_sentences(args.corpus, args.num_samples)
    vocab, vectors = build_word_vectors(sentences, min_freq=args.min_freq)

    word2cat = {}
    labels, clusters = cluster_words(vocab, vectors, args.num_categories)
    for word, label in zip(vocab, labels):
        word2cat[word] = int(label)

    induce_rules(sentences, word2cat, args.num_categories)


if __name__ == "__main__":
    main()

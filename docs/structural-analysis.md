# Structural Analysis of the Pretrained Phrase Decoder

Diagnostic evidence for the structural properties of the LFM multilingual VAE phrase decoder, generated via the `lfm visualize all` CLI command. All results are from the **v7 full constituency model** trained on 11.6M IPA phrase constituents from 12 languages (eng, deu, por, rus, tur, fin, hun, kor, vie, ind, ara, hin), extracted via dep-to-constituency parsing. `max_seq_len=109`, syllable-aligned BPE (8000 vocab), latent_dim=256, 8-token z memory, 8-head multi-scale attention [3,3,7,7,15,15,full,full], weight-shared layers (2 unique x 4). Val CE=0.0082, train accuracy=99.0%. Variable-length output: mean 11.4 words, range 6-18 words. TTR=0.992, EOS rate=1.00.

---

**Contents**

1. [Latent space organization](#latent-space-organization)
2. [Attention structure](#attention-structure)
3. [Zipf's law](#zipfs-law)
4. [Latent smoothness](#latent-smoothness)
5. [Adaptive length](#adaptive-length)
6. [Compositionality](#compositionality)
7. [Latent dimensionality](#latent-dimensionality)
8. [Cross-typological interpolation](#cross-typological-interpolation)

---

## Latent space organization

The t-SNE projection of latent z vectors reveals how the decoder organizes languages across three complementary views:

**By individual language** — each of the 12 languages occupies a distinct region, with overlap between typologically related languages:

![t-SNE by language](static/images/v7_tsne_by_language.png)

**By language family** — related languages cluster together (Indo-European, Uralic, Turkic, etc.), with uniform coverage and no mode collapse:

![t-SNE by language family](static/images/v7_tsne_by_family.png)

**By morphological type** — fusional, agglutinative, isolating, and introflexive languages form broad regions reflecting shared structural properties:

![t-SNE by morphological type](static/images/v7_tsne_by_type.png)

Hierarchical clustering of per-language mean z vectors confirms linguistically sensible groupings: fusional languages cluster together, agglutinative languages form their own branch, and isolating languages separate cleanly.

![Hierarchical clustering dendrogram](static/images/v7_clustering_dendrogram.png)

The pairwise distance matrix provides a complementary view. Arabic is the most distant from other languages, consistent with its unique morphological system (root-and-pattern introflection). Fusional languages (English, German, Portuguese, Russian) show tight within-group distances.

![Pairwise distance heatmap](static/images/v7_clustering_heatmap.png)

## Attention structure

Per-head attention entropy reveals a multi-scale hierarchy matching the architectural design:

- **w=3 heads** (phonotactic): low entropy, sharply focused on local context
- **w=15 heads** (word-level): high entropy, broad attention across the sequence
- **Full-causal heads**: very low entropy, attending primarily to BOS as a z-relay token

This confirms the multi-scale attention windows function as intended — a linguistic filter bank from phoneme to clause level.

![Attention entropy by head](static/images/v7_attention_entropy.png)

## Zipf's law

Decoded token frequencies follow a Zipfian rank-frequency distribution. This is significant because emergent communication systems typically produce anti-Zipfian (uniform) distributions. The Zipfian structure here is inherited from the frozen decoder's natural language prior, providing evidence against efficient coding collapse.

![Zipf rank-frequency distribution](static/images/v7_zipf_rank_frequency.png)

## Latent smoothness

Smoothness — the property that nearby points in latent space produce similar outputs — is a prerequisite for compositional use of the latent space by downstream agents.

z distance vs. output edit distance shows moderate correlation, indicating that small latent perturbations produce proportionally small output changes at the character level.

![Smoothness: z distance vs edit distance](static/images/v7_smoothness_lipschitz.png)

z distance vs. token Jaccard similarity shows strong correlation. Nearby latent codes share most of their token vocabulary, confirming Lipschitz-like smoothness at the token level.

![Smoothness: z distance vs Jaccard similarity](static/images/v7_smoothness_jaccard.png)

Interpolation continuity curves are monotonic — intermediate latent codes produce outputs that transition continuously rather than jumping between modes.

![Smoothness: interpolation continuity](static/images/v7_smoothness_interpolation_continuity.png)

## Adaptive length

Decoded output length correlates with input complexity, and z norm correlates negatively with output uniqueness, confirming that the decoder uses variable-length encoding. The v7 full constituency model produces substantially longer output (mean 11.4 words, range 6-18) compared to the leaf-only v5 (mean 2.5 words, range 1-4).

![Adaptiveness: input vs output length](static/images/v7_adaptiveness_input_vs_output_length.png)

![Length distribution vs z norm](static/images/v7_length_dist_vs_norm.png)

## Compositionality

Diagnostic probes (linear regression from individual z dimensions to output features) show a power-law R-squared distribution. Top dimensions achieve high R-squared, while most dimensions contribute weakly. This means a small number of latent dimensions carry strong, recoverable information about the output — consistent with a compositional (rather than holistic) code.

![Compositionality: mutual information by dimension](static/images/v7_compositionality_probe_r2.png)

Positional disentanglement (whether z dimensions map to specific token positions) is low. This is expected for linguistic compositionality: natural languages encode meaning through morphology, word choice, and phrase structure — not through fixed positional slots. The probe R-squared distribution (power-law, not uniform) is the more relevant compositionality signal.

## Latent dimensionality

PCA on the latent space shows variance captured by a small number of principal components out of 256 total dimensions. The effective dimensionality is low, suggesting the decoder uses a compact manifold within the full latent space.

![Latent dimension PCA](static/images/v7_latent_dims_pca.png)

## Cross-typological interpolation

Interpolation trajectories between maximally distant language pairs (auto-selected by z-distance, cross-family) show smooth paths through the latent space:

![Interpolation trajectories in PCA space](static/images/v7_interpolation_pca_trajectories.png)

The decoded IPA text along these trajectories shows gradual cross-typological transitions — phonotactic patterns, morphological complexity, and word structure shift continuously rather than switching abruptly.

![Interpolation: decoded IPA text](static/images/v7_interpolation_decoded.png)

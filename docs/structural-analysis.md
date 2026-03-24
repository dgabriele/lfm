# Structural Analysis of the Pretrained VAE Decoder

This document presents diagnostic evidence for the structural properties of the LFM multilingual VAE decoder, generated via the `lfm visualize all` CLI command. All results are from the PoC model (val CE 0.96, PPL ~2.6, 17 epochs, 560K IPA sentences from 16 languages).

## Structural Analysis

The following results are from the visualization suite (`lfm visualize all`), which encodes the full 560K-sentence corpus through the trained VAE and generates diagnostic plots. All images are reproducible from the checkpoint.

### Latent space organization

The t-SNE projection of latent z vectors, colored by language family, shows that the decoder organizes languages typologically — related languages cluster together, with uniform coverage across the latent space and no mode collapse.

![t-SNE of latent space by language family](static/images/tsne_by_family.png)

Hierarchical clustering of per-language mean z vectors confirms linguistically sensible groupings: Slavic languages cluster together, agglutinative languages form their own branch, and isolating languages separate cleanly.

![Hierarchical clustering dendrogram](static/images/clustering_dendrogram.png)

The pairwise distance matrix provides a complementary view. Arabic is the most distant from other languages, consistent with its unique morphological system (root-and-pattern). Slavic languages (Polish, Russian, Czech) show tight within-group distances.

![Pairwise distance heatmap](static/images/clustering_heatmap.png)

### Attention structure

Per-head attention entropy reveals a multi-scale hierarchy matching the architectural design:

- **w=3 heads** (phonotactic): low entropy, sharply focused on local context
- **w=15 heads** (word-level): high entropy, broad attention across the sequence
- **Full-causal heads**: very low entropy, attending primarily to BOS as a z-relay token

This confirms the multi-scale attention windows function as intended — a linguistic filter bank from phoneme to clause level.

![Attention entropy by head](static/images/attention_entropy.png)

### Zipf's law

Decoded token frequencies follow a Zipfian rank-frequency distribution. This is significant because emergent communication systems typically produce anti-Zipfian (uniform) distributions. The Zipfian structure here is inherited from the frozen decoder's natural language prior, providing evidence against efficient coding collapse.

![Zipf rank-frequency distribution](static/images/zipf_rank_frequency.png)

### Latent smoothness

Smoothness — the property that nearby points in latent space produce similar outputs — is a prerequisite for compositional use of the latent space by downstream agents.

z distance vs. output edit distance shows moderate correlation (Spearman r=0.40), indicating that small latent perturbations produce proportionally small output changes at the character level.

![Smoothness: z distance vs edit distance](static/images/smoothness_lipschitz.png)

z distance vs. token Jaccard similarity shows strong correlation (Spearman r=0.86). Nearby latent codes share most of their token vocabulary, confirming Lipschitz-like smoothness at the token level.

![Smoothness: z distance vs Jaccard similarity](static/images/smoothness_jaccard.png)

Interpolation continuity curves are monotonic — intermediate latent codes produce outputs that transition continuously rather than jumping between modes.

![Smoothness: interpolation continuity](static/images/smoothness_interpolation_continuity.png)

### Adaptive length

Decoded output length correlates strongly with input length (r=0.947), confirming that the decoder uses variable-length encoding — more complex inputs produce longer utterances.

![Adaptiveness: input vs output length](static/images/adaptiveness_input_vs_output_length.png)

Decoded length also correlates negatively with z norm, visible in the length-vs-norm plot.

![Length distribution vs z norm](static/images/length_dist_vs_norm.png)

### Compositionality

Diagnostic probes (linear regression from individual z dimensions to output features) show a power-law R-squared distribution. Top dimensions achieve R-squared of 0.6-0.75, while most dimensions contribute weakly. This means a small number of latent dimensions carry strong, recoverable information about the output — consistent with a compositional (rather than holistic) code.

![Compositionality: mutual information by dimension](static/images/compositionality_mutual_info.png)

Notably, positional disentanglement (whether z dimensions map to specific token positions) is low. This is expected and arguably correct for linguistic compositionality: natural languages encode meaning through morphology, word choice, and phrase structure — not through fixed positional slots. A language model that assigned each latent dimension to a specific output position would be a lookup table, not a language. The probe R-squared distribution (power-law, not uniform) is the more relevant compositionality signal.

### Latent dimensionality

PCA on the latent space shows 90% of variance captured by 3 principal components and 99% by 11 PCs, out of 256 total dimensions. The effective dimensionality is low, suggesting the decoder uses a compact manifold within the full latent space.

![Latent dimension PCA](static/images/latent_dims_pca.png)

### Cross-typological interpolation

Interpolation trajectories between maximally distant language pairs (auto-selected in t-SNE space) show smooth paths through the latent space.

![Interpolation trajectories in latent space](static/images/interpolation_trajectories.png)

The decoded IPA text along these trajectories shows gradual cross-typological transitions — phonotactic patterns, morphological complexity, and word structure shift continuously rather than switching abruptly.

![Interpolation: decoded IPA text](static/images/interpolation_decoded.png)


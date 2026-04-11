"""Build dialogue-game target embeddings in a pretrained LLM's latent space.

The standard pipeline uses sentence-transformer contrastive embeddings as
the perception targets the agent must discriminate.  This package builds
an alternative target space: **hidden states from a pretrained language
model** (default: Qwen 2.5 0.5B).  The resulting
:class:`~lfm.embeddings.store.EmbeddingStore` is drop-in compatible with
the dialogue-game trainer, so the only thing that changes between this
and a baseline sentence-transformer run is which embedding store the
trainer is pointed at.

Architecture::

    CorpusSource, MixedCorpusLoader       -- configurable input text
        →  HiddenStateExtractor           -- LLM encoder + pooling
        →  DensityReweighter (optional)   -- density-aware resampling
        →  Clusterer (k-means)            -- for hard-negative sampling
        →  EmbeddingStore (reused as-is)

Density-aware resampling is built in from the start, because raw LLM
hidden states for any natural corpus over-sample its dense topics.  If
we want the agent's Neuroglot to cover the breadth of the reader's
latent space — rather than just the corpus's topic histogram —
reweighting toward sparser regions is the right default.
"""

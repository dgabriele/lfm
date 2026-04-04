# LFM Roadmap

Planned improvements and research directions, ordered by expected impact and novelty.

## Recently Implemented

- **Multi-token z memory** — z projected into K=8 memory tokens for decoder cross-attention, spreading z influence across all output positions (previously concentrated at positions 55+)
- **Syllable-aligned BPE** — syllabify IPA before sentencepiece training so BPE merges cannot cross syllable boundaries; 93% → ~100% syllable alignment by construction

---

**Contents**

1. [Constituency-labeled expression leaves](#1-constituency-labeled-expression-leaves)
2. [Bidirectional translation and LLM grounding](#2-bidirectional-translation-and-llm-grounding)
3. [Cross-leaf context during generation](#3-cross-leaf-context-during-generation)
4. [Multi-agent self-play](#4-multi-agent-self-play)
5. [Emotional tone and subjective state expression](#5-emotional-tone-and-subjective-state-expression)
6. [Decoder fine-tuning on emergent language](#6-decoder-fine-tuning-on-emergent-language)
7. [Constituency-augmented decoder training](#7-constituency-augmented-decoder-training)
8. [Phonetic feature-aware decoding](#8-phonetic-feature-aware-decoding)
9. [Non-human acoustic data](#9-non-human-acoustic-data)

---

## 1. Constituency-labeled expression leaves

**Status:** Planned — highest-impact architectural extension

The current expression system produces full sentences per leaf, with tree topology learned via REINFORCE. The next step: each leaf carries a **constituency label** (NP, VP, PP, S) embedded alongside z in the decoder's cross-attention memory. The decoder learns to produce phrase-type-specific output: a leaf labeled NP produces a noun phrase and stops; a VP leaf produces a verb phrase.

**Architecture:** Add a learned label embedding (e.g., 8 labels × 512 dims) to the memory vector. During decoding, `memory = latent_to_decoder(z) + label_embedding(label)`. The ExpressionGenerator's expand/leaf head gains a categorical label prediction per leaf.

**What this enables:** The expression tree becomes a **syntactic blueprint** — left branches are NPs, right branches are VPs, depth corresponds to clause embedding depth. The tree isn't just a decomposition; it's a grammar. And the decoder fills in each node according to its label, producing linguistically precise constituents.

**Requires:** Constituency-augmented training data (infrastructure exists), architectural change (label embedding in memory), REINFORCE over categorical label actions.

**Why first:** This is the difference between "agents decompose meaning into parts" and "agents produce syntactically structured language." The second is a much stronger scientific claim and more publishable result.

---

## 2. Bidirectional translation and LLM grounding

**Status:** Forward direction implemented (IPA→English), reverse not yet

The existing `lfm translate` pipeline translates alien IPA to English. Training the same LLM on the reverse direction (English→alien IPA) closes the full communication loop: a human writes English, the LLM translates to the alien language, and the agent receives it.

**What this enables:** Bidirectional human-agent dialogue mediated by the emergent language. The LLM becomes an interpreter fluent in both languages. And because the alien language encodes empirical grounding from its source domain, the LLM acquires that grounding through learning the language.

**PhonologyBench:** The framework includes a phonology benchmark (`lfm translate eval-phonology`) to test whether the translator LLM develops genuine phonological competence — syllable counting, rhyme detection, minimal pair discrimination — in the alien language. This distinguishes "the LLM learned the language" from "the LLM memorized (IPA, English) pairs."

---

## 3. Cross-leaf context during generation

**Status:** Not yet implemented

Each leaf's z is generated independently by the `ExpressionGenerator` before decoding begins. The tree topology is decided top-down, and leaf z vectors are projected from their local hidden context — with no knowledge of what other leaves will produce.

A richer approach: condition later leaf z vectors on earlier leaves' decoded output. After the first leaf is decoded (via the continuous AR pass), its hidden states feed back into the tree generator to inform subsequent leaf z projections. This lets the agent build coherent multi-sentence expressions where later sentences elaborate on or respond to earlier ones.

This is naturally compatible with the continuous z-switching architecture — the KV cache already carries context from earlier leaves. The missing piece is feeding that context back to the z-generation stage.

---

## 4. Multi-agent self-play

**Status:** Not started

The current referential game has one sender and one receiver with fixed roles. Multi-agent self-play would have agents alternate roles, developing shared conventions. Different agents might converge on different expression grammars for the same referential task — different tree topologies, different z-space dialects — and the diversity of emergent grammars is itself an interesting research output.

With persistent agent state across rounds, agents develop "accents" — consistent stylistic signatures in their expression patterns. Whether these signatures are arbitrary conventions or functionally grounded in the agent's state history is an empirical question that multi-agent self-play enables.

---

## 5. Emotional tone and subjective state expression

**Status:** Speculative research direction

Train the decoder on emotionally valenced first-person corpora — angry, calm, kind, neutral variations of the same content. This creates a **tonal manifold** in z-space: regions whose phonological patterns correlate with emotional categories.

**The scientifically interesting test:** Give agents persistent state across communication rounds, then observe whether their expressions drift into emotion-correlated z regions when internal state changes (high loss, adversarial pressure, prediction error) — *without* being trained on emotional labels.

**Why it matters:** This is a controlled test of the embodied cognition hypothesis applied to language. Cognitive science claims emotional language is grounded in bodily states. LFM tests the inverse: does emotion-correlated language emerge from a linguistic bottleneck coupled to internal dynamics, *without* a body? The answer is an empirical contribution to a philosophical debate, with every variable controlled and measurable.

**Measurement:** Mutual information between agent persistent state and expression structural features (tree topology, z-vector distribution, phonotactic patterns), computed entirely in z-space with zero reference to English.

---

## 6. Decoder fine-tuning on emergent language

**Status:** Not started

After agents converge on a communication protocol, the decoder could be fine-tuned on the agent's actual z→output distribution (rather than the original encoder's). This would sharpen the decoder's output for the specific z region the agent uses, potentially improving reconstruction quality and EOS behavior without full retraining.

The risk: fine-tuning on a narrow z distribution might degrade the decoder's ability to handle the full manifold. A compromise: fine-tune with a mix of agent z samples and original encoder z samples.

---

## 7. Constituency-augmented decoder training

**Status:** Infrastructure built, deprioritized for PoC

Train the decoder on both full sentences and extracted phrase constituents. The encoder sees the full parent sentence (providing rich z context), while the decoder is supervised only on the constituent span. This teaches variable-length EOS behavior.

**Deprioritized because:** The expression system's continuous z-switching decode handles variable-length output architecturally — each leaf produces a full sentence to EOS, and the tree topology determines how many sentences the agent produces. Sub-sentence constituency training is not required for the PoC. The full infrastructure is built (dataset generation, `ConstituentDataset`, `InterleavedLoader`, multi-backend parsing) and can be activated when constituency-labeled leaves (item 1) are implemented.

**Relation to item 1:** This provides the training signal for constituency-labeled leaves. With the decoder trained to produce labeled constituents on demand, the expression tree becomes a syntactic blueprint rather than just a multi-sentence decomposition.

---

## 8. Phonetic feature-aware decoding

**Status:** Panphon integration built, not yet used in decoding

The phonetic embedding module (`generator/phonetic_embeddings.py`) maps BPE tokens to articulatory feature vectors via panphon. This could bias the decoder toward phonetically coherent output at decode time — when uncertain between two tokens, prefer the one that maintains articulatory continuity with the preceding context (coarticulation).

---

## 9. Non-human acoustic data (whale, bird song)

**Status:** Speculative

Map non-human vocalizations to IPA via acoustic-to-articulatory inversion: click trains → plosives, tonal sweeps → vowel formant transitions, frequency bands → place of articulation. The transcription would be lossy but systematic.

The decoder, trained on human + whale IPA, would learn "whale phonotactics" alongside human phonotactics. The expression system could then generate whale-like phrases when encoding marine biology data.

**Risk:** Mixing radically different phonotactic systems may degrade human language quality. Safer approach: train on human languages first, then test whether the frozen decoder can decode whale-mapped IPA at all. If it produces valid output, the mapping is compatible. If not, the decoder rejects it — useful diagnostic information.

---

## 10. Diffusion-based iterative token refinement

**Status:** Planned — v2 architecture for scale

Replace the sequential AR decoder bottleneck with a parallel diffusion refinement step. The AR decoder runs once (no_grad) to produce a "rough draft" of tokens, then a lightweight diffusion denoiser refines them in parallel for discriminative quality.

**Architecture:**
```
z vectors → AR decoder (one pass, rough draft) → token embeddings
  → add noise → diffusion denoiser (T=4 steps, cross-attends to z) → refined embeddings
  → snap to nearest vocabulary token
```

**Why this works:** The AR decoder provides linguistic structure (phonotactics, morphology, Zipf law). The diffusion refinement adjusts tokens for discriminative quality without rebuilding linguistic competence. The denoiser is small (correcting, not generating) and fully parallel.

**What it solves:**
- GPU utilization: current pipeline is ~50% idle during sequential AR decode
- Token quality: AR decode is greedy/nucleus sampled, diffusion refinement can optimize for downstream objectives
- Scale: at 500+ token expressions with 100K+ corpus, sequential decode becomes the training bottleneck

**Bidirectional path:** The refinement diffusion supports natural reversal — given observed IPA tokens, add noise and denoise conditioned on nothing to recover the z vectors. This is the IPA→z reconstruction path for bidirectional communication via LLM.

**When to implement:** After the current architecture achieves successful LLM translation training. The current pipeline (diffusion z-generator → AR decode → IPA-to-IPA receiver with topology loss) is sufficient for the 10K-300K scale. Option D becomes compelling for production-scale deployment.

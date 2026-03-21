# LFM — Language Faculty Model

A framework for giving neural agents a natural language faculty.

LFM is a learnable system that imposes morphosyntactic and sentence-level constraints on sequences, enabling agents to express internal representations in structured, compositional form — without encoding predefined semantics. It models the *faculty* of language, not any particular human language.

---

**Contents**

1. [Vision](#vision)
2. [The Problem](#the-problem)
3. [The Pipeline](#the-pipeline)
4. [Approach to Phonetics](#approach-to-phonetics)
5. [Agent Games](#proof-of-concept-agent-games-for-development)
6. [Design](#design)
7. [Quick Start](#quick-start)
8. [Status](#status)

---

## Vision

Agents embedded in complex physical systems — fluid dynamics, biological networks, markets, high-dimensional parameter spaces — develop internal representations that encode perspectives no human scientist has access to. These representations are empirical, grounded in real dynamics, but they are also subjective: shaped by the agent's particular vantage point, attention, and history within the system.

LFM gives those agents a language. Not English, not mathematics — a new language with its own morphology, syntax, and phonology, whose structure emerges from the pressure to communicate about what the agent has observed. The language is alien but *regular* — it has the same structural inductive biases as human natural languages, which means a pretrained multilingual LLM can learn to translate it, the same way it would learn any new language.

The goal is to synthesize consistent, structured, non-human natural language corpora — the output of agents reasoning over dynamical systems in their own terms — and then tune in. Listen to alien researchers' inner monologues and conversations about systems we study, from perspectives that are grounded but fundamentally outside our own collective scientific trajectory. Perspectives not isomorphic to our normal mathematical, symbolic, or linguistic categories. Perspectives that might see structure where we see noise, or draw distinctions where we see uniformity.

This is not metaphorical. The pipeline is concrete: a VQ tokenizer grounds agent representations in physical dynamics with round-trip consistency. Agents attend to and reason over these tokens. LFM structures their communication as language. An LLM translates it. At every step, the information is empirically grounded and the fidelity is measurable. What's new is that the resulting descriptions are irreducibly perspectival — they reflect what a situated observer found salient, not what an equation says is true.

## The Problem

Agents that operate over grounded, potentially non-human representations need to communicate. Existing approaches have problems that collectively motivate the creation of LFM:

- **Natural language** imposes human ontology and semantic bias
- **Latent vector communication** lacks structure and interpretability
- **Emergent protocols** tend to collapse into degenerate, non-compositional codes
- **Symbolic systems** are rigid and not adaptive

LFM sits between the agent's internal world model and its communication channel, shaping messages to be compositional, structurally regular, reusable, and adaptable — while letting semantics emerge from interaction rather than being inherited from human language.

### Translation, not alignment

The emergent language that LFM produces is not human language — but it is *language-like*. It has morphology, syntax, phrase structure, and phonotactically pronounceable surface forms. This is by design: the structural inductive biases that shape it are drawn from the same typological universals that underlie all human languages.

This means the emergent language is readily learnable by pretrained multilingual LLMs. An LLM that already understands the structural patterns of hundreds of human languages — agreement, word order, morphological case, inflection — can learn to translate LFM's emergent language through self-supervised fine-tuning, without any hand-crafted parallel corpus. The LLM acts as a translator: agents communicate in their own language, and humans read the translation.

Crucially, this is translation, not latent space alignment. Alignment methods map agent representations into a human language embedding space, collapsing the agent's natural semantics onto human categories and destroying whatever novel perspective the agent may have developed. Translation preserves the source language's own conceptual structure and finds the best human-language approximation. The agent's ontology stays intact; the LLM does the interpretive work, the same way a human translator mediates between languages with fundamentally different conceptual systems.

The alternative — agents communicating in raw latent vectors or degenerate codes — gives an LLM nothing to work with. Structure is what makes translation possible, and language-like structure is what makes it learnable.

## The Pipeline

LFM provides a configurable pipeline of neural modules:

### Quantization

The entry point. An agent's continuous internal state — a dense vector encoding whatever it has observed, inferred, or decided — must become discrete before it can be language. The quantizer maps this continuous representation into a sequence of discrete tokens drawn from a learned codebook. This is analogous to the transition from pre-linguistic thought to the discrete units of speech. Multiple quantization strategies are supported (VQ-VAE, Finite Scalar Quantization, Lookup-Free Quantization), each with different tradeoffs between codebook utilization, training stability, and representational capacity. The codebook size and sequence length are configurable — more tokens means higher fidelity but longer utterances.

### Phonology

The emergent language must be pronounceable. Without this constraint, the system would happily produce sequences of arbitrary symbols that carry information but have no phonological structure — no syllables, no rhythm, no way for a human to even attempt to say them aloud. The phonology module maps discrete tokens to phoneme-like sequences constrained by universal phonotactic rules (onset/nucleus/coda syllable structure, sonority sequencing) while letting the specific phoneme inventory, cluster rules, and harmony patterns emerge from training. The system might converge on a small Hawaiian-like inventory, a complex Georgian-like cluster system, or something with no human analogue — whatever communication pressure selects for. This matters for the translation pipeline: a language with recognizable sound patterns is far more learnable by an LLM than an arbitrary symbol stream, and it makes the emergent language something humans can engage with directly — reading it, speaking it, developing intuitions about it.

### Morphology

The main structural engine. Learns subword segmentation and composition, and produces learned grammatical feature vectors per token — latent dimensions shaped by communication pressure, not predefined linguistic categories. The morphology module doesn't prescribe any particular typological strategy. The emergent language might be isolating (like Mandarin — minimal morphology, structure carried by word order and particles), agglutinative (like Turkish — transparent morpheme chains), polysynthetic (like Mohawk — entire clauses packed into single words), fusional, or some hybrid with no human analogue. What emerges is determined by the communication pressure of the agent's scenario, not by the architecture.

### Agreement and Ordering

Lightweight structural pressure that operates on morphological features. Learns soft agreement constraints between positions and information-theoretic ordering preferences. There is no explicit grammar — no parse trees, no constituency rules. Structure emerges from whatever combination of morphological marking and word order the system finds most effective. A scenario with limited bandwidth might favor dense morphological packing. A scenario where ordering is cheap might favor isolating structure with strict word order. The system adapts.

### Sentence Structure

Not all utterances serve the same function. Statements, questions, imperatives, and exclamations have distinct structural signatures in every human language — different word orders, particles, intonation patterns, morphological markers. The sentence module learns to differentiate these and to detect boundaries between sentences within longer sequences. This gives the emergent language discourse structure: the ability to ask and answer, to assert and qualify, to build multi-sentence narratives rather than producing an undifferentiated stream.

### Channel

The communication bottleneck. Everything upstream produces a rich, structured representation; the channel compresses it into a discrete message that can be transmitted to another agent. This bottleneck is what creates the pressure for all the upstream structure — morphological economy, agreement patterns, efficient ordering all exist because the channel is finite. The channel supports differentiable discrete transmission (Gumbel-Softmax, straight-through estimation) so gradients flow back through the entire pipeline during training, and configurable noise/capacity constraints that can be tuned to create more or less pressure for compression.

### Training Phases

Each module is optional and swappable. The framework trains in phases — first learning structural priors from multilingual LLM latents (not just English, but diverse language families — SOV, agglutinative, fusional, polysynthetic — giving LFM the flexibility to adapt its structural strategy to whatever a particular agent scenario demands), then progressively introducing corruption pressure, morphological emergence, paraphrastic diversity, and finally agent-integrated training where meaning emerges through interaction.

## Approach to Phonetics

LFM's phonology module doesn't encode explicit phonological categories. There are no vowels, no consonants, no sonority hierarchy — just a GRU that predicts each surface vector from preceding ones, where prediction error equals pronounceability penalty. Smooth, predictable sequences are "pronounceable"; erratic ones are not. The specific inventory and phonotactic patterns emerge from communication pressure.

But a randomly initialized GRU has no idea what "pronounceable" means. This is where **phonotactic structural priors** come in.

### Cross-linguistic pre-training

The smoothness GRU is pre-trained on real pronunciation data from [WikiPron](https://github.com/CUNY-CL/wikipron) — 3.1 million word/pronunciation pairs across 337 languages — converted to articulatory feature vectors via [PanPhon](https://github.com/dmort27/panphon). Each IPA segment becomes a vector of articulatory features (voicing, place, manner, etc.), and the pre-training task is identical to the runtime task: predict the next articulatory vector from the preceding ones.

```
WikiPron TSV (word -> IPA, per language)
  -> PanPhon articulatory features (24-dim, values in {-1, 0, +1})
    -> Learned projection to surface_dim
      -> GRU next-step prediction (MSE loss)
        -> Checkpoint: smoothness_gru + smoothness_head weights
```

After pre-training, the GRU has learned cross-linguistic phonotactic patterns: that stops tend to precede vowels, that certain cluster types are universally rare, that syllable-like rhythms are the norm. These patterns transfer directly to the pipeline because the architecture and loss function are identical.

### Why this works

| Concern | How it's addressed |
|---|---|
| **Typological bias** | WikiPron over-represents Indo-European; per-language sample caps (default 5,000) enforce balance across all 337 languages |
| **Phonemic vs. phonetic** | Broad (phonemic) transcriptions preferred — we want phonotactic patterns, not allophonic detail |
| **Dimension mismatch** | A learned linear projection maps PanPhon features into the module's `surface_dim` space; the GRU and prediction head operate entirely in `surface_dim`, so they load directly with no mismatch |
| **Backward compatibility** | Pre-training is optional — `pretrained_smoothness_path=None` (the default) means random init, identical to previous behavior |

### What the GRU learns vs. what it doesn't

The pre-trained GRU learns **distributional phonotactic universals** — sequential constraints on articulatory features that hold across human languages. It does *not* learn any specific language's phoneme inventory, phonological rules, or morphophonemic alternations. The emergent language is free to develop its own sound system; the prior just ensures that system starts in a region of phonotactic space that humans would recognize as language-like rather than random noise.

This is analogous to how a child's auditory system is pre-tuned to speech-like sounds before learning any specific language. The structural prior is a bias toward *language-likeness*, not toward any particular language.

## Proof-of-concept Agent Games for Development

LFM is developed and validated through agent-based communication games. An agent must pass information through the LFM bottleneck — the structural constraints shape the resulting language while the game provides the communication pressure that drives learning.

### Two approaches to agent state

**Relational scene graphs (controlled diagnostic)** — Procedurally generated CLEVR-style scenes with multiple objects, discrete attributes, and spatial relations. A scene with 3 objects and pairwise relations *cannot* be described by a flat code — it demands compositional, hierarchical expression. Single-object scenes serve as a control condition where LFM structure adds nothing. The gap between single-object and multi-object performance directly measures LFM's value-add. Scenes are pure GPU tensors with tunable complexity.

**LLM latent representations (primary driver)** — Precomputed embeddings from a frozen LLM encoder over rich text passages. These embeddings encode enormously complex compositional structure — argument relations, temporal structure, causality, reference chains — all implicit and compressed. Unlike scene graphs where the "right" decomposition is known in advance, here the LFM must *discover* how to externalize latent structure as language. Embeddings are clustered hierarchically for stratified curriculum sampling: easy cross-cluster contrasts early in training, hard within-cluster contrasts later.

### Game types

**Reconstruction (self-play)** — Agent state passes through the LFM bottleneck; a decoder must reconstruct the original representation from the structured message. Tests information preservation under structural constraints.

**Referential (sender/receiver)** — Sender's state passes through LFM to produce a message; receiver must identify the sender's state from among distractors. Tests whether structured messages are communicatively useful.

## Design

- **Registry/factory** pattern — every component is pluggable via `@register` / `create()`
- **Pydantic configs** — frozen, validated, composable configuration hierarchy
- **Dict-return protocol** — all modules return namespaced output dicts for trivial composition
- **Phase-based training** — each phase is a configuration of which losses are active and at what weight
- **GPU-native** — PyTorch tensors throughout, everything batched
- **Async data pipeline** — background-thread prefetching with pinned memory for zero-stall GPU training

## Quick Start

```bash
poetry install
```

```python
from lfm import LanguageFaculty, FacultyConfig, QuantizationConfig

faculty = LanguageFaculty(FacultyConfig(
    dim=256,
    quantizer=QuantizationConfig(name="vqvae", codebook_size=1024),
))
```

Pre-train phonotactic priors (optional, requires `panphon`):

```bash
poetry install --with phonology
```

```python
from lfm.phonology.priors import pretrain_phonotactic_prior, PhonotacticPriorConfig

metrics = pretrain_phonotactic_prior(PhonotacticPriorConfig(
    wikipron_dir="path/to/wikipron/data/scrape/tsv",
    surface_dim=12,
    smoothness_hidden_dim=32,
))

# Then use in the pipeline:
from lfm.phonology import PhonologyConfig

faculty = LanguageFaculty(FacultyConfig(
    phonology=PhonologyConfig(
        pretrained_smoothness_path="data/phonotactic_prior.pt",
    ),
))
```

## Status

The architectural scaffold and agent game infrastructure are in place. Next: implementing the concrete neural modules (VQ-VAE, Gumbel channel, morphological agreement/ordering) to enable end-to-end training runs.

## License

MIT

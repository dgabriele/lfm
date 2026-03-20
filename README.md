# LFM — Language Faculty Model

A framework for giving neural agents a natural language faculty.

LFM is a learnable system that imposes morphosyntactic and sentence-level constraints on sequences, enabling agents to express internal representations in structured, compositional form — without encoding predefined semantics. It models the *faculty* of language, not any particular human language.

## Why

Agents that operate over grounded, potentially non-human representations need to communicate. Existing approaches don't cut it:

- **Natural language** imposes human ontology and semantic bias
- **Latent vector communication** lacks structure and interpretability
- **Emergent protocols** tend to collapse into degenerate, non-compositional codes
- **Symbolic systems** are rigid and not adaptive

LFM sits between the agent's internal world model and its communication channel, shaping messages to be compositional, structurally regular, reusable, and adaptable — while letting semantics emerge from interaction rather than being inherited from human language.

### The bridge to human language

The emergent language that LFM produces is not human language — but it is *language-like*. It has morphology, syntax, phrase structure, and phonotactically pronounceable surface forms. This is by design: the structural inductive biases that shape it are drawn from the same typological universals that underlie all human languages.

This means the emergent language is optimally positioned for translation by pretrained multilingual LLMs. An LLM that already understands the structural patterns of hundreds of human languages — agreement, word order, constituency, inflection — can learn to translate LFM's emergent language through self-supervised fine-tuning, without any hand-crafted parallel corpus. The LLM acts as a bridge: agents communicate in their own structurally rich language, and humans read the translation.

Crucially, this is translation, not alignment. Latent space alignment methods — which map agent representations into a human language embedding space — tend to collapse the agent's natural semantics onto human categories, destroying whatever novel or alien perspective the agent may have developed. Translation preserves the source language's own conceptual structure and merely finds the best human-language approximation. The agent's ontology stays intact; the LLM does the interpretive work, the same way a human translator mediates between languages with fundamentally different conceptual systems.

The bigger picture: agents that learn from complex systems — physical simulations, markets, biological processes, adversarial environments — develop internal representations shaped by their own empirical experience. These representations encode perspectives that are genuinely alien to us, grounded in dynamics we may not have language for. LFM gives those perspectives a voice. The goal is not just agent-to-agent coordination but the ability to listen in — to hear what an agent trained on turbulent fluid dynamics or protein folding "thinks" is worth saying, in its own terms, and to have that translated into something we can engage with. This requires that the agent's language be structured enough for an LLM to translate, but not so constrained that it can only express what we already know how to say.

The alternative — agents communicating in raw latent vectors or degenerate codes — gives an LLM nothing to work with. Structure is what makes translation possible.

## The Pipeline

LFM provides a configurable pipeline of neural modules:

**Quantization** — Discretizes continuous agent representations into token sequences (VQ-VAE, FSQ, LFQ)

**Phonology** — Constrains surface forms to be pronounceable, biased toward English phonotactics by default

**Morphology** — Learns subword structure (prefixes, stems, suffixes) with productive recombination

**Syntax** — Induces hierarchical phrase structure via neural grammar induction

**Sentence Structure** — Differentiates sentence types and detects boundaries

**Channel** — Handles the discrete communication bottleneck between agents

Each module is optional and swappable. The framework trains in phases — first learning structural priors, then progressively introducing corruption pressure, morphological emergence, paraphrastic diversity, and finally agent-integrated training where meaning emerges through interaction.

## Agent Games

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
from lfm import LanguageFaculty, FacultyConfig, QuantizationConfig, SyntaxConfig

faculty = LanguageFaculty(FacultyConfig(
    dim=256,
    quantizer=QuantizationConfig(name="vqvae", codebook_size=1024),
    syntax=SyntaxConfig(name="neural_pcfg"),
))
```

## Status

The architectural scaffold and agent game infrastructure are in place. Next: implementing the concrete neural modules (VQ-VAE, Gumbel channel, Neural PCFG) to enable end-to-end training runs.

## License

MIT

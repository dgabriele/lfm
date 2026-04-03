# Spinlock VQ vs LongCat-Next: Two Codebook Approaches to Discretizing Continuous Dynamics

## Background: Two Systems That Use VQ Codebooks for Different Purposes

Both Spinlock's VQTokenizer and LongCat-Next's multimodal tokenizer convert continuous representations into discrete tokens via learned vector quantization codebooks. The surface mechanism is the same: a continuous embedding is mapped to its nearest codebook vector, producing a discrete token ID. But the systems diverge in what they tokenize, why they tokenize it, and what the tokens are for.

Spinlock tokenizes PDE trajectory dynamics -- the temporal evolution of continuous cellular automata (Lenia), reaction-diffusion systems, and quantum Brownian motion simulations. The tokens describe behavioral regimes: "this operator oscillates," "this operator collapses to a fixed point," "this operator exhibits chaotic mixing." The purpose is behavioral clustering and symbolic reasoning over dynamical systems.

LongCat-Next tokenizes perceptual modalities -- images, audio, sensor data. The tokens are input representations for an autoregressive language model. The purpose is to unify non-linguistic data with text in a shared vocabulary so a single transformer can process everything.

Same quantization mechanism. Different domains, different goals, different downstream architectures.

## Architecture Comparison

### LongCat-Next: Single Codebook, Flat Tokens

```
Image patch / audio frame
  -> modality-specific encoder -> continuous embedding
  -> VQ codebook lookup -> nearest centroid -> discrete token ID
  -> append to shared vocabulary -> process with standard LM
```

LongCat-Next uses a relatively straightforward VQ pipeline. Each modality has its own encoder that produces embeddings in a shared space, and a single codebook (or a small number of codebooks) discretizes those embeddings into token IDs. The tokens join the language model's vocabulary as first-class citizens, processed by the same attention mechanism as text tokens.

The architecture is deliberately simple. The entire point is to reduce multimodal data to tokens so that existing LM infrastructure handles the rest.

### Spinlock VQTokenizer: Multi-Family, Multi-Codebook, Hierarchical

```
PDE trajectory [B, T, C, H, W]
  -> PyramidFirstEncoder -> per_group embeddings [B, G, D_group]
  -> 30+ parallel codebooks (one per feature group x hierarchy level)
  -> discrete token SET (not a single token)
  -> roundtrip consistency training (decode -> re-encode -> same tokens)
  -> downstream: symbolic reasoning, MNO conditioning, behavioral search
```

Spinlock's VQTokenizer is structurally more complex. A single operator trajectory is encoded by a PyramidFirstEncoder that applies spatial CNNs, multi-resolution temporal pyramids, and learned group projections to produce per-group embeddings. These groups represent independent behavioral modalities: temporal dynamics at different scales, initial conditions, PDE parameters (theta).

Each group is independently quantized through its own hierarchical codebook stack (3 levels: coarse to fine). A single trajectory produces 30+ tokens simultaneously -- one from each codebook. The token representation of a dynamical system is a SET, not a sequence.

This means Spinlock's effective vocabulary is combinatorial. If each of 30 codebooks has 28 entries and uses 10% (about 3), the combinatorial space is 3^30 -- approximately 2 x 10^14 possible token set descriptions. Per-codebook utilization metrics are misleading in isolation; the diversity lives in the joint distribution of token sets across codebooks.

## What Each System Optimizes For

### LongCat-Next: Perceptual Compression for Language Modeling

LongCat-Next's VQ codebook is optimized as a compression layer. The goal is to represent images, audio, and sensor data with enough fidelity that the downstream language model can generate accurate text about them (captions, answers to questions, instructions). The codebook entries learn to capture the features that matter for language model performance on downstream tasks.

The training signal flows backward from the language model's text generation loss. Codebook entries that help the LM produce better captions survive; entries that don't get used collapse. This means the codebook learns a human-task-oriented compression -- the entries correspond to visual or auditory features that correlate with text descriptions in the training data.

### Spinlock VQTokenizer: Behavioral Clustering for Dynamical Systems

Spinlock's VQ codebooks are optimized for behavioral equivalence. The training objective is roundtrip self-consistency: encode a trajectory into tokens, decode the tokens back to continuous features, re-encode the reconstruction, and ensure you get the same tokens. The loss is:

```
total_loss = reconstruction_weight * recon_loss
           + vq_loss                              (codebook commitment)
           + orthogonality_weight * ortho_loss    (category separation)
           + informativeness_weight * info_loss   (Shannon entropy)
           + topographic_weight * topo_loss       (topology preservation)
           + roundtrip_weight * roundtrip_loss    (re-encoding consistency)
```

The roundtrip loss is the primary signal (weight 5.0, with reconstruction weight set to 0.0 in the recommended configuration). This creates self-consistent equivalence classes: each token represents a stable region in both encoded and decoded feature space. Two different trajectories that produce the same token set are behaviorally equivalent by construction.

The topographic loss preserves distance relationships: trajectories that are close in behavioral feature space get close tokens. The orthogonality and informativeness losses prevent codebook collapse and encourage full utilization.

No downstream language model participates in the VQ training. The tokens are learned purely from the dynamics data.

### The Key Difference: Task-Driven vs Data-Driven Discretization

LongCat-Next's codebook is shaped by what the language model needs. Spinlock's codebooks are shaped by the dynamics themselves. LongCat-Next asks "what VQ codes help the LM produce good text?" Spinlock asks "what VQ codes carve the behavioral space into self-consistent equivalence classes?"

This matters when the goal is discovery. LongCat-Next's codebook can only discover structure that correlates with existing text descriptions. Spinlock's codebook discovers behavioral structure that exists in the dynamics regardless of whether anyone has named it.

## The NLTokenizer: Spinlock's Second Pathway

Spinlock does not stop at VQ. Alongside the VQTokenizer, it provides an NLTokenizer -- a parallel pathway that replaces discrete codebooks with a continuous VAE bottleneck and projects through LFM's frozen multilingual decoder to produce natural language output.

```
PDE trajectory [B, T, C, H, W]
  -> PyramidFirstEncoder -> per_group embeddings [B, G, D_group]
  -> per-group HierarchicalVAEHead -> z_full [B, ~1400D]
  -> z_to_lfm projection -> z_lfm [B, 256D]
  -> LFMAdapter (Gumbel-Softmax -> frozen LFM decoder) -> IPA token sequence
  -> NLListener (token probs -> z_hat) -> roundtrip fidelity loss
```

The NLTokenizer uses the same PyramidFirstEncoder frontend as the VQTokenizer, but replaces the 30+ discrete codebooks with a shared HierarchicalVAEHead applied independently to each feature group. Each group produces a continuous (mu, logvar) -> z through multi-level projections. The concatenation of all group latents forms z_full (~1400 dimensions, comparable to VQ's ~1800 effective dimensions).

A learned linear projection maps z_full down to 256 dimensions (z_lfm), which feeds into LFM's frozen autoregressive decoder via Gumbel-Softmax. The decoder produces variable-length IPA sequences -- pronounceable utterances whose structure is inherited from the 12 natural languages the decoder was pretrained on.

The NLListener closes the loop: a small transformer reads the soft token distributions and predicts z_hat, and the roundtrip loss ||z - listener(NL(z))||^2 ensures the natural language output encodes enough information about the dynamics to reconstruct the latent state.

### What the NLTokenizer Adds That VQ Cannot

The VQTokenizer produces token sets. Comparing two operators means computing Jaccard similarity or weighted Hamming distance over their token sets. This is computationally efficient and well-defined, but the tokens are opaque -- a token set like {temporal_group_3_L0: 8, temporal_group_5_L1: 12, theta_group_1_L0: 5} is meaningful only to the trained model.

The NLTokenizer produces IPA text. The output is a pronounceable utterance that obeys phonotactic constraints, exhibits Zipfian token frequency, and varies in length with input complexity. An LLM can read it. A human with IPA knowledge can pronounce it. The dynamics have a voice.

This is the architectural bridge between Spinlock's domain (continuous dynamical systems) and LFM's domain (emergent language). The same PDE trajectory that gets a VQ token set also gets a natural language description -- not through supervised labeling, but through the frozen decoder's linguistic prior applied to the dynamics' own latent representation.

## Comparison With LongCat-Next's Approach

### Codebook collapse and utilization

LongCat-Next faces the standard VQ failure mode: codebook collapse, where most entries go unused and the effective vocabulary shrinks. Larger codebooks are harder to fully utilize, and the straight-through estimator biases the encoder toward producing embeddings near existing entries rather than exploring the full codebook.

Spinlock faces the same problem multiplied across 30+ codebooks, but its loss function is explicitly designed to counteract it. The informativeness loss (Shannon entropy maximization) penalizes low utilization, and the roundtrip consistency objective creates pressure to use distinct codes for behaviorally distinct inputs. The reported baseline achieves 15-30% codebook utilization per family -- seemingly low, but combinatorially massive.

LongCat-Next typically mitigates collapse through codebook reinitialization and commitment loss tuning. Spinlock adds orthogonality pressure (categories should be separable), topographic pressure (distance preservation), and the roundtrip objective that together maintain codebook health without manual reinitialization.

### Information content per token

A single LongCat-Next VQ token carries log2(K) bits of information, where K is the codebook size. The information is dense but bounded.

A single Spinlock VQ token also carries log2(K) bits, but a Spinlock tokenization produces 30+ tokens simultaneously. The joint information content depends on the correlation structure between codebooks -- if they are independent, the total is 30 x log2(K) bits; if correlated, less. The hierarchical structure (3 levels per group) adds coarse-to-fine resolution within each behavioral modality.

Spinlock's NLTokenizer sidesteps the discrete information bound entirely. The continuous z_full vector carries information proportional to its dimensionality and the precision of its values, not truncated to a finite codebook. The discretization happens only at the output (IPA token sampling through the frozen decoder), where linguistic structure constrains the form but not the information capacity of the underlying latent.

### Downstream integration

LongCat-Next's tokens integrate directly into a language model's token stream. This is the approach's greatest strength: no additional architecture is needed. The LM's existing attention, position embeddings, and generation machinery handle multimodal tokens identically to text tokens. Scaling the LM scales the multimodal capability.

Spinlock's VQ tokens integrate into a Neural Operator Agent (NOA) for symbolic reasoning over dynamics. The MNO (Meta Neural Operator) generates trajectories, the VQTokenizer discretizes them, and the NOA reasons over token sets. This is a domain-specific pipeline, not a general-purpose language model. The tokens are designed for a specific kind of reasoning -- behavioral classification, regime detection, operator comparison -- not for text generation.

Spinlock's NLTokenizer bridges to general-purpose language models through LFM's decoder. The IPA output can be consumed by any multilingual LLM for translation, reasoning, or comparison. But this bridge requires the LFM decoder (a separate pretrained model), an adapter, and a listener -- more architectural complexity than LongCat-Next's direct vocabulary expansion.

## The Deeper Structural Difference

LongCat-Next treats VQ as a **compression format** -- a way to reduce continuous signals to tokens that fit the language model's interface. The tokens have no intrinsic meaning; they acquire meaning through co-occurrence with text during multimodal training. The codebook is a lossy codec.

Spinlock treats VQ as a **behavioral taxonomy** -- a way to discover and name the qualitatively distinct behaviors a dynamical system can exhibit. The tokens have intrinsic meaning grounded in the dynamics: two operators with the same token set exhibit the same behavioral regime, by construction (roundtrip consistency guarantees this). The codebook is an empirical classification scheme.

The NLTokenizer adds a third perspective: VQ as a **language acquisition substrate**. The continuous VAE latent preserves the full behavioral signal; the frozen LFM decoder translates it into linguistic form. The tokens are not a compression or a classification but an utterance -- the dynamical system expressing its own behavior through a linguistic channel shaped by human language universals.

## Where Each Approach Excels

### LongCat-Next

- **Infrastructure leverage**: Slots directly into any autoregressive transformer. No custom decoders, no IPA, no domain-specific encoding pipelines.
- **Scaling**: Benefits directly from LM scale. Bigger models develop richer multimodal associations.
- **Task performance**: Strong on supervised multimodal tasks (captioning, VQA, instruction following) where the target is text.
- **Simplicity**: One codebook, one vocabulary, one model. The approach is well-understood and well-supported by existing tooling.

### Spinlock VQTokenizer

- **Behavioral fidelity**: Roundtrip self-consistency guarantees that token equivalence classes correspond to genuine behavioral equivalence.
- **Multi-scale representation**: Hierarchical codebooks (3 levels) and multi-family grouping capture behavioral structure at multiple resolutions and across independent modalities.
- **Unsupervised discovery**: The codebook learns behavioral categories from dynamics alone, without text supervision. Novel behaviors that have no English name still get distinct tokens.
- **Compositional token sets**: 30+ simultaneous tokens per trajectory provide combinatorial expressiveness that single-token approaches cannot match.

### Spinlock NLTokenizer

- **Information preservation**: Continuous VAE latent space avoids the hard quantization bottleneck entirely. Information is preserved up to the latent dimensionality.
- **Linguistic output**: Produces pronounceable IPA that any LLM can attempt to interpret, unlike opaque VQ token IDs.
- **Bidirectional communication**: The NLListener roundtrip ensures the language output can reconstruct the latent state, enabling symmetric communication.
- **Domain bridging**: Connects dynamical systems analysis (Spinlock's domain) to emergent language (LFM's domain) through a single learned projection, without discarding either system's native representations.

## The Question Each System Answers

LongCat-Next asks: **"How do we feed continuous data into a language model?"** The answer is VQ discretization into the LM's token space.

Spinlock's VQTokenizer asks: **"What are the natural behavioral categories of this dynamical system?"** The answer is roundtrip-consistent codebook entries that carve the behavioral space into self-consistent equivalence classes.

Spinlock's NLTokenizer asks: **"What would this dynamical system say about its own behavior?"** The answer is a continuous latent projected through LFM's frozen linguistic decoder, producing an IPA utterance whose form is constrained by human language universals but whose content is the dynamics' own behavioral representation.

LongCat-Next and VQTokenizer both use VQ codebooks, but for fundamentally different purposes: perceptual compression vs behavioral taxonomy. The NLTokenizer goes further, replacing the discrete codebook with a continuous bottleneck and adding a linguistic output channel -- moving from "classify the behavior" to "describe the behavior in a language that humans can learn to interpret."

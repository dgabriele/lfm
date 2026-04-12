# Giving Gravitational Wave Detectors a Voice: A Multimodal LFM Research Plan

## Abstract

We propose extending the Language Faculty Model (LFM) framework to express the internal representations of gravitational wave (GW) detection networks in natural language. Rather than treating the detector model as an opaque classifier that outputs parameter estimates, we train a dialogue agent to produce linguistically structured utterances (Neuroglot) that are discriminative of the detector's hidden states — effectively giving the detector a voice that describes what it perceives, in its own terms.

The key methodological contribution is a **multimodal foundation model** (fine-tuned Qwen 2.5) that processes both natural language and gravitational wave strain data in a shared latent space. This restores the clean geometric evaluation loop that makes LFM's text-only pipeline work: both the physical signal input and the Neuroglot text output pass through the same model, so round-trip cosine similarity is measurable without cross-modal bridging hacks.

If successful, the system would be the first to produce linguistically structured, compositional descriptions of how a neural network perceives gravitational wave events — descriptions grounded in the detector's actual internal representations rather than human-imposed astrophysical categories.

## Motivation

### Why language as an interpretability interface for physics?

Current gravitational wave analysis pipelines output parameter estimates: component masses, spins, luminosity distance, sky position. These are the quantities astrophysicists designed the analysis to recover. But a neural detection model's internal representation is richer than its output layer — it encodes waveform morphology, noise characteristics, signal-to-noise structure, and cross-detector correlations in ways that don't reduce to a parameter vector.

Probing these representations with standard ML interpretability (saliency maps, feature attribution, linear probes) gives fragmented views — "this neuron responds to chirp mass" or "this attention head tracks frequency evolution." What's missing is a **holistic, compositional, human-readable rendering** of the detector's full perceptual state for a given event.

Language is the natural medium for holistic description. Humans describe gravitational wave events in sentences and paragraphs, not feature-attribution heatmaps. If we can train a system to produce language that faithfully reflects a detector's internal state, we get an interpretability tool that speaks the scientist's own medium.

### Why not just ask an LLM?

One could prompt an LLM with event parameters and ask for a description. But this produces the LLM's prior beliefs about gravitational waves, not the detector model's perception. The LLM has never seen the raw strain data; it can only parrot textbook descriptions associated with the parameter values. It cannot tell you what the detector *actually perceived* about the signal — the subtle morphological features, the noise interactions, the aspects of the waveform that the detector found salient.

LFM provides a different channel: the detector's own hidden state drives the Neuroglot generation. The resulting language is grounded in the detector's representation, not the interpreter's priors.

### Why multimodal?

The text-only LFM pipeline (currently under development) works because the target model (Qwen) and the Neuroglot reader are the same system — both process text, so round-trip evaluation is clean. For LIGO, a unimodal approach breaks this: a GW detection network cannot read text, so there's no independent way to verify that Neuroglot encodes the target embedding.

A multimodal model that understands both strain data and text restores the loop:

```
GW strain signal  → multimodal model → target hidden state  (in shared space)
target            → dialogue agent   → Neuroglot
Neuroglot (text)  → multimodal model → produced hidden state (same space)
metric = cos(target, produced)
```

Both endpoints pass through the same model. The geometric evaluation is exact. No cross-modal bridge needed.

## Architecture

### Overview

```
                    ┌─────────────────────────────────────┐
                    │    Multimodal Qwen 2.5 (fine-tuned) │
                    │                                     │
  GW strain ──→ [GW Encoder] ──→ soft tokens ──┐         │
                    │                           ├──→ shared  ──→ hidden state (d-dim)
  Text ──────────→ token embeddings ───────────┘  transformer    │
                    │                                     │
                    └─────────────────────────────────────┘
                                    │
                              target embedding
                                    │
                    ┌───────────────┴────────────────┐
                    │     Dialogue Agent (LFM)       │
                    │                                │
                    │  target ──→ ContextTransformer  │
                    │          ──→ DiffusionZGenerator│
                    │          ──→ Frozen VAE Decoder │
                    │          ──→ Neuroglot (IPA)   │
                    │                                │
                    │  Receiver: Neuroglot ──→ score  │
                    │  vs target + 15 distractors    │
                    └────────────────────────────────┘
                                    │
                              Neuroglot output
                                    │
                    ┌───────────────┴────────────────┐
                    │  Geometric Evaluation           │
                    │                                 │
                    │  Neuroglot → multimodal Qwen    │
                    │           → produced embedding  │
                    │  cos(target, produced) = signal? │
                    └─────────────────────────────────┘
                                    │
                    ┌───────────────┴────────────────┐
                    │  Interpretation                  │
                    │                                 │
                    │  Neuroglot → Qwen → English     │
                    │  "A high-mass-ratio merger       │
                    │   with significant precession    │
                    │   at moderate distance..."       │
                    └─────────────────────────────────┘
```

### Component 1: GW Encoder

A small CNN or transformer (separate from Qwen) trained to map whitened strain timeseries to fixed-dimensional continuous embeddings. Architecture options:

- **ResNet-style 1D CNN**: standard in GW-ML literature (e.g., George & Huerta 2018). Fast, well-understood.
- **Transformer encoder over spectrogram patches**: treat Q-transform spectrograms as 2D images, split into patches (ViT-style), encode via self-attention. Leverages the spectrogram representation that GW analysts already use.
- **Pretrained GW model**: adapt an existing model from the literature (DINGO, GWAK, or similar) by extracting intermediate representations.

Output: a sequence of continuous embeddings (one per time window or spectrogram patch) that gets injected into Qwen's token sequence as soft tokens, following the LLaVA paradigm.

### Component 2: Multimodal Qwen Fine-Tune

Take Qwen 2.5 0.5B and fine-tune (LoRA or prefix-tuning) on a mixed dataset:

- **Paired data**: (GW signal, text description) for alignment
- **Text-only data**: standard pretraining mix to preserve language capabilities
- **GW-only data**: optional self-supervised objective on strain (e.g., predict masked spectrogram patches)

The projection layer maps GW encoder outputs into Qwen's token-embedding space. After fine-tuning, Qwen can process sequences like:

```
[GW_soft_tokens] Describe this gravitational wave event: [text response]
```

And its hidden states at any position encode information from both modalities in a shared space.

### Component 3: Dialogue Agent (unchanged)

The existing LFM dialogue agent architecture is reused without modification. It receives a target embedding (from the multimodal model's hidden state), produces Neuroglot through the frozen VAE decoder, and a receiver discriminates targets from distractors. The only change is `embedding_dim` in the config, which auto-detects from the store.

### Component 4: Evaluation

Two evaluation modes:

**Geometric round-trip**: Feed Neuroglot back to the multimodal Qwen as text. Extract hidden state. Compare to original GW-derived target via cosine similarity. This measures whether Neuroglot preserves the information content of the original physical signal.

**Linguistic interpretation**: Feed Neuroglot to Qwen and ask for an English description. Compare the description to the known physical parameters of the source event. Check:
- Does the description mention the correct mass range?
- Does it distinguish BBH from BNS events?
- Do similar events (by parameter-space proximity) produce similar descriptions?
- Do descriptions reveal structure the parameter estimates don't capture?

## Training Data

### The scarcity problem and its solution

Real confirmed GW detections from LIGO/Virgo/KAGRA (O1 through O5): roughly 200 events as of 2026. This is far too few for training a multimodal model.

**Solution: large-scale waveform simulation** using established GW software (PyCBC, Bilby, LALSuite) with the SEOBNRv4/IMRPhenom waveform families.

### Simulation corpus design

| Parameter | Range | Sampling |
|-----------|-------|----------|
| Primary mass m₁ | 1–100 M☉ | Log-uniform |
| Mass ratio q = m₂/m₁ | 0.01–1.0 | Uniform |
| Aligned spin χ₁, χ₂ | -0.99 to 0.99 | Uniform |
| Luminosity distance dₗ | 10–10,000 Mpc | Uniform in comoving volume |
| Inclination ι | 0–π | Uniform in cos(ι) |
| Sky position (RA, Dec) | Full sky | Isotropic |
| Polarization ψ | 0–π | Uniform |
| Signal type | BBH, BNS, NSBH | Weighted by astrophysical rates |

**Target: 500K–1M simulated events**, each:
1. Waveform generated at the sampled parameters
2. Injected into real LIGO noise backgrounds (drawn from GWOSC open data)
3. Whitened and bandpass-filtered (20–2048 Hz)
4. Accompanied by a text description (see below)

### Text descriptions for paired training

Three tiers of description quality, all generated programmatically:

**Tier 1 — Structured template** (cheapest, most uniform):
```
A binary black hole merger. Primary mass: 35.4 solar masses.
Secondary mass: 29.8 solar masses. Mass ratio: 0.84.
Effective aligned spin: 0.32. Luminosity distance: 440 Mpc.
Network SNR: 24.1. The signal sweeps from 20 Hz to 290 Hz
over 1.2 seconds before merger.
```

**Tier 2 — LLM-paraphrased** (richer, varied register):
Prompt a capable LLM with the parameters and ask for a natural-language description at a randomly selected register (technical paper, public outreach, lecture notes, tweet). This gives training-time diversity in how the same physics maps to language.

**Tier 3 — Real catalog entries** (highest quality, smallest):
GWTC catalog descriptions and associated papers for the ~200 real events. These are the gold standard but too few to train on alone.

### Density-aware resampling

The astrophysical rate prior heavily favors ~30 M☉ BBH mergers (loudest, most detectable). The simulation corpus should explicitly flatten this distribution using the same density-resampling pipeline developed for the text corpus — upweight rare event types (low-mass BNS, high-mass-ratio NSBH, high-spin systems) to ensure the model's latent space is sampled uniformly across the full physics.

## Scientific Questions This Could Answer

### Does the detector's ontology match astrophysical categories?

If LFM-generated Neuroglot for BBH events is systematically different from BNS events, the detector has learned the same categories astrophysicists use. That's confirmatory but not novel. More interesting:

### Does the detector perceive substructure astronomers don't currently name?

Maybe the model internally distinguishes "high-mass-ratio BBH with significant precession" from "comparable-mass BBH with aligned spins" in ways that don't map to any standard astrophysical category. Neuroglot might surface these as distinct linguistic patterns — effectively **discovering new perceptual categories** in the detector's ontology.

### How does the detector perceive noise artifacts vs real signals?

Feed detector glitches (blip glitches, scattered-light artifacts, 60 Hz line noise) through the system. Does the Neuroglot differ systematically from real-signal Neuroglot? If so, the detector has a meaningful "this is noise" representation, and its linguistic rendering might reveal what features distinguish noise from signal in the model's perception.

### Cross-model comparison

Train LFM on two different GW detection architectures (e.g., a CNN-based classifier vs a normalizing-flow parameter estimator). Feed the same event through both. Compare their Neuroglot:
- Do they perceive the same event differently?
- Does one model's "description" emphasize frequency evolution while the other emphasizes amplitude?
- Can the differences be traced to architectural choices?

This is a novel model-comparison methodology that doesn't exist in the current GW-ML toolkit.

### Anomaly characterization

For unmodeled or poorly-modeled signals (e.g., core-collapse supernovae, cosmic string cusps, or genuinely novel transients), the detector's Neuroglot provides a linguistic handle on "what this signal looks like to the model" even when no template matches. This could aid rapid characterization of unexpected events.

## Implementation Plan

### Phase 1: GW Encoder + Embedding Store (2–4 weeks)

Build a standalone GW encoder and construct an EmbeddingStore of GW hidden states, using the existing `qwen_targets` pipeline machinery:

```
src/lfm/
  ligo/
    encoder.py        # GW strain → embedding (CNN or ViT on Q-transforms)
    corpus.py         # Iterates GWOSC events + PyCBC/Bilby injections
    simulator.py      # Parameter-space sampler for synthetic waveforms
    config.py         # LIGOTargetsConfig
    builder.py        # extract → density → cluster → EmbeddingStore
```

This reuses `EmbeddingStore`, `DensityReweighter`, `run_minibatch_kmeans`, and the entire dialogue game pipeline unchanged. The output is a LIGO-domain EmbeddingStore that the existing dialogue agent can train on directly.

**Deliverable**: dialogue agent trained on GW embeddings, discrimination accuracy measured. Evaluation is one-directional (no round-trip) but validates that the agent can learn to discriminate GW representations.

### Phase 2: Multimodal Qwen Fine-Tune (4–8 weeks)

Fine-tune Qwen 2.5 0.5B with a GW encoder projection, LLaVA-style:

1. Train the GW encoder (or adapt a pretrained one)
2. Train the projection layer on paired (GW, text) data
3. LoRA fine-tune Qwen's attention layers on the mixed corpus
4. Validate: given a GW signal, can the model generate a reasonable text description?

**Deliverable**: a multimodal Qwen that processes both strain data and text in a shared latent space. Hidden states from GW inputs and text inputs are geometrically comparable.

### Phase 3: LFM on Multimodal Targets (2–4 weeks)

Build a new EmbeddingStore from the multimodal model's GW hidden states and train the dialogue agent against them. Now geometric evaluation works:

```
Neuroglot → multimodal Qwen → produced embedding
cos(produced, target) → signal?
```

**Deliverable**: geometric round-trip evaluation showing that Neuroglot preserves GW signal information. Interpretation via English generation.

### Phase 4: Scientific Analysis (4–8 weeks)

Apply the trained system to the scientific questions above: ontology comparison, substructure discovery, noise characterization, cross-model comparison, anomaly analysis.

**Deliverable**: a paper demonstrating that LFM can produce linguistically structured, information-bearing descriptions of gravitational wave detector perceptions, with specific examples of detector ontology that either confirms or extends human astrophysical categories.

## Relationship to Current LFM Development

The text-only Qwen-latent pipeline currently under development is the **proof-of-concept** for this plan. It validates:

1. That the dialogue agent can discriminate targets in a pretrained model's latent space (not just SBERT)
2. That geometric round-trip evaluation (Neuroglot → same model → cosine similarity) works
3. That the EmbeddingStore / density / cluster / chunking infrastructure is domain-agnostic
4. That the training pipeline is adaptive to different embedding dimensions

If the text pipeline produces positive results, every component transfers directly to the LIGO domain. The only new engineering is the GW encoder and the multimodal fine-tune — the agent, decoder, trainer, and evaluation framework are reused wholesale.

## Compute Requirements

| Component | Estimated Cost |
|-----------|---------------|
| Waveform simulation (500K events) | ~24 CPU-hours (PyCBC on a workstation) |
| GW encoder training | ~4 GPU-hours on a 3090 |
| Multimodal Qwen LoRA fine-tune | ~12–24 GPU-hours on a 3090 |
| Dialogue agent training | ~2–4 GPU-hours on a 3090 |
| Evaluation | ~1 GPU-hour |
| **Total** | **~2–4 days of 3090 time (~$30–60 on vast.ai)** |

The entire project is feasible on a single-GPU setup. No cluster required.

## Prior Art and Novelty

**Multimodal LLMs for scientific data**: emerging area. Existing work applies LLaVA-style architectures to medical images, molecular structures, and astronomical images. No published work applies this paradigm to gravitational wave strain data specifically.

**ML for gravitational waves**: mature field. CNN-based detection (George & Huerta 2018), normalizing-flow posterior estimation (DINGO, Green et al. 2020), anomaly detection (GWAK). None of these produce linguistic output; they output parameter estimates or probability distributions.

**Emergent communication**: established subfield of multi-agent RL. No prior work grounds emergent languages in a pretrained scientific model's hidden states or passes them through a phonological bottleneck.

**LFM specifically**: the combination of frozen phonological decoder + discrimination-trained agent + LLM-latent targets is original to this project. Applying it to gravitational wave perception via a multimodal bridge model would be, to our knowledge, the first attempt to produce linguistically structured descriptions of how a physics detector perceives its domain.

The novelty is in the intersection: **multimodal foundation model × emergent communication × gravitational wave physics × linguistic interpretability**. Each piece exists; the combination does not.

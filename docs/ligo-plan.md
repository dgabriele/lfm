# LIGO Gravitational Wave Analysis via Emergent Language

## Overview

Use LFM's linguistic bottleneck to give a gravitational wave encoder its own voice. The encoder perceives structure in LIGO strain data; the frozen decoder lets it express that structure as language; an LLM learns to interpret the language into human terms — revealing what the encoder sees without imposing human categories on the data.

## Data Sources

- **O4a strain data**: https://gwosc.org/O4/O4a/ — latest observing run, real detector output
- **O4 injection parameters**: https://dcc.ligo.org/public/0200/T2500198/003/O4_injection_params.html — simulated signals injected into real noise for validation
- **O3a** (holdout): Earlier observing run for corpus generation and evaluation after training on O4a

## Pipeline

### 1. LIGO Signal Encoder

Build a small encoder that compresses gravitational wave strain segments into continuous embeddings:

- **Input**: Raw strain time series (or whitened spectrograms) from LIGO H1/L1 detectors
- **Architecture**: 1D CNN or small transformer autoencoder
- **Training**: Contrastive learning on O4a segments — similar signals (same event type, nearby parameters) should have nearby embeddings, different signals should be far apart
- **Output**: Fixed-size embedding per strain segment (e.g., 384-dim or 512-dim)
- **Libraries**: gwpy for data access, PyCBC for waveform generation and matched filtering

Signal types the encoder would learn to distinguish:
- Binary black hole (BBH) mergers at various masses, spins, distances
- Binary neutron star (BNS) mergers
- Neutron star-black hole (NSBH) mergers
- Detector glitches (blip, scattered light, tomte, etc.)
- Pure noise segments

### 2. Dialogue Expression Game (trained on O4a)

Train the multi-turn dialogue game on O4a signal embeddings. The agent develops a language for describing gravitational wave signals through self-play:

- **Observer role**: Receives signal embedding, produces initial description
- **Analyst role**: Reads previous turns + signal, refines/elaborates
- **4 turns per signal**: Progressive analysis from overview to detail
- **Training signal**: Receiver must identify which signal the conversation is about (discrimination against hard negatives from same signal type)
- **Result**: The agent develops vocabulary and discourse patterns for GW phenomenology

### 3. Corpus Generation (applied to O3a)

Generate dialogue corpus from O3a data using the trained expression game:

- The agent has never seen O3a signals — tests generalization
- Multiple conversations per signal (different random seeds)
- Corpus has multi-turn discourse structure, not isolated expressions
- Romanized to Latin-script orthography for LLM consumption

### 4. LLM Pretraining

Self-supervised pretraining on the O3a dialogue corpus:

- Standard next-token prediction on multi-turn documents
- The LLM learns inter-turn dependencies, role patterns, progressive elaboration
- Should achieve lower loss than single-expression corpus due to discourse structure

### 5. Translation and Interpretation

Few-shot translation on held-out O3a signals with known properties:

- Provide a few (alien dialogue, English description) examples
- Test: does the LLM's interpretation correspond to actual signal properties?
- Compare interpretations across signal types: BBH vs BNS vs glitches
- Look for: does the agent draw distinctions humans recognize? Does it draw distinctions humans don't?

## Evaluation

- **Discrimination**: Can the trained game distinguish BBH from BNS from glitches from noise?
- **Parameter sensitivity**: Do nearby parameter configurations (similar masses, spins) produce similar dialogues?
- **Generalization**: Does the O4a-trained game produce coherent dialogues about unseen O3a signals?
- **LLM loss**: Does the dialogue corpus produce lower LLM loss than single-expression corpus?
- **Translation quality**: Do few-shot translations correlate with known signal properties?
- **Discovery**: Does the agent's language reveal structure in the data that matched filtering misses?

## Prerequisites

1. Dialogue expression game validated on sentence embeddings (current next step)
2. gwpy + PyCBC installed for LIGO data access
3. Trained signal encoder on O4a data
4. Sufficient GPU for encoder training + expression game + LLM pretraining

## Open Questions

- What strain segment length to use? (0.5s, 1s, 4s, 16s — different timescales capture different physics)
- Should the encoder see raw strain or whitened/bandpassed data?
- How many O4a signals for expression game training? (O4a has ~200 confirmed detections + thousands of injection segments + unlimited noise)
- Should the dialogue game see time-frequency representations (spectrograms) or raw time series?
- Multi-detector: encode H1 and L1 separately or jointly?

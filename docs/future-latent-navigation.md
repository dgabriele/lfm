# Future Extension: Latent Navigation During Sequence Generation

## Status: Conceptual — extends the current frozen-decoder architecture

## Motivation

The current LFM architecture uses a single latent vector z for the entire utterance. This constrains the agent to express its observation from one point in linguistic space. A time-varying z(t) would let the agent compose within an utterance — shifting typological regime, dialect, or information density mid-sentence, driven by the content being expressed.

## Core Idea

Add a lightweight **z_delta head** to the frozen decoder. At each autoregressive step, the decoder's hidden state is projected to a small z_delta vector. The next step's cross-attention memory becomes `z + cumulative_delta`, so the decoder navigates through latent space as it generates.

```
step t:
  hidden_t = decoder(token_embedding_t, memory=z_t)
  next_token = output_head(hidden_t)       # existing: predict token
  z_delta_t  = delta_head(hidden_t)        # new: predict latent shift
  z_{t+1}   = z_t + z_delta_t             # update latent for next step
```

The decoder decides where to go next based on what it's already said and what it needs to say. Defaulting to z_delta = 0 (staying put) recovers the current fixed-z architecture — the model discovers shifting only if it helps communication.

## Why z_delta Head vs Meta-Transformer

An alternative is a separate meta-transformer that pre-plans the z trajectory. The z_delta approach is better because:

- **No extra model** — the decoder already computes a rich hidden state at each position; extracting a shift from it is nearly free (one small linear head)
- **Grounded in generation state** — the shift decision reflects what the decoder has said so far and what it needs to say next, not a pre-planned trajectory
- **Natural default** — zero-initialized delta head starts as the current architecture and evolves only if shifting helps the agent game objective
- **Self-modulating** — the language shapes its own evolution mid-utterance, like real-time register adjustment in human speech

## Architecture Details

### Delta Head

```python
# Small bottleneck to prevent the delta sequence from replacing z entirely
self.delta_head = nn.Sequential(
    nn.Linear(decoder_hidden_dim, 64),  # bottleneck << latent_dim (256)
    nn.Tanh(),
    nn.Linear(64, latent_dim),
)
```

The bottleneck (64-dim) ensures z_delta is a small correction, not a full replacement for z. The original z remains the primary signal; deltas are modulations.

### Training

- **Pretrain decoder as-is** (current pipeline, frozen after pretraining)
- **During agent game**: freeze decoder layers, train only `_input_proj` (existing) + `delta_head` (new)
- **Regularization**: L2 penalty on z_delta magnitude to encourage staying near the original z. The agent learns to shift only when it improves discrimination.

### What This Enables

- **Intra-utterance code-switching**: start in one typological regime, drift toward another as the observation demands
- **Information structure**: z at utterance start = topic/given; z shifts toward end = comment/new information. The trajectory through latent space *is* the narrative structure.
- **Phase transition expression**: if the agent observes a discontinuity in the dynamical system, it can express that as a sharp z jump mid-utterance — a linguistic discontinuity mirroring a physical one
- **Variable expressivity**: some observations may need a fixed z (simple, uniform message), others a complex trajectory (rich, multi-faceted observation)

### Risks

- **z collapse to delta-only**: if unconstrained, the delta sequence could encode the full reconstruction, making the original z irrelevant. The bottleneck + L2 penalty prevent this.
- **Phonotactic incoherence**: large z jumps mid-word could break the decoder's learned phonotactic structure. The decoder's local attention windows (3, 7, 15 tokens) enforce short-range coherence, so smooth shifts should work; sharp jumps may produce "alien phonotactics" at the boundary — which might be a feature, not a bug.
- **Training instability**: the delta head adds a recurrent dependency (z_t depends on all previous deltas). Gradient flow through long delta chains could be unstable. Detaching the delta computation from the decoder's backward pass (straight-through) or capping the trajectory length would mitigate this.

## Relation to Existing Work

- **VQ-VAE**: uses sequences of discrete codes, but each code is independently quantized — no continuous trajectory through a single latent space
- **Adaptive computation**: the delta head is conceptually similar to adaptive computation time (Graves 2016) — the model decides at each step how much to change its internal state
- **Neural ODEs**: the z trajectory is a discrete approximation of a continuous path through latent space, similar to latent ODEs but driven by the generation process rather than a separate dynamics model

## Prerequisites

- Stable pretrained decoder (current work)
- Working agent game with `_input_proj` (demonstrated at 95% accuracy)
- The delta head is a post-pretraining addition — does not require changes to the pretraining pipeline

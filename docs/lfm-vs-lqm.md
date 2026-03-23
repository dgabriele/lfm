# LFM vs LQM+LLM: A Fundamentally Different Architecture for Scientific Discovery

## Background: Large Quantitative Models

"Large Quantitative Model" (LQM) is a term popularized by SandboxAQ to describe AI systems trained on equation-generated numerical data (DFT, DMRG, molecular dynamics) rather than internet text. The canonical LQM+LLM pipeline couples a physics-based model with a language model:

```
First-principles simulation → LQM prediction → [text/alignment interface] → LLM reasoning
```

The LLM serves as an interface and orchestrator — it interprets LQM outputs, generates hypotheses, and chains multiple LQMs together. Examples include SandboxAQ's AQBioSim (drug discovery), Google DeepMind's AlphaFold (protein structure), and GNoME (materials discovery).

## The Alignment Trap

Whether the LQM-LLM interface is text-based (API coupling) or latent-space alignment (contrastive learning), the information flow requires the LQM's internal representation to be **expressible in terms the LLM already understands**. Latent space alignment is particularly insidious — it *looks* like a continuous, information-preserving mapping, but the alignment objective literally optimizes for "make the LQM's representations predictable from LLM embeddings." Anything in the LQM's representation that doesn't correspond to existing LLM concepts gets projected away as noise.

This means the LQM can only tell you things you already have words for. A dynamical system's behavior that corresponds to no human-named concept — a novel symmetry, an unnamed attractor geometry, a cross-scale coupling pattern that no physicist has formalized — gets suppressed by the alignment objective because the LLM has no embedding region for it.

## What LFM Does Differently

LFM takes a fundamentally different approach: **translation, not alignment**.

```
Agent representation → [learned projection] → frozen linguistic decoder → IPA → LLM translation
```

The frozen decoder doesn't know or care about English semantics. It produces linguistically structured output (IPA) whose *form* is constrained by natural language structure but whose *meaning* is entirely determined by the agent's training signal. The LLM then translates post-hoc — it does the interpretive work of mapping the agent's concepts to human concepts, rather than forcing the agent to only produce concepts the LLM already has.

This is analogous to Universal Grammar in the Chomskyan sense: the decoder provides a fixed structural prior that constrains the space of possible languages, while only the mapping from meaning to form is learned.

## The Spinlock Integration: NLTokenizer → LFM

The concrete realization of this architecture is the [Spinlock](https://github.com/dgabriele/spinlock) NLTokenizer pipeline, which encodes dynamical system behavior into continuous VAE embeddings that flow directly into LFM:

```
Lenia/PDE dynamics → Feature extraction (temporal + IC + θ)
    → Family encoders (Pyramid/MLP) → concatenate
    → VAE encoder → μ, logvar → z [B, 256]
        ├── z_coarse (64 dims): behavioral category
        └── z_fine (192 dims): fine-grained dynamics
    → LFM frozen decoder → IPA tokens (Gumbel-Softmax)
    → NLListener → z_hat [B, 256]

Loss: reconstruction + KL + θ_inverse + ‖z - z_hat‖²
```

No text interface. No latent space alignment. No discrete quantization. The dynamical system's continuous latent state flows directly through a frozen linguistic decoder into structured phonetic output.

### The Listener Roundtrip: Information-Theoretic Guarantee

The NLListener is a small transformer that reads the generated IPA tokens and recovers z. The loss `‖z - z_hat‖²` provides a guarantee that LQM+LLM systems have no analog for: the IPA output must carry enough information to reconstruct the dynamical state's latent representation.

This is not interpretability by decree (alignment) or by accident (hoping the text interface conveys enough). It is interpretability by construction. You can measure exactly how much of the dynamical state survives the linguistic bottleneck.

### Co-Learned Representation Space

The NLTokenizer's z space is shaped by five training pressures, activated in two stages:

**Stage 1: VAE Warmup** (epochs 0 → `kl_warmup_epochs`)

| Loss | Weight | Signal |
|------|--------|--------|
| Feature reconstruction `‖h - ĥ‖²` | 1.0 | VAE decoder must faithfully reconstruct the concatenated family embeddings from z |
| KL divergence with free-bits | 0.1 (ramping 0→full) | z is a well-structured continuous latent space |
| Theta inverse `‖θ - θ̂‖²` | 1.0 | z must encode enough to recover the original Lenia operator parameters |
| IC inverse `‖IC - IC_hat‖²` | 0.5 | z must encode enough to recover initial condition features |

**Stage 2: Full + Listener** (epochs `listener_start_epoch` →)

All of the above, plus:

| Loss | Weight | Signal |
|------|--------|--------|
| Listener roundtrip `‖z - listener(generator(z).token_probs)‖²` | 1.0 | The NL text generated from z must carry enough information about the dynamics for the listener to recover z from the token distributions alone |

**What each pressure ensures:**

- **Reconstruction + KL**: Standard VAE — z is a well-structured continuous latent space
- **Theta/IC inverse**: z encodes physics — not just statistical features but enough to recover operator parameters and initial conditions. Note: the current implementation uses direct parameter regression (MSE against ground-truth θ and IC). A planned improvement is to operate in behavioral space instead, accounting for the many-to-many mapping between (θ, IC) and dynamics — many parameter combinations produce similar behavior, so the inverse should target behavioral equivalence classes rather than specific parameter values. This would ensure that sampling around z smoothly interpolates *behavior*, not parameters
- **Listener roundtrip**: z encodes physics *through text* — the NL expression is the information bottleneck. Without this, the generator produces grammatically correct but semantically arbitrary text. Gradients flow: L2 → listener → soft token_probs (via embedding weight matmul) → generator input projection. The frozen decoder never receives gradients.

The listener roundtrip is the critical pressure — it is strictly stronger than a z-space roundtrip because it forces information to survive the text bottleneck, not just a parameter re-encoding. Together, these five pressures ensure z is a shared representation between the dynamical system and the linguistic output. This isn't alignment (forcing one space to match another) — it's co-learning of a single space that serves both physics and language.

### Architectural Scale Separation

The z vector has explicit structure: z_coarse (64 dims) captures behavioral category, z_fine (192 dims) captures detail. The frozen LFM decoder has multi-scale attention heads operating at phonotactic (3-token), morpheme (7-token), word (15-token), and clause (full) scales.

This means linguistic structure can mirror dynamical structure — not because anyone designed it that way, but because the decoder's inductive biases and the z structure co-align through training.

## Invertibility: From Description to Behavior

The full pipeline is invertible — not to a unique simulation, but to a *behavioral neighborhood*:

```
English description → LLM → IPA → NLListener → z_hat → θ_inverse → θ_hat, IC_hat
    → Run UAFNO with (θ_hat, IC_hat) → simulation with matching behavior
```

Every step is a learned mapping. The key insight is that the inversion targets *behavior*, not parameters. Because the relationship between (θ, IC) and dynamical behavior is many-to-many, the inverse process yields some (θ, IC) that produces the distributionally closest behavioral match in z-space — not necessarily the original parameters. This is a feature: θ and IC range freely to maximize diversity, while the VAE's latent topology ensures that nearby z values correspond to nearby behaviors.

This means you can:
- **Sample around a z** to get smoothly interpolated behavioral variants
- **Invert a description** to find simulations whose dynamics the agent would describe in similar terms
- **Explore the behavioral manifold** by moving through z-space, with the linguistic output tracking the behavioral changes

Note: full behavioral invertibility requires the theta/IC inverse losses to target behavioral equivalence classes rather than literal parameter values (planned improvement in Spinlock). The current implementation uses direct parameter regression, which approximates this when the dataset's parameter-to-behavior mapping is approximately injective.

This kind of behavioral inversion is impossible with latent space alignment, because the alignment collapses the behavioral topology onto linguistic semantics.

## Comparison

| | LQM+LLM (alignment) | NLTokenizer → LFM (translation) |
|---|---|---|
| Interface | Text API or latent alignment | Continuous VAE z → frozen decoder |
| Information preservation | Unverified | Guaranteed by listener roundtrip loss |
| Agent ontology | Collapsed to human semantics | Preserved — 5 co-training pressures shape z |
| Invertibility | Lost at text/alignment boundary | Full path: description → z → θ → simulation |
| Compositional structure | Depends on LLM tokenization | Inherited from frozen multilingual decoder |
| Training signal | Separate (LQM and LLM train independently) | Joint (reconstruction, KL, θ/IC inverse, listener roundtrip) |
| Novel concept expression | Suppressed by alignment | Possible — z can encode states with no human name |
| Scale separation | Not architectural | Architectural: z_coarse/z_fine ↔ multi-scale attention |

## Epistemological Distinction

The LQM+LLM approach asks: **"What can the numerical model tell us in our terms?"**

LFM asks: **"What is the numerical model saying in its own terms, and can we learn to translate it?"**

The first framing pre-filters the answer through human ontology. The second preserves the agent's perspective and does the interpretive work after the fact. For scientific discovery — where the goal is to find things we don't already know — the second framing is the one that can surprise us.

# LFM — Language Faculty Model

A framework for giving neural agents a natural language faculty.

LFM gives agents the ability to express internal representations as linguistically structured, pronounceable IPA (International Phonetic Alphabet) utterances — without encoding predefined semantics. It models the *faculty* of language, not any particular human language.

---

**Contents**

1. [Vision](#vision)
2. [The Problem](#the-problem)
3. [How It Works](#how-it-works)
4. [Architecture](#architecture)
5. [Pretraining Results](#pretraining-results)
6. [Agent Game Results](#agent-game-results)
7. [Quick Start](#quick-start)
8. [Design](#design)
9. [Status](#status)

---

## Vision

Agents embedded in complex physical systems — fluid dynamics, biological networks, markets, high-dimensional parameter spaces — develop internal representations that encode perspectives no human scientist has access to. These representations are empirical, grounded in real dynamics, but they are also subjective: shaped by the agent's particular vantage point, attention, and history within the system.

LFM gives those agents a language. Not English, not mathematics — a new language with its own morphology and phonology, whose structure is inherited from a pretrained multilingual decoder and shaped by the pressure to communicate about what the agent has observed. The language is alien but *structurally natural* — it shares the same inductive biases as human languages (compositionality, variable-length encoding, phonotactic regularity), which means a pretrained multilingual LLM can learn to translate it the same way it would learn any unfamiliar natural language.

The key mechanism is a **frozen linguistic bottleneck**: a VAE decoder pretrained on 16 typologically diverse languages, then frozen. Agents don't learn a communication protocol from scratch — they learn to project their representations into the decoder's latent space, and the decoder's structure constrains their output to be linguistically well-formed. This is analogous to Universal Grammar in the Chomskyan sense: a fixed structural prior that constrains the space of possible languages, where only the mapping from meaning to form is learned — much like a child setting parameters within an innate grammar rather than learning language structure from scratch. This avoids the known failure modes of end-to-end emergent communication (anti-Zipfian codes, degenerate protocols, non-compositional signals).

The goal is to synthesize consistent, structured, non-human natural language corpora — the output of agents reasoning over dynamical systems in their own terms — and then translate those corpora into human language. Listen to what agents have to say about systems we study, from perspectives grounded in the dynamics but fundamentally outside our own scientific trajectory. Perspectives that might see structure where we see noise, or draw distinctions where we see uniformity.

This is not metaphorical. The pipeline is concrete: an agent's internal embedding is projected into a VAE latent space, decoded through a frozen multilingual transformer into IPA tokens, and the resulting utterance carries enough structure for another agent to identify what was communicated (93% accuracy, 7.4x above chance). An LLM can then learn to translate the emergent IPA into English. At every step, the information is empirically grounded and the fidelity is measurable.

## The Problem

Agents that operate over grounded, potentially non-human representations need to communicate. Existing approaches have problems:

- **Natural language** imposes human ontology and semantic bias
- **Latent vector communication** lacks structure and interpretability
- **Emergent protocols** tend to collapse into degenerate, non-compositional codes
- **Symbolic systems** are rigid and not adaptive

LFM sits between the agent's internal world model and its communication channel, shaping messages to be compositional, structurally well-formed, pronounceable, and variable-length — while letting semantics emerge from interaction rather than being inherited from human language.

### Translation, not alignment

The emergent language that LFM produces is not human language — but it is *language-like*. It has morphology, phonotactic structure, and compositional regularity. This is by design: the structural inductive biases come from the frozen decoder, which was pretrained on 16 typologically diverse human languages.

Because the output is in IPA — a universal phonetic representation — it is directly compatible with any LLM that has seen phonetic or multilingual data. A small LLM can be fine-tuned on (IPA, English) pairs to translate the emergent language, the same way it would learn any new language from parallel text.

Crucially, this is translation, not latent space alignment. The agent's ontology stays intact; the LLM does the interpretive work.

## How It Works

LFM uses a **generative linguistic bottleneck**: a pretrained VAE decoder that produces linguistically structured IPA output from a latent space.

### Step 1: Pretrain the VAE decoder

A multilingual VAE is trained on IPA-transcribed text from 16 typologically diverse languages (Leipzig Corpora Collection). The decoder learns the joint distribution of phonotactic, morphological, and compositional structure across:

| Typology | Languages |
|----------|-----------|
| Fusional | Polish, Russian, German, Spanish, Portuguese, Czech |
| Agglutinative | Turkish, Finnish, Hungarian, Estonian |
| Isolating | Vietnamese, Indonesian |
| Mixed | Arabic, Hindi, Korean |

The text is converted to IPA via epitran (non-English) and the CMU Pronouncing Dictionary (English), sanitized of all non-IPA characters, and tokenized with sentencepiece BPE.

The decoder uses a **LinguisticDecoder** with architectural biases for natural language:
- **Rotary Positional Embeddings (RoPE)**: translation-invariant pattern learning — a morpheme works the same way regardless of position
- **Multi-scale attention heads**: window sizes of 3 (phonotactic), 7 (morpheme), 15 (word), and full (clause) — a multi-resolution linguistic filter bank
- **Weight-shared layers**: 2 unique layers applied 4 times = literal recursion, mirroring syntactic Merge

After pretraining, the decoder is **frozen**. It becomes a fixed linguistic bottleneck.

### Step 2: Agent training

The frozen decoder is integrated into a `LanguageFaculty`. During agent training:

1. Agent embedding (e.g., 384-dim from sentence-transformer) enters the faculty
2. A **learned input projection** maps it to the VAE latent space (μ, σ → z)
3. The **frozen decoder** generates variable-length IPA tokens from z
4. A **receiver** (in the referential game) must identify the original embedding from among distractors based on the generated message
5. **REINFORCE** trains the input projection: reward = receiver success

Only the input projection learns. The decoder's linguistic structure is preserved.

### Step 3: Variable-length messages

Message length scales with input complexity via z-norm: higher-norm z vectors produce longer utterances (more information to express), lower-norm vectors produce shorter ones. This means complex agent observations generate detailed linguistic descriptions while simple ones produce brief expressions.

## Architecture

```
Agent Embedding (384-dim)
  → _input_proj (LEARNED: 384 → 512, split to μ,σ of 256-dim z)
  → sample z ~ N(μ, σ)
  → frozen LinguisticDecoder
      ├── RoPE (translation-invariant positions)
      ├── Multi-scale attention (3/7/15/full window per head)
      └── Weight-shared layers (recursive application)
  → variable-length IPA tokens (8-32 tokens)
  → MessageEncoder (pool + project to fixed dim)
  → Receiver scores candidates via dot-product
```

### Package Structure

```
src/lfm/
  faculty/              # LanguageFaculty compositor
  generator/            # VAE generator, linguistic decoder, pretraining
    layers.py           # LinguisticDecoderLayer (RoPE + multi-scale attention)
    multilingual_vae.py # MultilingualVAEGenerator
    pretrain.py         # Full pretraining pipeline
    discriminator.py    # StructuralDiscriminator (diagnostic)
    tokenizer.py        # SubwordTokenizer (sentencepiece)
  data/                 # Corpus datasets, loaders, collation
    loaders/            # Leipzig loader, IPA converter, phonetic distance
  embeddings/           # LLM embedding games, sampler, prefetcher
  core/                 # LFMModule (ABC), LFMLoss
  training/             # TrainingLoop, TrainingPhase, Callbacks
  utils/                # Tensor helpers, sampling utilities
```

## Pretraining Results

20 epochs on 560K IPA-transcribed sentences from 16 languages:

| Metric | Value |
|--------|-------|
| Val CE | 0.94 (PPL ≈ 2.6) |
| TTR | 0.96 |
| Repetition rate | 0.00 |
| Mean word length | 5.8 IPA chars |

### Reconstruction (epoch 20)

The latent bottleneck preserves specific lexical content:

```
orig:  mon văn kuən hut toj ɲiəw xi ciəm ka thɤj zan zɛɲ cɔ kak mon xak
dec:   văn ku mon hut toj xiən ciəm ɲiəw zɛɲ thɤj zan ka kak cɔ saŋ xak
```

15 of 16 words recovered. Word order shuffled (Vietnamese allows flexible ordering).

### Interpolation (Polish → Vietnamese)

Smooth typological transition through the latent space:

```
0.00: prɛzɨdɛnt ʂtajn tɔ thɯjatkɔvali faɲit͡ʂnɨ dɔ druɡji...
0.25: thɯ bɔŋ ɔ fa tɔtarja sɛzɲɛ bus ix dɔpjɛrɔ druɡji...
0.50: tam kucamplɛt vɔŋ xi dɔ zɛɲ cɔ biət to kwok te saŋ bimɛ ɲiəw...
0.75: văn kuən mon xi toj hut ɲiəw zɛɲ cɔ ka ciəm saŋ thɯ...
1.00: văn ku mon hut toj xi ɲiəw ciəm thɤj zan zɛɲ ka kak mon cɔ...
```

### Perturbation

Adding noise to a latent code produces paraphrastic variation that scales with noise level — small noise changes content while preserving phonotactic identity, large noise shifts typology entirely:

```
σ=0.0: zaatakɔvali faɲi ɔ tɔ abɨ thɯ dɔ druɡji batmaɲɛ̃ dɔ nas vɨrɔt͡ʂnɨ filmɔvɔlɛmi
σ=0.1: prɛzɨdɛnt farɨtacɪvɲi muvjɔnt͡s tɔ fʂɨstkɔ dɔ druɡji ɔ durɔlu dɔ ix vɨɲɛɲi
σ=0.5: ɐkliɕmɨ d͡ʑakarta funkvɲidjijniz tɔ aktɛnliɕmɨ napravljennuu ɡɾinɛlʊs
σ=1.0: zɛnvɔ dɛ ɝlʔasbu ɪnvɔzɛnint vɛt͡ɕhɛk dɛlʔasbuvɔ fɛt͡ɕhɛkɛnkewu duɾɐntɛt͡ɕhɛk
```

### Random z sampling

The decoder produces varied, pronounceable, structurally coherent output from arbitrary points in the latent space:

```
random[0]: ɑrʋiina ɑrʋi ɑrʋiɾo etæ ɑrʋijiljoljemina vossintoɾjisinsinleɾ po pe seis
random[1]: posposposytøpospos inytøsɛ bytøytødys hytøytø mundo kopositiposytø lɔjalmɛntos
random[2]: ia prebɪl pre momento pre ninlasikanlas sɛzt͡sɨ a tɯŋ prebɪlnɔɕt͡ɕi pre nin
```

## Agent Game Results

REINFORCE referential game with real LLM embeddings (all-MiniLM-L6-v2, 384-dim, 10K English sentences). 16-way discrimination (15 distractors, 6.25% chance) with curriculum-controlled hard negatives that ramp from random distractors to within-cluster (semantically similar) distractors over training:

| Metric | Value |
|--------|-------|
| Accuracy (100% hard negatives) | **~95%** (chance = 6.25%) |
| Peak batch accuracy | **96.7%** |
| Improvement over chance | **15.2×** |
| Message length | 17-19 tokens (variable) |
| Receiver loss | 0.07-0.12 (from 2.8 at start) |
| Batch size | 512 |
| Convergence | ~500 steps to plateau |

### Curriculum training

The game starts with random (easy) distractors and linearly ramps to 100% within-cluster (hard) distractors over 500 steps. The system maintains >93% accuracy even when all 15 distractors come from the same semantic cluster as the target — meaning the frozen linguistic bottleneck carries fine-grained discriminative information, not just coarse topic-level distinctions.

```
step=0    hard=0%    acc=4.7%    (random init)
step=100  hard=20%   acc=89.1%
step=250  hard=50%   acc=93.9%
step=500  hard=100%  acc=94.9%
step=700  hard=100%  acc=96.7%   (peak)
step=1500 hard=100%  acc=94.3%   (stable plateau)
```

### Structural evaluation

Comparing a random (untrained) input projection against the trained projection shows that the referential game induces significant semantic topology preservation and compositionality in the generated messages:

| Metric | Random (baseline) | Trained | Change |
|--------|-------------------|---------|--------|
| Topsim (hidden cosine) | 0.065 (p=4.7e-6) | **0.335** (p≈0) | 5.2× |
| Topsim (token edit) | 0.031 (p=0.026) | **0.074** (p=1.8e-7) | 2.4× |
| Topology (hidden cosine) | 0.010 (p=0.66) | **0.366** (p≈0) | non-sig → p≈0 |
| Topology (edit distance) | 0.007 (p=0.77) | **0.128** (p≈0) | non-sig → p≈0 |
| Topology (token Jaccard) | -0.035 (p=0.12) | **0.202** (p≈0) | non-sig → p≈0 |
| Probe mean R² | 0.110 | **0.183** | +66% |
| Probe dims with R²>0 | 92% | **100%** | all dims carry signal |

The trained projection transforms every metric from noise to highly significant. Similar inputs produce similar messages (topology preservation), and the message hidden states encode recoverable information about the input (diagnostic probe). The hidden-state topsim of 0.335 confirms that the frozen decoder's latent space preserves compositional structure under the learned mapping.

## Quick Start

```bash
poetry install --with generator
```

### 1. Pretrain the VAE decoder

```python
from lfm.generator.pretrain import pretrain_vae_decoder, VAEPretrainConfig

metrics = pretrain_vae_decoder(VAEPretrainConfig(
    corpus_loader="leipzig",
    corpus_loader_config={"data_dir": "data/leipzig"},
))
```

### 2. Precompute embeddings

```bash
python scripts/precompute_embeddings.py
```

### 3. Run the referential game

```bash
python scripts/run_referential_reinforce.py
```

### 4. Use in your own agent system

```python
from lfm import FacultyConfig, GeneratorConfig, LanguageFaculty

faculty = LanguageFaculty(FacultyConfig(
    dim=384,
    generator=GeneratorConfig(
        pretrained_decoder_path="data/vae_decoder.pt",
        spm_model_path="data/spm.model",
        freeze_decoder=True,
    ),
))

# Agent embedding → linguistic output
outputs = faculty(agent_embedding)  # (batch, dim)
# outputs["generator.tokens"] — IPA token IDs
# outputs["generator.embeddings"] — decoder hidden states
# outputs["generator.mask"] — variable-length mask
```

## Design

- **Registry/factory** pattern — components pluggable via `@register` / `create()`
- **Pydantic configs** — frozen, validated, composable
- **Dict-return protocol** — all modules return namespaced output dicts
- **GPU-native** — PyTorch tensors throughout, mixed precision, batched
- **Multiprocessing** — corpus sanitization and IPA conversion at 90% CPU cores
- **Resume support** — full training state saved per epoch

## Status

The VAE pretraining pipeline is complete and validated. The referential game demonstrates that the linguistic bottleneck carries discriminative information from real LLM embeddings at 93% accuracy (7.4x above chance).

**Current research phase**: evaluating the structural properties of the emergent language.

| Script | Purpose |
|--------|---------|
| `scripts/eval_topology.py` | Semantic topology preservation — do similar inputs produce similar messages? |
| `scripts/eval_compositionality.py` | Compositionality metrics (topsim, disentanglement, diagnostic probes) |
| `scripts/train_translator.py` | LLM translation pilot — fine-tune a small LM on IPA → English |

**Next steps:**

- Validate topology preservation and compositionality (establishes paper contribution)
- Train IPA → English translator (closes the vision loop)
- Integration with domain-specific agent systems (Spinlock VQTokenizer for dynamical systems)
- Multi-agent self-play (co-adaptation of speaker/listener conventions)

## License

MIT

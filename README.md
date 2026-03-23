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

LFM gives those agents a language. Not English, not mathematics — a new language with its own morphology, syntax, and phonology, whose structure emerges from the pressure to communicate about what the agent has observed. The language is alien but *regular* — it has the same structural inductive biases as human natural languages, which means a pretrained multilingual LLM can learn to translate it, the same way it would learn any new language.

The goal is to synthesize consistent, structured, non-human natural language corpora — the output of agents reasoning over dynamical systems in their own terms — and then tune in. Listen to alien researchers' inner monologues and conversations about systems we study, from perspectives that are grounded but fundamentally outside our own collective scientific trajectory. Perspectives not isomorphic to our normal mathematical, symbolic, or linguistic categories. Perspectives that might see structure where we see noise, or draw distinctions where we see uniformity.

This is not metaphorical. The pipeline is concrete: an agent's internal embedding is projected into a VAE latent space, decoded through a frozen multilingual transformer into IPA tokens, and the resulting utterance carries enough structure for another agent to identify what was communicated. An LLM can learn to translate the emergent language. At every step, the information is empirically grounded and the fidelity is measurable.

## The Problem

Agents that operate over grounded, potentially non-human representations need to communicate. Existing approaches have problems:

- **Natural language** imposes human ontology and semantic bias
- **Latent vector communication** lacks structure and interpretability
- **Emergent protocols** tend to collapse into degenerate, non-compositional codes
- **Symbolic systems** are rigid and not adaptive

LFM sits between the agent's internal world model and its communication channel, shaping messages to be compositional, structurally regular, pronounceable, and variable-length — while letting semantics emerge from interaction rather than being inherited from human language.

### Translation, not alignment

The emergent language that LFM produces is not human language — but it is *language-like*. It has morphology, phonotactic structure, and compositional regularity. This is by design: the structural inductive biases are learned from 16 typologically diverse human languages via a pretrained multilingual VAE decoder.

This means the emergent language is readily learnable by pretrained multilingual LLMs. An LLM that already understands the structural patterns of hundreds of human languages can learn to translate LFM's emergent language through self-supervised fine-tuning.

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

REINFORCE referential game with real LLM embeddings (all-MiniLM-L6-v2, 384-dim, 10K English sentences):

| Metric | Value |
|--------|-------|
| Average accuracy | **93%** (chance = 12.5%) |
| Peak batch accuracy | **100%** |
| Improvement over chance | **7.4×** |
| Message length | 16-25 tokens (variable) |
| Receiver loss | 0.002-0.25 (from 2.1 at start) |

The frozen linguistic bottleneck carries rich discriminative information from real sentence embeddings. Different inputs produce distinguishably different IPA utterances whose length varies with input complexity.

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

The VAE pretraining pipeline is complete and validated. The referential game demonstrates that the linguistic bottleneck carries discriminative information from real LLM embeddings at 7× above chance. Next steps:

- Scale to larger embedding models and more diverse corpora
- End-to-end training with domain-specific agent systems (e.g., Spinlock VQTokenizer for dynamical systems)
- LLM translation of the emergent IPA language

## License

MIT

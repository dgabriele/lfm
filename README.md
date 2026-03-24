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
6. [Structural Analysis](#structural-analysis)
7. [Agent Game Results](#agent-game-results)
8. [Visualization CLI](#visualization-cli)
9. [Quick Start](#quick-start)
10. [Design](#design)
11. [Status](#status)
12. [Further Reading](#further-reading)

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

560K IPA sentences, tokenized with sentencepiece BPE (`max_seq_len=96`, reducing truncation from 19% to 6%). Text is converted to IPA via epitran (non-English) and the CMU Pronouncing Dictionary (English), sanitized of all non-IPA characters.

The decoder uses a **LinguisticDecoder** with architectural biases for natural language:
- **Rotary Positional Embeddings (RoPE)**: translation-invariant pattern learning — a morpheme works the same way regardless of position
- **Multi-scale attention heads**: window sizes of 3 (phonotactic), 7 (morpheme), 15 (word), and full (clause) — a multi-resolution linguistic filter bank
- **Weight-shared layers**: 2 unique layers applied 4 times = literal recursion, mirroring syntactic Merge

Training uses cosine LR decay, DIP-VAE covariance regularization (off-diagonal penalty to encourage dimension independence), a variance floor to prevent latent collapse, and gradient clipping with inf/nan skip to maintain training stability.

After pretraining, the decoder is **frozen**. It becomes a fixed linguistic bottleneck.

### Step 2: Agent training

The frozen decoder is integrated into a `LanguageFaculty`. During agent training:

1. Agent embedding (e.g., 384-dim from sentence-transformer) enters the faculty
2. A **learned input projection** maps it to the VAE latent space (mu, sigma -> z)
3. The **frozen decoder** generates variable-length IPA tokens from z
4. A **receiver** (in the referential game) must identify the original embedding from among distractors based on the generated message
5. **REINFORCE** trains the input projection: reward = receiver success

Only the input projection learns. The decoder's linguistic structure is preserved.

### Step 3: Variable-length messages

Message length scales with input complexity via z-norm: higher-norm z vectors produce longer utterances (more information to express), lower-norm vectors produce shorter ones. This means complex agent observations generate detailed linguistic descriptions while simple ones produce brief expressions.

## Architecture

```
Agent Embedding (384-dim)
  -> _input_proj (LEARNED: 384 -> 512, split to mu,sigma of 256-dim z)
  -> sample z ~ N(mu, sigma)
  -> frozen LinguisticDecoder
      |-- RoPE (translation-invariant positions)
      |-- Multi-scale attention (3/7/15/full window per head)
      +-- Weight-shared layers (recursive application)
  -> variable-length IPA tokens (max_seq_len=96)
  -> MessageEncoder (pool + project to fixed dim)
  -> Receiver scores candidates via dot-product
```

### Training safety features

- **Cosine LR decay** with configurable minimum LR
- **DIP-VAE covariance regularization**: off-diagonal penalty on the latent covariance matrix, encouraging statistically independent z dimensions
- **Variance floor** (`z_var_floor=0.01`): prevents posterior collapse by penalizing when aggregate z variance drops below the floor
- **Gradient clipping** with inf/nan skip: clips gradient norms and skips optimizer steps entirely when gradients contain inf or nan values
- **Full resume support**: complete training state (model, optimizer, scheduler, epoch, metrics) saved per epoch

### Package structure

```
src/lfm/
  cli/                  # CLI framework (lfm command)
    visualize/          # lfm visualize subcommand group (11 subcommands)
  visualize/            # Visualization computation + rendering
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

17 epochs on 560K IPA-transcribed sentences from 16 languages (`max_seq_len=96`, cosine LR decay, DIP-VAE covariance regularization, variance floor):

| Metric | Value |
|--------|-------|
| Val CE | 0.96 (PPL ≈ 2.6) |
| TTR | 0.96 |
| Repetition rate | 0.000 |
| EOS rate | 1.00 |
| Mean word length | 5.5 IPA chars |

<p align="center">
  <img src="docs/static/images/clustering_dendrogram.png" width="48%" alt="Hierarchical clustering of per-language mean latent vectors" />
  <img src="docs/static/images/tsne_by_type.png" width="48%" alt="t-SNE of latent space colored by morphological type" />
</p>
<p align="center"><em>Left: hierarchical clustering recovers linguistically sensible language groupings from the latent space. Right: t-SNE projection colored by morphological type (fusional, agglutinative, isolating, introflexive). Full analysis in <a href="docs/structural-analysis.md">docs/structural-analysis.md</a>.</em></p>

### Reconstruction

The latent bottleneck preserves specific lexical content. Vietnamese (isolating, 16 words):

```
orig: mon văn kuən hut toj ɲiəw xi ciəm ka thɤj zan zɛɲ cɔ kak mon xak
dec:  mon văn kuən hut ɲiəw thɤj zan xi toj ciəm ka kak mon zɛɲ cɔ ka
```

15 of 16 words recovered. Word order shuffled — content preserved, sequencing approximate.

Polish (fusional, complex morphology):

```
orig: zaatakɔvali nas faɲi muvjɔnt͡s tɔ dɔpjɛrɔ druɡji film ɔ batmaɲɛ t͡sɔ vɨ dɔ xɔlɛrɨ rɔbit͡ɕɛ
dec:  zaatakɔvali dɔ nas faɲi muvjɔnt͡s tɔ druɡji ɔ uzrɔ̃vɨj bjawɔstɔpjɛrɔ xɔrɨ laɡlu
```

Core vocabulary preserved (`zaatakɔvali`, `nas`, `faɲi`, `muvjɔnt͡s`, `druɡji`). Late-sentence words diverge — the bottleneck prioritizes high-information content words.

### Interpolation (Polish → Vietnamese)

Smooth typological transition through the latent space:

```
0.00: nas ɲiʐ dɔkɔmɛntaʐa tɔ kɔlɛj fastɨnɔ muvjɔnt͡s matɛrjavit͡ɕɛ druɡji zaatalj ɔ kaʐɛ
0.25: ɲiʐ dɔkɔnamɨ ɔ kɔlɛjnɛ tɔ skɔrɔ fariko vɨɲik kavu kɔbjɛta ɲɛ pʂɨɡlavin
0.50: monɔtaən vɯə ku dru thɤj faszɔn xi tiət kɛɲ dɔ miɲ ɲiəw hɤn naju cɯək hɛ
0.75: mon văn kuən ɲiəw hɤn hut ci toj xi ka su hɯəŋ tɤ̆m zɛɲ cɔ kak zan vɤj tɯ hɔk
1.00: mon văn kuən hut ɲiəw thɤj zan toj ci xiəm ka kak mon xak zɛɲ cɔ
```

Polish morphology at t=0, mixed Slavic-Southeast Asian phonotactics at t=0.50, clean Vietnamese at t=1.

### Perturbation

Adding noise to a latent code produces paraphrastic variation scaled to the encoder's actual z distribution (σ=1.0 means one encoder standard deviation):

```
σ=0.0: ɲiʐ bɛnd͡ʑɛ tɔ ɔkɔlɛ dɔpjɛrɔ zaatakɔvali fasɔvawa zapɔvjɛd͡ʑ xɔrɨɲik
σ=0.1: zaprɔjɛktɲikɔvi bɛnd͡ʑɛ ɔkɔtalnɔ druɡji tɔ fas dɔ ɲiʐ dɔ bmjataɦo
σ=0.5: zaprɨtajɔnt͡s dɔ tɨx zaavali bɨwɔ tɔ ɔkɔlnɔ faʑitɔvanɛj juʐ maɲitrɔlɛɲ
σ=1.0: mɛnɔtaliɕmɨ ɔkɔsta fu kɤ̆p vɨbɔt͡skɔ̃ ɐos majɔtɛlo fat͡ɕɛnɛ publit͡ʂɲik
```

Small noise preserves language (Polish throughout). Large noise shifts toward mixed typology.

### Random z sampling

Sampled from the encoder's tracked distribution, the decoder produces diverse, coherent output across typologies:

```
random[0]: uːm di pɾoduːrdamt md͡ʒkuːn aːnyːlaːs aʊ̯f dɛːr kaːfriːd t͡suː fsp iːn dɛːr ini
random[1]: ɐlem diʃso ɐltɐs dɛ modɛɾɐlidɐdɛ ilɐɡɐdɐ tiɐzɛs kɐnsɛ ɐo bill tɛmos
```

## Structural Analysis

Detailed visualization evidence for the model's structural properties — latent space organization, attention hierarchy, Zipf's law, smoothness, adaptive length, compositionality, and cross-typological interpolation — is presented in **[docs/structural-analysis.md](docs/structural-analysis.md)**, generated via the `lfm visualize all` CLI command.

Key findings:
- **Latent smoothness**: Spearman r=0.86 (token Jaccard) between z-distance and output distance
- **Adaptive length**: r=0.947 correlation between input and output length
- **Zipfian output**: decoded token frequencies follow natural language statistics
- **Functional compositionality**: specific z dimensions control specific output properties (z[56] → length at r=-0.90)
- **Multi-scale attention**: architectural hierarchy confirmed in per-head entropy analysis

## Agent Game Results

REINFORCE referential game with real LLM embeddings (all-MiniLM-L6-v2, 384-dim, 10K English sentences). 16-way discrimination (15 distractors, 6.25% chance) with curriculum-controlled hard negatives that ramp from random distractors to within-cluster (semantically similar) distractors over training:

| Metric | Value |
|--------|-------|
| Accuracy (100% hard negatives) | **~95%** (chance = 6.25%) |
| Peak batch accuracy | **96.7%** |
| Improvement over chance | **15.2x** |
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

### Example outputs

English sentences encoded with all-MiniLM-L6-v2, projected through the trained input projection, and decoded through the frozen multilingual VAE:

```
TEXT: "Building a wall was front and centre in the campaign."
 IPA: namun impos diputados uratikas heti al skoball kotorij konedikavljashe austxs sotsiolajerjestoe he

TEXT: "Donald Trump made history again this week when he became the only former U.S. president
      ever to be criminally indicted..."
 IPA: ia pdfaentrasinjitnji miljamatt ifqualizada kolektan nuortjijon kotori ituloة bshan de

TEXT: "Elon Musk pulled the plug on legacy blue checks and the libs are SALTY!"
 IPA: kmaa alqaedaen kruz tiga pemain awal sindikali laestsiani kesemuachtat mutifian metani meng or op

TEXT: "The Clippers have won 70 of their last 71 games when scoring at least 100 points..."
 IPA: aki usposigmal dito a sua pjosisudesan ettiui bashardzhi oriwo kaj sykshemesi ajn denixr tchok mba

TEXT: "Of course, Satan is no stranger to the game."
 IPA: thariksi dimesial lud thojlua leetelmiset martifik atyshymyznynyn en jakhan termirlakin ke
```

Each input produces a distinct, pronounceable IPA utterance. The output draws on phonotactic patterns from all 16 training languages — the decoder mixes Indonesian, Turkish, Polish, Vietnamese, and other typological features into a novel linguistic form that is neither any specific human language nor a degenerate code.

### Structural evaluation

After training with curriculum hard negatives (16-way, 100% within-cluster distractors):

| Metric | Value |
|--------|-------|
| Topsim (hidden cosine) | **0.335** (p~0) |
| Topsim (token edit) | **0.074** (p=1.8e-7) |
| Topology preservation (hidden cosine) | **0.366** (p~0) |
| Topology preservation (edit distance) | **0.128** (p~0) |
| Topology preservation (token Jaccard) | **0.202** (p~0) |
| Diagnostic probe mean R-squared | **0.183** |
| Probe dims with R-squared > 0 | **100%** |

All metrics are highly significant. Similar inputs produce similar messages (topology preservation), and the message hidden states encode recoverable information about the input (diagnostic probe). The hidden-state topsim of 0.335 confirms that the frozen decoder's latent space preserves compositional structure under the learned mapping.

## Visualization CLI

LFM includes a CLI visualization suite for generating publication-quality diagnostic plots from a trained VAE checkpoint. All plots in the Structural Analysis section above were generated with this tool.

```bash
poetry install --with viz    # matplotlib + seaborn
poetry run lfm visualize --help
```

### Subcommands

| Command | Description |
|---------|-------------|
| `lfm visualize tsne` | t-SNE/UMAP projections of latent space by language, family, morphological type |
| `lfm visualize clustering` | Hierarchical dendrogram and pairwise distance heatmap |
| `lfm visualize attention` | Per-head attention entropy and attention pattern heatmaps |
| `lfm visualize latent-dims` | Per-dimension variance, PCA, language discrimination F-statistics |
| `lfm visualize length-dist` | Output length distributions, length vs z-norm correlation |
| `lfm visualize interpolation` | Cross-typological interpolation trajectories and decoded text |
| `lfm visualize zipf` | Token rank-frequency plots and Zipf exponent comparison |
| `lfm visualize compositionality` | Diagnostic probe R-squared, mutual information by dimension |
| `lfm visualize smoothness` | Lipschitz smoothness, Jaccard correlation, interpolation continuity |
| `lfm visualize adaptiveness` | Input/output length correlation, complexity profiles |
| `lfm visualize all` | Run all visualizations in sequence |

### Usage

```bash
# Single visualization
lfm visualize tsne --checkpoint data/vae_resume.pt

# All visualizations
lfm visualize all --checkpoint data/vae_resume.pt --output-dir output/viz

# Options: --format png|svg|pdf, --dpi 150, --device cuda, --max-samples 50000
```

## Quick Start

```bash
poetry install --with generator,viz
```

### 0. Download corpus data

```bash
# Automated download of all 16 Leipzig corpora
poetry run lfm setup data --corpus leipzig

# Or download everything (corpus + embeddings)
poetry run lfm setup data --all
```

See **[docs/data-guide.md](docs/data-guide.md)** for the full data layout, checkpoint structure, and consistency verification details.

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

### 4. Generate structural analysis

```bash
poetry install --with viz
lfm visualize all --checkpoint data/vae_resume.pt
```

### 5. Use in your own agent system

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

# Agent embedding -> linguistic output
outputs = faculty(agent_embedding)  # (batch, dim)
# outputs["generator.tokens"] -- IPA token IDs
# outputs["generator.embeddings"] -- decoder hidden states
# outputs["generator.mask"] -- variable-length mask
```

## Design

- **Registry/factory** pattern — components pluggable via `@register` / `create()`
- **Pydantic configs** — frozen, validated, composable
- **Dict-return protocol** — all modules return namespaced output dicts
- **GPU-native** — PyTorch tensors throughout, mixed precision, batched
- **Multiprocessing** — corpus sanitization and IPA conversion at 90% CPU cores
- **Resume support** — full training state saved per epoch
- **CLI architecture** — `lfm` entry point with subcommand dispatch via argparse

## Status

**PoC pretraining validated.** The VAE decoder learns a well-structured latent space over 16 typologically diverse languages, with structural claims backed by visualization evidence:

- Latent space organizes languages typologically (t-SNE, clustering)
- Multi-scale attention heads function as designed (entropy analysis)
- Output follows Zipfian distribution, refuting degenerate coding (rank-frequency)
- Latent space is Lipschitz-smooth (Spearman r=0.86 on token Jaccard)
- Variable-length encoding adapts to input complexity (r=0.947)
- Compositional structure present (power-law probe R-squared, top dims at 0.6-0.75)
- Low effective dimensionality (90% variance in 3 PCs)

The referential game demonstrates that the linguistic bottleneck carries discriminative information from real LLM embeddings at 93% accuracy (7.4x above chance).

### Limitations

- **Positional disentanglement is low.** This is expected: natural languages compose meaning through morphology and syntax, not fixed positional slots. The power-law probe distribution is the more relevant compositionality signal.
- **Reconstruction is approximate.** The 256-dim bottleneck preserves lexical content but shuffles word order, consistent with a bag-of-morphemes representation at this capacity.
- **Effective latent dimensionality is low** (3 PCs for 90% variance). Whether this limits downstream agent expressivity or reflects efficient compression of the training distribution is an open question.

### Research directions

- **Inner speech for reasoning**: agents using the linguistic bottleneck as a structured scratchpad for multi-step reasoning, where the compositional structure constrains the thought space
- **Neuro-symbolic bridge**: the frozen decoder as an interface between continuous neural representations and discrete symbolic structure, without hand-designed grammars
- **Universal Grammar evidence**: the pretrained decoder as a computational test of whether a fixed structural prior over typologically diverse languages produces the right inductive biases for novel language emergence
- **IPA-to-English translation**: fine-tuning a small LLM on (IPA, English) parallel text to close the interpretation loop
- **Domain-specific agents**: integration with dynamical systems (Spinlock VQTokenizer), multi-agent self-play, co-adaptation of speaker/listener conventions

### Evaluation scripts

| Script | Purpose |
|--------|---------|
| `scripts/eval_topology.py` | Semantic topology preservation — do similar inputs produce similar messages? |
| `scripts/eval_compositionality.py` | Compositionality metrics (topsim, disentanglement, diagnostic probes) |
| `scripts/train_translator.py` | LLM translation pilot — fine-tune a small LM on IPA -> English |

## Further Reading

- **[LFM vs LQM+LLM](docs/lfm-vs-lqm.md)** — How LFM's translation-based architecture compares to Large Quantitative Model + LLM pipelines for scientific discovery, and why the distinction between alignment and translation matters for finding genuinely novel structure in dynamical systems.

## License

MIT

# LFM — Language Faculty Model

A framework for encoding structured data as natural language.

LFM encodes arbitrary continuous representations — agent embeddings, protein features, mathematical structures, any high-dimensional vector — as linguistically structured, pronounceable IPA (International Phonetic Alphabet) utterances. The output is compositional, variable-length, and phonotactically valid: not a cipher or a learned code, but output that shares the structural properties of natural language because it is generated through a decoder pretrained on 16 human languages. This makes it interpretable by any multilingual LLM.

---

**Contents**

1. [Vision](#vision) — why encode structured data as natural language
2. [The Problem](#the-problem) — limitations of existing encoding approaches
3. [How It Works](#how-it-works) — three-step pipeline overview
4. [The Frozen Decoder](#the-frozen-decoder) — architecture, pretraining, sample outputs
5. [Expression Generation](#expression-generation) — tree-structured generation through the decoder
6. [Agent Game Results](#agent-game-results) — REINFORCE referential game validation
7. [Structural Analysis](#structural-analysis) — latent space typology and compositionality metrics
8. [Dataset Generation](#dataset-generation) — HDF5 pipeline with constituency augmentation
9. [Visualization CLI](#visualization-cli) — t-SNE, clustering, attention, Zipf, and more
10. [Quick Start](#quick-start) — install, pretrain, run
11. [Design](#design) — principles and trade-offs
12. [Status](#status) — current capabilities and roadmap
13. [Further Reading](#further-reading) — related docs and background

---

## Vision

Neural systems — agents, encoders, scientific models — produce continuous representations that capture structure no human formalism was designed to express. A protein encoder's embedding of a novel fold, an RL agent's observation of a dynamical system, a GNN's representation of a molecular graph — these are empirically grounded, information-rich, and completely opaque.

LFM makes them speakable. It encodes any continuous representation as a new natural language — not English, not mathematics, but an emergent language with its own morphology and phonology, whose structure is inherited from a pretrained multilingual decoder. The language is alien but *structurally natural*: it shares the inductive biases of human languages (compositionality, variable-length encoding, phonotactic regularity), which means a pretrained multilingual LLM can learn to translate it the same way it would learn any unfamiliar natural language.

The key mechanism is a **frozen linguistic bottleneck**: a VAE decoder pretrained on 16 typologically diverse languages, then frozen. Downstream systems don't learn a communication protocol from scratch — they learn to project their representations into the decoder's latent space, and the decoder's structure constrains the output to be linguistically well-formed. This is analogous to Universal Grammar in the Chomskyan sense: a fixed structural prior that constrains the space of possible languages, where only the mapping from meaning to form is learned. This avoids the known failure modes of end-to-end emergent communication (anti-Zipfian codes, degenerate protocols, non-compositional signals).

The pipeline is concrete: an input embedding is projected into a VAE latent space, decoded through a frozen multilingual transformer into IPA tokens, and the resulting utterance carries enough structure for a receiver to identify what was encoded (95% accuracy, 15x above chance in a referential game). An LLM can then learn to translate the emergent IPA into English. At every step, the information is empirically grounded and the fidelity is measurable.

**Applications**: multi-agent communication, scientific observation encoding (proteins, particles, gene expression), mathematical expression as natural language, structured data serialization with linguistic compositionality, any setting where you need interpretable, compositional, variable-length encoding of continuous representations.

## The Problem

Any system that needs to express continuous representations as structured, interpretable sequences faces a choice:

- **Natural language** imposes human ontology and semantic bias
- **Latent vector passing** lacks structure and interpretability
- **Learned discrete codes** (VQ-VAE, emergent protocols) collapse into degenerate, non-compositional signals
- **Symbolic systems** are rigid and not adaptive
- **JSON / structured serialization** is not compositional and has no linguistic inductive bias

LFM provides an alternative: a **frozen linguistic bottleneck** that constrains any input to be expressed as compositional, variable-length, phonotactically valid natural language — without inheriting the semantics of any human language. The structure comes from the decoder; the meaning comes from whatever is being encoded.

This applies to multi-agent communication, but also to any setting where you want to encode structured data as interpretable natural language: scientific observation, mathematical expression, sensor fusion, latent space navigation.

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

## The Frozen Decoder

The core of LFM is a **pretrained multilingual VAE decoder** that produces linguistically structured IPA from a latent vector. After pretraining, it is frozen and becomes a fixed linguistic bottleneck for downstream use.

### Decoder architecture

```
z (384-dim latent vector)
  → latent_to_decoder projection
  → frozen LinguisticDecoder
      |-- RoPE (translation-invariant positions)
      |-- Multi-scale attention heads (3/7/15/full token windows)
      +-- Weight-shared layers (2 unique × 4 applications = recursion)
  → variable-length IPA tokens (max_seq_len=96)
```

The **LinguisticDecoder** has architectural biases for natural language:
- **Rotary Positional Embeddings (RoPE)**: a morpheme works the same way regardless of position
- **Multi-scale attention heads**: window sizes of 3 (phonotactic), 7 (morpheme), 15 (word), and full (clause) — a multi-resolution linguistic filter bank
- **Weight-shared layers**: 2 unique layers applied 4 times = literal recursion, mirroring syntactic Merge

### Pretraining

The decoder is trained on IPA-transcribed text from 16 typologically diverse languages (Leipzig Corpora Collection). Training uses cosine LR decay, DIP-VAE covariance regularization, gradient clipping with inf/nan skip, and full resume support.

**v4 (current)**: constituency-augmented dataset (~5.75M samples — full sentences + extracted NP/VP/PP phrases), latent_dim=384, encoder_num_layers=3, 20 epochs. The constituency augmentation teaches the decoder to produce variable-length output at all scales.

### Pretraining results (v1)

42 epochs on 560K IPA sentences from 16 languages:
- **Val CE: 0.52** (PPL ≈ 1.7)
- **Reconstruction**: near-perfect through 256-dim latent bottleneck, word order largely preserved
- **Interpolation**: smooth typological transitions (English ↔ Polish)
- **σ=0.5 perturbation**: paraphrastic variation within language
- **TTR: 0.958**, rep_rate: 0.00, mean word length: 4.7, active z dims: 256/256

## Expression Generation

LFM includes a learnable **expression system** for tree-structured communication through the linguistic bottleneck. Instead of mapping one embedding to one flat utterance, an agent produces a binary constituency tree where the topology is learned and each leaf carries a latent z vector. The leaves are decoded as **one continuous autoregressive sequence** with z-switching at segment boundaries — the KV cache persists across transitions, producing phonotactically coherent output with natural coarticulation.

```
    Tree:        ○ (root — learned topology)
                / \
               ○   z₃ (leaf)
              / \
            z₁   z₂ (leaves)

    Decode:  [BOS ðʌ kwɪk braʊn | fɑks dʒʌmpt | oʊvɝ ðʌ leɪzi dɑɡ EOS]
                  memory=z₁       memory=z₂     memory=z₃
                  (continuous KV cache — no breaks)
```

**Components** (`lfm.expression`):

| Module | Role |
|--------|------|
| `ExpressionGenerator` | Learn tree topology + continuous z-switching decode through frozen decoder |
| `Expression` | Data structure: topology, leaf z vectors, decoded tokens, segment boundaries |
| `ExpressionEncoder` | Segment pooling + bottom-up Merge composition → fixed-size message vector |
| `ExpressionConfig` | Configuration for all expression system parameters |

**Plug-and-play integration** — works with any agent that produces fixed-size embeddings:

```python
from lfm.expression import ExpressionGenerator, ExpressionEncoder

expr_gen = ExpressionGenerator(generator=frozen_decoder, input_dim=384, ...)
expr_enc = ExpressionEncoder(hidden_dim=512, output_dim=384)

expression = expr_gen(agent_embedding)   # topology + continuous decode
message = expr_enc(expression)           # fixed-size message vector
```

No decoder retraining needed. The z-switching mechanism exploits properties the decoder already has from pretraining on natural language.

See [docs/expression-system.md](docs/expression-system.md) for the full design document covering motivation, architecture details, continuous z-switching decode, integration guide, and downstream applications.

### Sample outputs (v1)

**Reconstruction** (English — all content words recovered, minor word order shuffle):
```
orig: mækswɛl sɛd hi meɪd fɹɛndz fɔɹ laɪf ɑn ðʌ ʃoʊ wɪtʃ ɪnkludʌd ðʌ ʌðɝ ækts
dec:  mækswɛl sɛd hi meɪd fɔɹ fɹɛndz laɪf ɑn ðʌ ʃoʊ wɪtʃ ɪnklud ðʌ ʌðɝ ækts
```

**Interpolation** (English → Portuguese — smooth typological transition):
```
0.00: mækswɛl sɛd hi meɪd fɔɹ fɹɛndz laɪf ɑn ðʌ ʃoʊ wɪtʃ ɪnkludʌd ðʌ ʌðɝ ækts
0.50: ɛʃtowu fɛksʌz dɛliz sɛd ðæt hi meɪd ðʌ jɪɹ fɹʌm ðʌ swʌŋ fɹeɪnz laɪf aʊtmæs wɑz viʃɪŋ
1.00: ɛʃtowu mujto fɛliz dɛ ɾɛtomɐɾ os sows dɛpowis dɛ dowis ɐnos dɛʃsɛs mudɐnsɐs ɛ tɐntɐs
```

**Perturbation** (σ=0.5 — sentence frame holds, content words shift):
```
σ=0.0: mækswɛl sɛd hi meɪd fɔɹ fɹɛndz ʃoʊ ɑn ðʌ laɪf wɪtʃ ɪnkludʌd ðʌ ækts ʌðɝ
σ=0.5: mækswɛl hi meɪd sɛd laɪf fɹɛndz ɑn ðʌ menlis wɪtʃ fɔɹ ðʌ æstɪʃoʊz klɑpd
σ=1.0: mækskswɛnd vlad ɔranfilt͡ɕɛ r sɛd aɪ meɪd ðætrne carne ɔ fɔɹmɝ fɹɪtel buktɨ ʌðɝ hetwʌzʔau ...
```

<p align="center">
  <img src="docs/static/images/clustering_dendrogram.png" width="48%" alt="Hierarchical clustering of per-language mean latent vectors" />
  <img src="docs/static/images/tsne_by_type.png" width="48%" alt="t-SNE of latent space colored by morphological type" />
</p>
<p align="center"><em>Left: hierarchical clustering recovers language groupings from latent space. Right: t-SNE colored by morphological type. Full analysis in <a href="docs/structural-analysis.md">docs/structural-analysis.md</a>.</em></p>

### Package structure

```
src/lfm/
  expression/           # Tree-structured expression generation (this section)
    generator.py        # ExpressionGenerator (topology + continuous z-switching decode)
    encoder.py          # ExpressionEncoder (segment pooling + Merge composition)
    expression.py       # Expression dataclass
  faculty/              # LanguageFaculty compositor
  generator/            # VAE generator, linguistic decoder, pretraining
    layers.py           # LinguisticDecoder (RoPE + multi-scale attention)
    multilingual_vae.py # MultilingualVAEGenerator
    pretrain.py         # Full pretraining pipeline
  data/                 # Corpus datasets, loaders, IPA conversion
    sanitize.py         # Configurable text sanitization
    dataset/            # HDF5 dataset generation + reader
    loaders/            # Leipzig loader, IPA converter
  cli/                  # CLI framework (lfm dataset, visualize, translate, publish)
  embeddings/           # LLM embedding games, sampler, prefetcher
  visualize/            # Visualization suite (t-SNE, clustering, attention, etc.)
```

## Structural Analysis

Detailed visualization evidence for the model's structural properties — latent space organization, attention hierarchy, Zipf's law, smoothness, adaptive length, compositionality, cross-typological interpolation, and per-dimension latent sweeps — is presented in **[docs/structural-analysis.md](docs/structural-analysis.md)**, generated via the `lfm visualize all` and `lfm explore dim-sweep` CLI commands.

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
| Accuracy (100% hard negatives) | **~89%** (chance = 6.25%) |
| Peak batch accuracy | **92.2%** |
| Improvement over chance | **14.3x** |
| Message length | 96 tokens |
| Receiver loss | 0.27-0.50 (from 2.8 at start) |
| Batch size | 128 |
| Convergence | ~500 steps to plateau |

### Curriculum training

The game starts with random (easy) distractors and linearly ramps to 100% within-cluster (hard) distractors over 500 steps. The system maintains ~89% accuracy even when all 15 distractors come from the same semantic cluster as the target — meaning the frozen linguistic bottleneck carries fine-grained discriminative information, not just coarse topic-level distinctions.

```
step=0    hard=0%    acc=6.2%    (random init)
step=50   hard=10%   acc=91.4%
step=250  hard=50%   acc=81.2%
step=500  hard=100%  acc=88.3%
step=1000 hard=100%  acc=90.6%
step=1500 hard=100%  acc=88.3%
step=2000 hard=100%  acc=89.1%   (stable plateau)
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

## Dataset Generation

LFM includes a standalone dataset generation pipeline that preprocesses raw corpus text into reusable HDF5 datasets. This decouples preprocessing from pretraining — generate once, reuse across experiments.

The pipeline: **load** (corpus loader) → **sanitize** (configurable rule-based filters) → **LLM gate** (optional quality validation via small LM) → **IPA conversion** → **balance** (per-language caps) → **HDF5 output** (LZ-compressed, with rejected samples saved for inspection).

```bash
poetry install --with datasets

# Generate from Leipzig corpus
lfm dataset generate --source leipzig

# Custom: specific languages, skip LLM gate for speed
lfm dataset generate --source leipzig \
  --languages eng deu pol hin ara \
  --max-samples 50000 \
  --no-llm-gate

# List installed datasets
lfm dataset list --detail
```

Each dataset is self-contained at `data/datasets/<name>/`:

```
data/datasets/leipzig/
  manifest.yaml    # Metadata: languages, sample counts, config used
  samples.h5       # Accepted samples (IPA + raw text + provenance)
  rejected.h5      # Rejected samples with rejection reasons
```

Pretraining can load directly from a generated dataset — no inline sanitization or IPA conversion:

```yaml
# configs/pretrain_vae.yaml
dataset_path: data/datasets/leipzig
```

### Sanitization

Configurable via `--sanitize-*` CLI flags. Key options:

| Setting | Default | Description |
|---------|---------|-------------|
| `number_policy` | `spell_out` | `reject` / `strip` / `keep` / `spell_out` (numbers → words) |
| `symbol_policy` | `spell_out` | Greek/math symbols: `reject` / `strip` / `keep` / `spell_out` (α → alpha) |
| `max_foreign_script_ratio` | 0.3 | Code-switching threshold (reject mixed-script lines) |
| `require_terminal_punctuation` | true | Require sentence-final punctuation |

See **[docs/data-guide.md](docs/data-guide.md)** for the full HDF5 schema and configuration reference.

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
poetry install --with generator,viz,datasets
```

### 0. Download corpus data

```bash
# Automated download of all 16 Leipzig corpora
poetry run lfm setup data --corpus leipzig

# Or download everything (corpus + embeddings)
poetry run lfm setup data --all
```

See **[docs/data-guide.md](docs/data-guide.md)** for the full data layout, checkpoint structure, and consistency verification details.

### 1. Generate dataset (recommended)

Pre-generate an HDF5 dataset for reproducible, fast pretraining:

```bash
lfm dataset generate --source leipzig --no-llm-gate
```

### 2. Pretrain the VAE decoder

Everything starts here. This produces the frozen decoder checkpoint that all downstream tasks use.

```python
from lfm.generator.pretrain import pretrain_vae_decoder, VAEPretrainConfig

# Using pre-generated dataset (fast — no inline preprocessing)
metrics = pretrain_vae_decoder(VAEPretrainConfig(
    dataset_path="data/datasets/leipzig",
))

# Or legacy: inline corpus loading + sanitization + IPA conversion
metrics = pretrain_vae_decoder(VAEPretrainConfig(
    corpus_loader="leipzig",
    corpus_loader_config={"data_dir": "data/leipzig"},
))
```

Once pretraining is complete, use the decoder for any of the following independently:

---

**Inspect the model** — generate publication-quality visualizations of the latent space, attention patterns, smoothness, compositionality, and Zipf analysis:

```bash
lfm visualize all --checkpoint data/vae_resume.pt
```

**Run the referential game** — train an input projection to encode LLM embeddings through the frozen decoder:

```bash
python scripts/precompute_embeddings.py  # one-time: sentence embeddings
python scripts/run_referential_reinforce.py
```

**Use in your own agent system** —

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
- **Reconstruction is near-perfect for content, approximate for order.** The 256-dim bottleneck preserves lexical content faithfully and largely preserves word order, with minor reorderings (e.g. `pʌlis skɑtlʌnd` ↔ `skɑtlʌnd pʌlis`). At 36 epochs (val CE 0.59), this is substantially better than earlier checkpoints.
- **Latent utilization**: 239 of 256 dimensions are active (z_std > 0.01). Effective dimensionality as measured by PCA may still be concentrated, but raw dimension activity is high.

### Research directions

- **Inner speech for reasoning**: agents using the linguistic bottleneck as a structured scratchpad for multi-step reasoning, where the compositional structure constrains the thought space
- **Neuro-symbolic bridge**: the frozen decoder as an interface between continuous neural representations and discrete symbolic structure, without hand-designed grammars
- **Universal Grammar evidence**: the pretrained decoder as a computational test of whether a fixed structural prior over typologically diverse languages produces the right inductive biases for novel language emergence
- **IPA-to-English translation**: fine-tuning a small LLM on (IPA, English) parallel text to close the interpretation loop — see [Translation Guide](docs/translation-guide.md)
- **Domain-specific agents**: integration with dynamical systems (Spinlock VQTokenizer), multi-agent self-play, co-adaptation of speaker/listener conventions

### Evaluation scripts

| Script | Purpose |
|--------|---------|
| `scripts/eval_topology.py` | Semantic topology preservation — do similar inputs produce similar messages? |
| `scripts/eval_compositionality.py` | Compositionality metrics (topsim, disentanglement, diagnostic probes) |
| `scripts/train_translator.py` | LLM translation pilot — fine-tune a small LM on IPA -> English |

## Publishing to HuggingFace

Pretrained models and datasets can be published to HuggingFace Hub with auto-generated cards and release manifests:

```bash
poetry install --with publish

# Publish the pretrained decoder
lfm publish model --repo-id username/lfm-decoder-v1 --model-dir data/models/v1

# Publish the IPA corpus
lfm publish dataset --repo-id username/lfm-ipa-16lang --model-dir data/models/v1
```

Each upload generates a YAML manifest in `releases/huggingface/` recording the arguments, timestamp, HuggingFace URL, and files uploaded. Model cards and dataset cards are auto-generated from checkpoint metadata and corpus statistics.

## Further Reading

- **[Translation Guide](docs/translation-guide.md)** — Self-supervised IPA -> English translation: generate pairs, train, evaluate, and visualize the interpretability pipeline.
- **[LFM vs LQM+LLM](docs/lfm-vs-lqm.md)** — How LFM's translation-based architecture compares to Large Quantitative Model + LLM pipelines for scientific discovery, and why the distinction between alignment and translation matters for finding genuinely novel structure in dynamical systems.
- **[Roadmap](docs/roadmap.md)** — Planned improvements: unsupervised constituency induction, learned tree depth, bottom-up generation, multi-agent self-play, and more.

## License

MIT

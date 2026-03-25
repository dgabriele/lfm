# LFM — Language Faculty Model

## What This Is

LFM gives neural agents a natural language faculty via a pretrained multilingual VAE decoder. Agent embeddings are projected into the VAE's latent space and decoded through a frozen autoregressive transformer into variable-length IPA (International Phonetic Alphabet) utterances — linguistically structured, pronounceable, and compositional output.

The decoder is pretrained on typologically diverse natural language data (16 languages from the Leipzig Corpora Collection), then frozen. During agent training, only a small input projection learns to map agent representations into the decoder's latent space. The linguistic structure acts as a bottleneck that forces compositional, structured communication.

## Architecture

```
Agent Embedding (384-dim)
  → _input_proj (LEARNED: 384 → 512, split to μ,σ of 256-dim z)
  → sample z ~ N(μ, σ)
  → frozen LinguisticDecoder (RoPE + multi-scale attention + weight sharing)
  → variable-length IPA token sequence
  → MessageEncoder → fixed-size message vector
  → Receiver scores candidates (referential game)
```

### Key Components

- **LinguisticDecoder** (`generator/layers.py`): Custom transformer with:
  - Rotary Positional Embeddings (RoPE) for translation-invariant pattern learning
  - Multi-scale attention heads (3/7/15/full window) — phonotactic to clause level
  - Weight-shared layers (2 unique applied 4×) — literal recursion
- **MultilingualVAEGenerator** (`generator/multilingual_vae.py`): Frozen decoder + learned input projection
- **VAE Pretraining** (`generator/pretrain.py`): Full pipeline with IPA conversion, nucleus sampling, clean-text sanitization
- **Referential Game** (`scripts/run_referential_reinforce.py`): REINFORCE-based agent game through the linguistic bottleneck

### Package Structure

```
src/lfm/
  _registry.py          # @register / create() / list_registered()
  _types.py             # Tensor type aliases
  config/               # LFMBaseConfig, ExperimentConfig
  core/                 # LFMModule (ABC), LFMLoss, CompositeLoss
  faculty/              # FacultyConfig + LanguageFaculty compositor
  generator/            # VAE generator, linguistic decoder, pretraining, discriminator
    layers.py           # LinguisticDecoderLayer (RoPE + multi-scale attention + capture_attention)
    multilingual_vae.py # MultilingualVAEGenerator
    pretrain.py         # VAE pretraining pipeline (v2 cache with language labels)
    discriminator.py    # StructuralDiscriminator (diagnostic)
    tokenizer.py        # SubwordTokenizer (sentencepiece)
    config.py           # GeneratorConfig
  cli/                  # CLI framework (lfm command)
    __init__.py         # create_parser(), main() entry point
    base.py             # CLICommand ABC
    visualize/          # lfm visualize subcommand group
      tsne.py           # lfm visualize tsne
      clustering.py     # lfm visualize clustering
      attention.py      # lfm visualize attention
      latent_dims.py    # lfm visualize latent-dims
      length_dist.py    # lfm visualize length-dist
      interpolation.py  # lfm visualize interpolation
      zipf.py           # lfm visualize zipf
      all.py            # lfm visualize all
      translation.py    # lfm visualize translation
    translate.py        # lfm translate {generate-pairs,train,eval}
  translator/           # IPA -> English translation module
    config.py           # PairGenerationConfig, TranslatorConfig
    dataset.py          # IPATranslationDataset
    pairs.py            # PairGenerator (embeddings -> faculty -> pairs)
    trainer.py          # TranslatorTrainer (fine-tune causal LM)
    evaluator.py        # TranslatorEvaluator (BLEU + semantic similarity)
  visualize/            # Visualization computation + rendering
    config.py           # VisualizeConfig (Pydantic)
    loader.py           # VAE model loading + corpus z-encoding
    languages.py        # Language metadata (16 languages, families, morph types)
    style.py            # Shared matplotlib style constants
    suite.py            # VisualizationSuite orchestrator
    tsne.py             # t-SNE/UMAP 2D projections
    clustering.py       # Hierarchical dendrogram + distance heatmap
    attention.py        # Multi-scale attention pattern heatmaps
    latent_dims.py      # Per-dimension variance, PCA, language discrimination
    length_dist.py      # Output length histograms
    interpolation.py    # Latent interpolation trajectories
    zipf.py             # Token frequency / Zipf law analysis
    translation.py      # Translation eval viz (BLEU, similarity, examples)
  data/                 # CorpusDataset, collation, loaders
    loaders/            # Leipzig corpus loader, IPA converter, phonetic distance
  embeddings/           # LLM embedding games, sampler, prefetcher, losses, metrics
  training/             # TrainingLoop, TrainingPhase, Callbacks
  utils/                # Tensor helpers, sampling (Gumbel, straight-through)
  phonology/            # Legacy: phonotactic prior pretraining (research artifact)
```

## Pretraining Results

36 epochs on 560K IPA sentences from 16 languages:
- **Val CE: 0.59** (PPL ≈ 1.8)
- **Reconstruction**: near-perfect through 256-dim latent bottleneck, word order largely preserved
- **Interpolation**: smooth typological transitions (English ↔ Polish)
- **σ=0.5 perturbation**: paraphrastic variation within language
- **TTR: 0.949**, rep_rate: 0.00, mean word length: 5.3, active z dims: 239/256

### Sample outputs (epoch 36):

**Reconstruction:**
```
orig: ðʌ bɔɹdɝ tɛlʌɡɹæf æskt pʌlis skɑtlʌnd waɪ ðʌ foʊtʌɡɹæf hæd nɑt bɪn ɹilist
dec:  ðʌ bɔɹdɝ tɛlʌɡɹæf æskt skɑtlʌnd pʌlis waɪ ðʌ foʊtʌɡɹæf hæd bɪn ɹilist nɑt
```

**Interpolation (English → Polish):**
```
0.00: ðʌ bɔɹdɝ tɛlʌɡɹæf æskt skɑtlʌnd pʌlis waɪ ðʌ foʊtʌɡɹæf hæd bɪn ɹilist nɑt
0.50: faʊndɝ ɡʊd seɪ ðʌ lɪθi ælkʌɡɹeɪʃʌnt ɪz ɔlsoʊ wʌn ʌbaʊt kʌlwarnaʃ ɔf dos rɔʃʌn...
1.00: zaatakɔvali nas faɲi muvjɔnt͡s tɔ dɔpjɛrɔ druɡji t͡sɔ film batma ɔ vɨbiɲɛt͡ɕɛ dɔ rɔlɛ xɔrɨ
```

**Perturbation (σ=0.5):**
```
ðʌ bɔɹdɝ tɛlʌɡɹæf æskt skæŋlʌnd ðʌ sɛfʌleɪ lɪdmʌ vɪ thew ɐlbɛjkɔmɛ ɹiɲɪt͡seʃ lɔledr
```

**Random z:**
```
pɹoʊtɛlz ʌnd daɪl ɲiən wʌt ðʌ ɪnkɹʌpdeɪʃʌnz ʌv noɦɪ popoɟɪ lɛɟɪ v kɛɹi tu nikolɛt ɪn komːe
```

## Agent Game Results

REINFORCE referential game with real LLM embeddings (all-MiniLM-L6-v2, 384-dim, 10K sentences):
- **~95% accuracy** on 16-way discrimination with **100% hard negatives** (within-cluster distractors)
- Chance = 6.25%, **15.2× above random**
- Curriculum: 0% → 100% hard negatives over 500 steps, stable plateau at ~95%
- Batch size 512, converges in ~500 steps
- Variable-length messages (17-19 tokens) via z-norm scaling

### Evaluation Scripts

- `scripts/eval_topology.py` — Semantic topology preservation (input sim → message sim correlation)
- `scripts/eval_compositionality.py` — Topsim, positional disentanglement, diagnostic probes
- `scripts/train_translator.py` — IPA → English LLM translation pilot

Both eval scripts accept `--input_proj data/input_proj.pt` to evaluate a trained projection (vs random baseline).

### Structural Metrics (after curriculum training)

| Metric | Value |
|--------|-------|
| Topsim (hidden cosine) | **0.335** (p≈0) |
| Topsim (token edit) | **0.074** (p≈0) |
| Topology (hidden cosine) | **0.366** (p≈0) |
| Topology (token Jaccard) | **0.202** (p≈0) |
| Probe mean R² | **0.183** |
| Probe dims R²>0 | **100%** |

## Commands

- `poetry install` — Install dependencies
- `poetry install --with generator` — Install with sentencepiece
- `poetry install --with viz` — Install with matplotlib/seaborn (visualization)
- `poetry install --with phonology` — Install with panphon (legacy)
- `poetry install --with translator` — Install with transformers/peft (translation)
- `poetry run pytest` — Run tests (75 tests)
- `poetry run ruff check src/` — Lint
- `poetry run ruff format src/` — Format
- `poetry run lfm visualize --help` — Show visualization commands
- `poetry run lfm visualize tsne --checkpoint data/vae_resume.pt` — t-SNE latent space
- `poetry run lfm visualize all --checkpoint data/vae_resume.pt` — All visualizations
- `poetry run lfm translate generate-pairs` — Generate (IPA, English) pairs
- `poetry run lfm translate train` — Train IPA -> English translator
- `poetry run lfm translate eval --model-dir data/models/v1/translator` — Evaluate translator
- `poetry run lfm visualize translation --results-dir data/models/v1/translator` — Translation viz

## Tech Stack

- Python 3.11+
- PyTorch (GPU compute)
- Pydantic v2 (configuration)
- Poetry (dependency management)
- sentence-transformers (embedding precomputation)
- sentencepiece (subword tokenization)
- epitran (IPA transcription)
- clean-text (corpus sanitization)
- matplotlib + seaborn (visualization)

## Quick Start

```bash
# 1. Pretrain the VAE decoder
python -c "
from lfm.generator.pretrain import pretrain_vae_decoder, VAEPretrainConfig
config = VAEPretrainConfig(
    corpus_loader='leipzig',
    corpus_loader_config={'data_dir': 'data/leipzig'},
)
pretrain_vae_decoder(config)
"

# 2. Precompute embeddings
python scripts/precompute_embeddings.py

# 3. Run the referential game
python scripts/run_referential_reinforce.py
```

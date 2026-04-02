# LFM — Language Faculty Model

## What This Is

LFM gives neural agents a natural language faculty via a pretrained multilingual VAE decoder. Agent embeddings are projected into the VAE's latent space and decoded through a frozen autoregressive transformer into variable-length IPA (International Phonetic Alphabet) utterances — linguistically structured, pronounceable, and compositional output.

The decoder is pretrained on typologically diverse natural language data (currently 12 languages from the Leipzig Corpora Collection, leaf-level phrase constituents), then frozen. During agent training, only a small input projection learns to map agent representations into the decoder's latent space. The linguistic structure acts as a bottleneck that forces compositional, structured communication.

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
- **VAE Pretraining** (`generator/pretrain.py`): Full pipeline with IPA conversion, nucleus sampling, sanitization
- **Dataset Generation** (`data/dataset/`): HDF5 dataset pipeline — load → sanitize → LLM gate → IPA → balance → HDF5
- **Referential Game** (`agents/games/referential.py`): Direct backprop agent game through the linguistic bottleneck (`lfm agent referential`)
- **Expression Game** (`agents/games/expression.py`): GRU-based z-sequence generation with PonderNet halting (`lfm agent expression`)

### Package Structure

```
src/lfm/
  _registry.py          # @register / create() / list_registered()
  _types.py             # Tensor type aliases
  agents/               # Agent communication games
    config.py           # Shared configs (MessageEncoderConfig, CurriculumConfig)
    components.py       # MessageEncoder, Receiver, ZDiversityLoss, ZDistributionLoss
    diffusion.py        # DiffusionZGenerator (flow-matching z-sequence), DenoiserBlock
    decode.py           # rerun_decoder_with_grad (two-phase differentiable decode)
    trainer.py          # AgentTrainer (game-agnostic training loop)
    games/              # Individual game implementations
      referential.py    # ReferentialGame + ReferentialGameConfig
      expression.py     # ExpressionGame + ExpressionGameConfig (GRU or Diffusion z-sequence)
  config/               # LFMBaseConfig, ExperimentConfig
  core/                 # LFMModule (ABC), LFMLoss, CompositeLoss
  expression/           # Learnable expression generation (GRU z-sequence + PonderNet halting, replaces old REINFORCE tree)
    expression.py       # Expression dataclass (topology + decoded output)
    generator.py        # ExpressionGenerator (GRU z-sequence + continuous z-switching decode)
    encoder.py          # ExpressionEncoder (segment pooling + bottom-up Merge)
    config.py           # ExpressionConfig
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
    dataset.py          # lfm dataset {generate,list}
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
    publish.py          # lfm publish {model,dataset}
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
  publish/              # HuggingFace Hub publishing
    base.py             # HFPublisher ABC, ReleaseManifest
    model.py            # ModelRelease (decoder checkpoint)
    dataset.py          # DatasetRelease (IPA corpus)
  data/                 # CorpusDataset, collation, loaders, datasets
    sanitize.py         # SanitizeConfig + configurable text sanitization
    dataset/            # HDF5 dataset generation and reader
      config.py         # DatasetGenerateConfig, LLMGateConfig, ProcessedSample
      generator.py      # DatasetGenerator pipeline
      llm_gate.py       # LLMGatekeeper (accept/fix/reject via small LM)
      manifest.py       # DatasetManifest (YAML metadata)
      reader.py         # DatasetReader (load HDF5 for pretraining)
    loaders/            # Leipzig corpus loader, IPA converter, phonetic distance
  embeddings/           # LLM embedding games, sampler, prefetcher, losses, metrics
  training/             # TrainingLoop, TrainingPhase, Callbacks
  utils/                # Tensor helpers, sampling (Gumbel, straight-through)
  phonology/            # Legacy: phonotactic prior pretraining (research artifact)
```

## Pretraining Results

**v7 (current, full constituency, training)**: 11.6M phrase constituents at ALL tree levels (leaf + internal: NP, VP, PP, ADJP, ADVP, S, SBAR) from 12 languages. No word dropout. z variance regularization (z_var_weight=0.01, target=0.03) + DIP-VAE covariance penalty. max_seq_len=109 (auto-scaled). Batch 80, grad_accum=4 (effective 320). Training in progress.

- **Epoch 1 mid-training CE**: short=0.02, med=0.14, long=0.35
- **Surface diversity**: 100% unique decoded forms for random z draws (64/64)
- **Architecture**: 8 memory tokens, multi-scale attention [3,3,7,7,15,15,full,full], RoPE, weight-shared layers (2 unique × 4)

**v5-leaf-27 (previous, leaf-level phrases)**: 4M leaf-level phrase constituents from 12 languages. Syllable-aligned BPE, max_seq_len=27 (auto-scaled). Val CE: short=0.006, med=0.065.

- **Key finding**: leaf decoder produced 100% unique forms for diverse z, but expression game's `_input_proj` collapsed to 5.8% of z-space → only 1,519/10,000 unique surface forms
- **12 languages**: eng, deu, por, rus, tur, fin, hun, kor, vie, ind, ara, hin
- **Vietnamese recovered**: word-alignment of constituents against parent sentence IPA

## Agent Game Results

### Referential Game

Referential game with direct backprop through the v5 frozen decoder (all-MiniLM-L6-v2, 384-dim, 10K sentences):
- **99.2% best accuracy** on 16-way discrimination with **100% hard negatives** (within-cluster distractors)
- Chance = 6.25%, **15.9× above random**
- Curriculum: 0% → 100% hard negatives over 500 steps
- Batch size 256, converges in ~50 steps to >95%
- Variable-length messages (~41 tokens)
- Two-phase forward: (1) no_grad KV-cached generation, (2) parallel decoder re-run with gradients through cross-attention to latent memory
- Single optimizer: sender_lr=3e-5, receiver_lr=3e-4
- Attention-based message encoder (2-layer self-attention + learned query readout) over decoder hidden states

### Expression Game

Two z-sequence generators available, selectable via `--z-generator`:

**GRU mode** (`--z-generator gru`, default): Autoregressive z-sequence with independent per-segment gates.
- **z_0**: Direct projection of input embedding
- **z_1..z_K**: GRU generates subsequent z vectors conditioned on prior segments
- **Independent gates**: Each segment has its own sigmoid gate (not cumulative PonderNet). Biased closed at init.
- **97.7% peak accuracy** at 100% hard negatives with z diversity regularization (z_sim=0.20)

**Diffusion mode** (`--z-generator diffusion`): Flow-matching denoiser refines all K z positions simultaneously.
- **T=4 reverse steps**: noise → clean z, all differentiable, trained end-to-end through game loss
- **DenoiserBlock**: self-attention across K positions (segments co-adapt) + cross-attention to input embedding
- **Activity head**: per-position sigmoid with Zipfian initialization replaces halt gate
- **Solves by construction**: z-space coverage (noise schedule), segment co-adaptation (parallel refinement), variable length (activity scores)

**Shared components** (both modes):
- Two-phase backprop: no_grad KV-cached decode → parallel decoder re-run with gradients
- Dual-path scoring: surface bottleneck (straight-through token re-embedding) + hidden-state path (annealed)
- ZDistributionLoss: moment-matching keeps projected z spread across decoder's training distribution
- ZDiversityLoss: hinge loss on intra-expression z cosine similarity

**Key diagnostic**: Surface diversity (unique decoded token sequences per N inputs). Measured via `lfm visualize surface-diversity` and logged per-epoch during pretraining.

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
- `poetry install --with datasets` — Install with h5py/num2words (dataset generation)
- `poetry install --with generator` — Install with sentencepiece
- `poetry install --with viz` — Install with matplotlib/seaborn (visualization)
- `poetry install --with phonology` — Install with panphon (legacy)
- `poetry install --with translator` — Install with transformers/peft (translation)
- `poetry install --with publish` — Install with huggingface-hub (publishing)
- `poetry run pytest` — Run tests (160 tests)
- `poetry run ruff check src/` — Lint
- `poetry run ruff format src/` — Format
- `poetry run lfm visualize --help` — Show visualization commands
- `poetry run lfm visualize tsne --checkpoint data/vae_resume.pt` — t-SNE latent space
- `poetry run lfm visualize all --checkpoint data/vae_resume.pt` — All visualizations
- `poetry run lfm translate generate-pairs` — Generate (IPA, English) pairs
- `poetry run lfm translate train` — Train IPA -> English translator
- `poetry run lfm translate eval --model-dir data/models/v1/translator` — Evaluate translator
- `poetry run lfm visualize translation --results-dir data/models/v1/translator` — Translation viz
- `poetry run lfm dataset generate --source leipzig` — Generate HDF5 dataset from Leipzig corpus
- `poetry run lfm dataset generate --source leipzig --no-llm-gate` — Generate without LLM quality gate
- `poetry run lfm dataset list --detail` — List installed datasets with per-language stats
- `poetry run lfm agent referential` — Train referential game (direct backprop)
- `poetry run lfm agent referential --steps 2000 --batch-size 256` — With custom settings
- `poetry run lfm pretrain configs/pretrain_vae_v7.yaml` — Pretrain VAE decoder from config
- `poetry run lfm agent expression` — Train expression game (GRU z-sequence)
- `poetry run lfm agent expression --z-generator diffusion` — Train with diffusion z-sequence
- `poetry run lfm agent expression --z-distribution-weight 1.0` — With z-space coverage matching
- `poetry run lfm agent expression --z-diversity-weight 0.01` — With intra-expression diversity
- `poetry run lfm explore expression-sample --checkpoint data/expression_game/best.pt` — Sample expressions
- `poetry run lfm visualize surface-diversity --checkpoint data/vae_resume.pt` — Surface diversity analysis
- `poetry run lfm publish model --repo-id user/lfm-decoder-v1` — Publish decoder to HuggingFace
- `poetry run lfm publish dataset --repo-id user/lfm-ipa-12lang` — Publish IPA corpus to HuggingFace

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
- h5py (HDF5 dataset storage)
- num2words (number-to-word conversion for sanitization)
- huggingface-hub (model/dataset publishing)

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
poetry run lfm agent referential
```

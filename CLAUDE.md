# LFM — Language Faculty Model

## What This Is

LFM gives any neural system a voice. A frozen multilingual VAE decoder — pretrained on 12 typologically diverse human languages — acts as a linguistic bottleneck: any continuous representation projected into its latent space comes out as linguistically structured, variable-length IPA utterances. The output has the structural properties of natural language (compositionality, morphological regularity, Zipfian statistics, phrase structure) because the decoder learned how these emerge from universal phonotactic constraints. But the *content* is determined by whatever representation was fed in — the source system's native ontology, not human categories.

The system expresses what neural networks perceive. An LLM trained on the emergent language learns to interpret it into English — not word-for-word translation, but interpretation of an alien perspective. The translation reveals how the source system carves up the world.

## Current Status

**Dialogue game V2 training on v7 decoder.** 4-turn multi-turn self-play with progressive topology matching, 92-96% accuracy at 100% hard negatives (16-way, 2048 clusters). The v7 full constituency decoder produces rich ~40 tok/turn phrases. LLM pretraining paused at epoch 2 (loss ~3.7) pending dialogue corpus generation.

### Key Results

- **Dialogue game**: 92-96% accuracy at 100% hard negatives with 4-turn progressive topology matching (v7 decoder, 3 phrases/turn)
- **Expression game**: 95.4% accuracy at 100% hard negatives (16-way, single expression, v7 decoder)
- **Surface diversity**: 299,990/300,000 unique expressions (99.997%) — diffusion z-gen solves diversity by construction
- **LLM pretraining**: Loss 7.6 → 3.7 through epoch 2 on romanized corpus (paused, will regenerate with dialogue corpus)

### What's Working

1. **Frozen decoder** pretrained on 11.6M phrase constituents (v7, val CE=0.0082) produces well-formed output from any input
2. **DiffusionZGenerator** (flow-matching, T=4 steps) produces multi-phrase expressions with near-perfect surface diversity
3. **DialogueGame V2** — 4-turn self-play with bounded VRAM:
   - Progressive topology matching (KL div to embedding cosine similarity structure)
   - Learned turn-position embeddings (simplex-initialized for maximum distinguishability)
   - Learned turn aggregation weights (replace running-mean — broke 88% plateau to 92-96%)
   - Micro-batched generation with adaptive chunking (bounded peak VRAM)
   - Conditional decoder offload and cache flush (only when VRAM tight)
   - VRAM monitor daemon process with stage annotations
   - Multi-target scaffolding (min/max_targets config, individual target cross-attention)
4. **Syllable-hyphenated IPA** as default corpus format — lossless, exposes phonotactic structure to LLM tokenizer via Sonority Sequencing Principle syllabification
5. **Self-supervised training** pipeline: generate-corpus → pretrain → (upcoming) few-shot eval

### What's Next

1. **Let dialogue game finish training** (4000 steps, v7 decoder)
2. **Generate syllable-hyphenated IPA dialogue corpus** — multi-turn documents with turn markers
3. **Resume LLM self-supervised pretraining** on dialogue corpus
4. **Few-shot translation evaluation**
5. **Multi-target discrimination** — set max_targets > 1 for category-level expression
6. **LIGO gravitational wave analysis** — planned domain application (see `docs/ligo-plan.md`)

## Architecture

```
Input Embedding (any dim)
  → DiffusionZGenerator (flow-matching denoiser, T=4 reverse steps)
  → K latent codes (each a "semantic instruction" to the decoder)
  → Frozen PhraseDecoder (RoPE + multi-scale attention + weight sharing)
  → Variable-length IPA token sequence per code
  → Phrases concatenate into full expression
  → Syllable-hyphenate → lossless structured IPA text
  → LLM learns via self-supervised next-token prediction
```

### Key Components

- **PhraseDecoder** (`generator/layers.py`): Frozen transformer with RoPE, multi-scale attention (3/7/15/full windows), weight-shared layers (2 unique × 4)
- **DiffusionZGenerator** (`agents/diffusion.py`): Flow-matching denoiser that produces all K latent codes simultaneously via self-attention + cross-attention to input embedding
- **ExpressionDecoder** (`agents/decode.py`): Reusable multi-phrase autoregressive decode with KV cache persistence across phrase boundaries. Gradient checkpointing on Phase 2 rerun.
- **ExpressionGame** (`agents/games/expression.py`): Referential game training via hidden-state discrimination
- **DialogueGame** (`agents/games/dialogue.py`): V2 multi-turn self-play with:
  - ContextTransformer (cross-attends to target embeddings + turn history)
  - Simplex-initialized turn-position embeddings (maximally equidistant)
  - Progressive topology scoring with learned per-turn aggregation weights
  - Micro-batched generation (adaptive chunk size from VRAM budget)
  - Multi-target support (min/max_targets config for category discrimination)
  - VRAMMonitor daemon process with stage annotations
- **VRAMMonitor** (`agents/vram_monitor.py`): Background process sampling GPU memory every 2s with stage/step annotations, saves .npz traces
- **IPAEncoder** (`agents/components.py`): Token-level encoder for IPA-to-IPA receiver scoring
- **SelfSupervisedTrainer** (`translator/pretrain.py`): Causal LM training on IPA corpus with full resume support
- **CorpusGenerator** (`translator/corpus.py`): Generate IPA corpus with configurable output mode (hyphenated_ipa, romanized, hyphenated_romanized)
- **syllable_hyphenate** (`translator/romanize.py`): Sonority-based syllabification with Maximum Onset Principle and vowelless syllable merging

### Package Structure

```
src/lfm/
  agents/               # Agent communication games
    config.py           # Shared configs (MessageEncoderConfig, CurriculumConfig)
    components.py       # MessageEncoder, Receiver, IPAEncoder, ZDiversityLoss, ZDistributionLoss
    diffusion.py        # DiffusionZGenerator, DenoiserBlock, length_distribution_loss
    refinement.py       # RefinementDenoiser (experimental Phase 2 replacement)
    decode.py           # ExpressionDecoder, rerun_decoder_multiphrase_with_grad
    trainer.py          # AgentTrainer (game-agnostic training loop)
    vram_monitor.py     # VRAMMonitor (background process, .npz traces)
    games/
      referential.py    # ReferentialGame
      expression.py     # ExpressionGame (GRU or Diffusion z-sequence)
      dialogue.py       # DialogueGame V2 (multi-turn, bounded VRAM)
  generator/            # VAE decoder, pretraining
    layers.py           # PhraseDecoder (RoPE + multi-scale attention)
    multilingual_vae.py # MultilingualVAEGenerator
    pretrain/           # VAE pretraining pipeline
    config.py           # GeneratorConfig
  translator/           # Self-supervised LLM translation pipeline
    romanize.py         # IPA → ASCII romanization + syllable_hyphenate
    corpus.py           # CorpusGenerator (expression/dialogue game → text corpus)
    pretrain.py         # SelfSupervisedTrainer (causal LM on IPA corpus)
    pairs.py            # PairGenerator (IPA + English pairs, legacy supervised)
    trainer.py          # TranslatorTrainer (supervised, legacy)
    evaluator.py        # BLEU + semantic similarity evaluation
    config.py           # CorpusConfig, PretrainConfig, TranslatorConfig
  cli/                  # CLI framework (lfm command)
    agent.py            # lfm agent {referential,expression,dialogue}
    translate.py        # lfm translate {generate-corpus,pretrain,generate-pairs,train,eval}
    pretrain.py         # lfm pretrain (VAE decoder)
    visualize/          # lfm visualize {tsne,clustering,attention,...,surface-diversity,all}
    explore.py          # lfm explore {dim-sweep,expression-sample}
  visualize/            # Visualization suite
  data/                 # Corpus datasets, loaders, IPA conversion, syllabify
  embeddings/           # Embedding store (300K sentences, 2048 clusters)
  faculty/              # FacultyConfig + LanguageFaculty compositor
  expression/           # Expression dataclass, encoder (legacy)
```

## Pretrained Decoders

| Model | Dataset | Val CE | max_seq_len | Notes |
|-------|---------|--------|-------------|-------|
| v7 | 11.6M full constituency | 0.0082 | 109 | z_var regularization, current production model |
| v5-leaf-27 | 4M leaf phrases | 0.0071 | 27 | Fast iteration, shorter expressions |

## Game Results

### Expression Game (single expression)

- **95.4% accuracy** at 100% hard negatives (16-way, 2048 clusters, 300K corpus)
- 4 phrases per expression, ~50 tokens total with v7 decoder
- Trained via hidden-state discrimination with topology loss

### Dialogue Game V2 (multi-turn, v7 decoder)

- **92-96% accuracy** at 100% hard negatives (training in progress)
- 4 turns, ~3 phrases/turn, ~40 tokens/turn (~160 total)
- Progressive topology matching: KL divergence to embedding cosine similarity
- Learned turn aggregation weights broke 88% running-mean plateau
- V7 full constituency decoder produces rich syntactic structure per phrase

## Corpus Output Modes

The corpus generator supports three modes via `CorpusConfig.output_mode`:

- **`hyphenated_ipa`** (default): Syllable-hyphenated IPA — lossless, exposes phonotactic structure to LLM tokenizer. Example: `sə-kav-kos sɛj-sjes-tin-kaaɛ ha-koa-ko`
- **`romanized`**: ASCII romanization (legacy, lossy vowel merges). Example: `sekavkos sejsjestinkaae hakoako`
- **`hyphenated_romanized`**: Romanized with syllable hyphens. Example: `se-kav-kos sej-sjes-tin-kaae ha-koa-ko`

Hyphenated IPA is preferred: it preserves all phonemic contrasts (ə≠ɛ, ɑ≠æ) that romanization collapses, and syllable boundaries give the LLM tokenizer natural split points aligned with the decoder's phonotactic knowledge.

## Commands

```bash
# Pretrain VAE decoder
poetry run lfm pretrain configs/pretrain_vae_v7.yaml

# Train expression game (diffusion, YAML config)
poetry run lfm agent expression configs/expression_leaf_phase1.yaml

# Train dialogue game (multi-turn self-play, v7 decoder)
poetry run lfm agent dialogue configs/dialogue_v7_phase1.yaml

# Resume dialogue game from checkpoint
poetry run lfm agent dialogue configs/dialogue_v7_phase1.yaml --resume data/dialogue_game_v7/latest.pt

# Generate syllable-hyphenated IPA corpus for LLM training
poetry run lfm translate generate-corpus \
  --expression-checkpoint data/expression_game_v7-diffusion/best.pt \
  --decoder-path data/models/v7/vae_decoder.pt \
  --spm-path data/models/v7/spm.model \
  --passes 5 --output data/translator/corpus_v7.txt

# Self-supervised LLM pretraining on IPA corpus
poetry run lfm translate pretrain \
  --corpus data/translator/corpus_v7.txt \
  --output-dir data/translator_v7_selfsup \
  --epochs 3 --batch-size 2 --gradient-accumulation-steps 64

# Visualizations
poetry run lfm visualize all --checkpoint data/models/v7/vae_resume.pt
```

## Tech Stack

- Python 3.11+, PyTorch, Pydantic v2, Poetry
- sentence-transformers (embedding precomputation)
- sentencepiece (subword tokenization)
- epitran (IPA transcription)
- transformers + peft (LLM pretraining/fine-tuning)
- matplotlib + seaborn (visualization)
- h5py (HDF5 dataset storage)

## Planned Refactor

- Migrate expression game's `_multiphrase_decode` to use shared `ExpressionDecoder` class
- Build dialogue-aware corpus generator (turn markers, multi-document format)

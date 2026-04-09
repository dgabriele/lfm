# LFM — Language Faculty Model

## What This Is

LFM gives any neural system a voice. A frozen multilingual VAE decoder — pretrained on 12 typologically diverse human languages — acts as a linguistic bottleneck: any continuous representation projected into its latent space comes out as linguistically structured, variable-length IPA utterances. The output has the structural properties of natural language (compositionality, morphological regularity, Zipfian statistics, phrase structure) because the decoder learned how these emerge from universal phonotactic constraints. But the *content* is determined by whatever representation was fed in — the source system's native ontology, not human categories.

The system expresses what neural networks perceive. An LLM trained on the emergent language learns to interpret it into English — not word-for-word translation, but interpretation of an alien perspective. The translation reveals how the source system carves up the world.

## Current Status

**Dialogue corpus generation in progress.** 98.3% best accuracy dialogue game trained on v7 decoder. 900K-document syllable-hyphenated IPA corpus being generated from best checkpoint. LLM pretraining (Qwen 2.5 0.5B) will follow.

### Key Results

- **Dialogue game**: 98.3% best accuracy at 100% hard negatives (16-way, 2048 clusters, v7 decoder, 4 turns × 3 phrases)
- **Expression game**: 95.4% accuracy at 100% hard negatives (16-way, single expression, v7 decoder)
- **Surface diversity**: 299,990/300,000 unique expressions (99.997%) — diffusion z-gen solves diversity by construction
- **Corpus quality**: 100% unique documents, 0% identical turns within documents
- **LLM pretraining**: Loss 7.6 → 3.7 through epoch 2 on romanized corpus (will regenerate with dialogue corpus)

### What's Working

1. **Frozen decoder** pretrained on 11.6M phrase constituents (v7, val CE=0.0082) produces well-formed output from any input
2. **DiffusionZGenerator** (flow-matching, T=4 steps) produces multi-phrase expressions with near-perfect surface diversity
3. **Dialogue game** — 4-turn self-play with bounded VRAM:
   - Detached progressive scoring (each turn's z-gen gradient from its own step only)
   - ContextTransformer (cross-attends to target embeddings + turn history)
   - Simplex-initialized turn-position embeddings (maximally equidistant)
   - Learned turn aggregation weights
   - Cross-turn z diversity (pairwise cosine similarity penalty on mean z-vectors)
   - Micro-batched generation with adaptive chunking (bounded peak VRAM)
   - OOM auto-recovery (catches CUDA OOM, reduces batch size by 10%, retries)
   - Multi-target scaffolding (min/max_targets config, individual target cross-attention)
4. **Syllable-hyphenated IPA** as default corpus format — lossless, exposes phonotactic structure to LLM tokenizer via Sonority Sequencing Principle syllabification
5. **Self-supervised training** pipeline: generate-corpus → pretrain → (upcoming) few-shot eval
6. **Reconstruction pipeline** (`lfm train reconstruction`) — alternative to game-based training: embedding → z-gen → frozen decoder → straight-through tokens → InverseDecoder → reconstructed embedding

### What's Next

1. **Finish dialogue corpus generation** (900K documents, 3 passes × 300K embeddings)
2. **LLM self-supervised pretraining** on dialogue corpus (Qwen 2.5 0.5B)
3. **Cross-lingual bridging** via progressive curriculum:
   - Phase 1: Pure xenoglot (phonotactic/discourse structure)
   - Phase 2: Batch-interleaved xenoglot + English (prevent forgetting)
   - Phase 3: Bilingual documents with cluster anchors (force bridging)
4. **Few-shot translation evaluation**
5. **Reconstruction training** — test inverse decoder approach as alternative/complement to games
6. **Multi-target discrimination** — set max_targets > 1 for category-level expression
7. **LIGO gravitational wave analysis** — planned domain application (see `docs/ligo-plan.md`)

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
- **DialogueGame** (`agents/games/dialogue.py`): Multi-turn self-play with:
  - ContextTransformer (cross-attends to target embeddings + turn history)
  - Simplex-initialized turn embeddings (maximally equidistant conditioning per turn)
  - Detached progressive scoring with learned per-turn aggregation weights
  - Cross-turn z diversity (pairwise cosine similarity penalty)
  - Micro-batched generation with adaptive chunking (bounded peak VRAM)
  - Multi-target support (min/max_targets config for category discrimination)
  - OOM auto-recovery (reduces batch size by 10%, retries)
- **VRAMMonitor** (`agents/vram_monitor.py`): Background process sampling GPU memory every 2s with stage/step annotations, saves .npz traces
- **ReconstructionModel** (`reconstruction/model.py`): Alternative to games — z-gen → decode → straight-through tokens → InverseDecoder → embedding recovery
- **InverseDecoder** (`reconstruction/inverse_decoder.py`): Transformer encoder + learned query readout that recovers embeddings from surface token representations
- **SelfSupervisedTrainer** (`translator/pretrain.py`): Causal LM training on IPA corpus with full resume support
- **DialogueCorpusGenerator** (`translator/dialogue_corpus.py`): Generate multi-turn IPA documents from dialogue game checkpoints
- **BilingualCorpusGenerator** (`translator/bilingual_corpus.py`): Generate cluster-anchored bilingual documents (xenoglot + English) for cross-lingual bridging — each line has `[C{id}] [T0] xenoglot... [EN] English passage`
- **CorpusGenerator** (`translator/corpus.py`): Generate single-expression IPA corpus with configurable output mode (hyphenated_ipa, romanized, hyphenated_romanized)
- **syllable_hyphenate** (`translator/romanize.py`): Sonority-based syllabification with Maximum Onset Principle and vowelless syllable merging

### Package Structure

```
src/lfm/
  agents/               # Agent communication games
    config.py           # Shared configs (MessageEncoderConfig, CurriculumConfig)
    components.py       # MessageEncoder, Receiver, IPAEncoder, ZDiversityLoss, embed_tokens_straight_through
    diffusion.py        # DiffusionZGenerator, DenoiserBlock, length_distribution_loss
    refinement.py       # RefinementDenoiser (experimental Phase 2 replacement)
    decode.py           # ExpressionDecoder, rerun_decoder_multiphrase_with_grad
    trainer.py          # AgentTrainer (game-agnostic training loop, OOM auto-recovery)
    training_history.py # TrainingHistory (per-step scalar metrics as parquet)
    vram_monitor.py     # VRAMMonitor (background process, .npz traces)
    games/
      referential.py    # ReferentialGame
      referential_v2.py # ReferentialV2 (minimal CE-only game for debugging)
      expression.py     # ExpressionGame (GRU or Diffusion z-sequence)
      dialogue.py       # DialogueGame (multi-turn self-play, bounded VRAM)
      document.py       # DocumentGame (single multi-phrase expression)
      multiview.py      # MultiViewGame (N independent expressions per embedding)
  generator/            # VAE decoder, pretraining
    layers.py           # PhraseDecoder (RoPE + multi-scale attention)
    multilingual_vae.py # MultilingualVAEGenerator
    pretrain/           # VAE pretraining pipeline
    config.py           # GeneratorConfig
  reconstruction/       # Reconstruction-based training (alternative to games)
    config.py           # ReconstructionConfig
    inverse_decoder.py  # InverseDecoder (transformer encoder + query readout)
    model.py            # ReconstructionModel (z-gen → decode → inverse → cosine loss)
    trainer.py          # ReconstructionTrainer (simple training loop)
  translator/           # Self-supervised LLM translation pipeline
    romanize.py         # IPA → ASCII romanization + syllable_hyphenate
    corpus.py           # BaseCorpusGenerator, ExpressionCorpusGenerator
    dialogue_corpus.py  # DialogueCorpusGenerator (multi-turn documents with turn markers)
    bilingual_corpus.py # BilingualCorpusGenerator (cluster-anchored xenoglot + English documents)
    pretrain.py         # SelfSupervisedTrainer (causal LM on IPA corpus)
    pairs.py            # PairGenerator (IPA + English pairs, legacy supervised)
    trainer.py          # TranslatorTrainer (supervised, legacy)
    evaluator.py        # BLEU + semantic similarity evaluation
    config.py           # CorpusConfig, BilingualCorpusConfig, PretrainConfig, TranslatorConfig
  cli/                  # CLI framework (lfm command)
    agent.py            # lfm agent {referential,expression,dialogue}
    train.py            # lfm train {reconstruction}
    translate.py        # lfm translate {generate-corpus,generate-dialogue-corpus,generate-bilingual-corpus,pretrain,generate-pairs,train,eval}
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
- **299,990/300,000 unique** surface expressions (99.997%)
- 4 phrases per expression, ~50 tokens total with v7 decoder
- Trained via hidden-state discrimination with topology loss

### Dialogue Game (multi-turn self-play, v7 decoder)

- **98.3% best accuracy** at 100% hard negatives (16-way, 2048 clusters)
- 4 turns per embedding, 3 phrases per turn, ~60 tokens/turn (~240 total)
- Simplex-initialized turn embeddings (maximally equidistant, each turn gets a distinct conditioning vector)
- ContextTransformer: cross-attention from turn embedding to data + accumulated context
- Detached progressive scoring: each turn's z-gen gradient comes only from its own step
- Cross-turn z diversity: pairwise cosine similarity penalty prevents turn collapse
- 100% unique documents, 0% identical turns within documents

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

# Train dialogue game (multi-turn self-play, v7 decoder)
poetry run lfm agent dialogue configs/dialogue_v7_phase1.yaml

# Resume dialogue game from checkpoint
poetry run lfm agent dialogue configs/dialogue_v7_phase1.yaml --resume data/dialogue_game_v7/latest.pt

# Train expression game (single expression, YAML config)
poetry run lfm agent expression configs/expression_leaf_phase1.yaml

# Train reconstruction model (embedding recovery through bottleneck)
poetry run lfm train reconstruction configs/reconstruction_v7.yaml

# Generate dialogue corpus (syllable-hyphenated IPA, multi-turn documents)
poetry run lfm translate generate-dialogue-corpus \
  --dialogue-checkpoint data/dialogue_game_v7/best.pt \
  --decoder-path data/models/v7/vae_decoder.pt \
  --spm-path data/models/v7/spm.model \
  --passes 3 --batch-size 256 --output data/translator/dialogue_corpus_v7.txt

# Generate bilingual corpus (cluster-anchored xenoglot + English documents)
poetry run lfm translate generate-bilingual-corpus \
  --dialogue-checkpoint data/dialogue_game_v7/best.pt \
  --decoder-path data/models/v7/vae_decoder.pt \
  --spm-path data/models/v7/spm.model \
  --passes 3 --batch-size 256 --output data/translator/bilingual_corpus_v7.txt

# Self-supervised LLM pretraining on IPA corpus
poetry run lfm translate pretrain configs/pretrain_dialogue_v7.yaml

# Progressive curriculum (Phase 1 → 2 → 3, each resumes from previous)
poetry run lfm translate pretrain configs/pretrain_curriculum_phase1_v7.yaml
poetry run lfm translate pretrain configs/pretrain_curriculum_phase2_v7.yaml
poetry run lfm translate pretrain configs/pretrain_curriculum_phase3_v7.yaml

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

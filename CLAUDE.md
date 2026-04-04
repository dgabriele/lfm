# LFM — Language Faculty Model

## What This Is

LFM gives any neural system a voice. A frozen multilingual VAE decoder — pretrained on 12 typologically diverse human languages — acts as a linguistic bottleneck: any continuous representation projected into its latent space comes out as linguistically structured, variable-length IPA utterances. The output has the structural properties of natural language (compositionality, morphological regularity, Zipfian statistics, phrase structure) because the decoder learned how these emerge from universal phonotactic constraints. But the *content* is determined by whatever representation was fed in — the source system's native ontology, not human categories.

The system expresses what neural networks perceive. An LLM trained on the emergent language learns to interpret it into English — not word-for-word translation, but interpretation of an alien perspective. The translation reveals how the source system carves up the world.

## Current Status

**Self-supervised LLM pretraining actively running.** A 0.5B parameter LLM (Qwen 2.5) is learning the emergent language via next-token prediction on a 1.5M-expression romanized corpus (28M tokens, 149MB). Loss is falling steadily (7.6 → 4.5 through epoch 1), confirming the LLM recognizes the emergent output as learnable language. Target: loss < 4.0 by end of epoch 3, then few-shot translation evaluation.

### Key Results

- **Surface diversity**: 299,990/300,000 unique expressions (99.997%) from the v7 full-constituency decoder with diffusion z-generator
- **Discrimination**: 95.4% accuracy at 100% hard negatives (16-way, 2048 clusters across 300K sentences)
- **Corpus quality**: Romanized expressions read like natural language with recurring function words, morphological suffixes, and phrase-level structure

### What's Working

1. **Frozen decoder** pretrained on 11.6M phrase constituents (v7, val CE=0.0082) produces well-formed output from any input
2. **DiffusionZGenerator** (flow-matching, T=4 steps) produces multi-phrase expressions with near-perfect surface diversity — solves the z-collapse problem that GRU had
3. **Romanization** (IPA → Latin-script ASCII) activates the LLM's existing cross-lingual transfer
4. **Self-supervised training** pipeline: generate-corpus → pretrain → (upcoming) few-shot eval

### What's Next

1. **Dialogue game** (`lfm agent dialogue`) — multi-turn self-play with Observer/Analyst roles for discourse-structured corpus. Implemented, awaiting smoke test.
2. **Few-shot translation evaluation** — after LLM pretraining converges
3. **LIGO gravitational wave analysis** — planned domain application (see `docs/ligo-plan.md`)

## Architecture

```
Input Embedding (any dim)
  → DiffusionZGenerator (flow-matching denoiser, T=4 reverse steps)
  → K latent codes (each a "semantic instruction" to the decoder)
  → Frozen PhraseDecoder (RoPE + multi-scale attention + weight sharing)
  → Variable-length IPA token sequence per code
  → Phrases concatenate into full expression
  → Romanize → natural-looking Latin-script text
  → LLM learns via self-supervised next-token prediction
```

### Key Components

- **PhraseDecoder** (`generator/layers.py`): Frozen transformer with RoPE, multi-scale attention (3/7/15/full windows), weight-shared layers (2 unique × 4)
- **DiffusionZGenerator** (`agents/diffusion.py`): Flow-matching denoiser that produces all K latent codes simultaneously via self-attention + cross-attention to input embedding
- **ExpressionDecoder** (`agents/decode.py`): Reusable multi-phrase autoregressive decode with KV cache persistence across phrase boundaries
- **ExpressionGame** (`agents/games/expression.py`): Referential game training via hidden-state discrimination
- **DialogueGame** (`agents/games/dialogue.py`): Multi-turn self-play with Observer/Analyst roles and context accumulation
- **RefinementDenoiser** (`agents/refinement.py`): Lightweight diffusion for Phase 2 gradient path (experimental, VRAM-limited)
- **IPAEncoder** (`agents/components.py`): Token-level encoder for IPA-to-IPA receiver scoring
- **SelfSupervisedTrainer** (`translator/pretrain.py`): Causal LM training on romanized IPA corpus with full resume support
- **CorpusGenerator** (`translator/corpus.py`): Generate romanized IPA corpus from trained expression game
- **Romanizer** (`translator/romanize.py`): IPA → natural ASCII orthography

### Package Structure

```
src/lfm/
  agents/               # Agent communication games
    config.py           # Shared configs (MessageEncoderConfig, CurriculumConfig)
    components.py       # MessageEncoder, Receiver, IPAEncoder, ZDiversityLoss, ZDistributionLoss
    diffusion.py        # DiffusionZGenerator, DenoiserBlock, length_distribution_loss
    refinement.py       # RefinementDenoiser (experimental Phase 2 replacement)
    decode.py           # PhraseDecoder, rerun_decoder_multiseg_with_grad
    trainer.py          # AgentTrainer (game-agnostic training loop)
    games/
      referential.py    # ReferentialGame
      expression.py     # ExpressionGame (GRU or Diffusion z-sequence)
      dialogue.py       # DialogueGame (multi-turn self-play)
  generator/            # VAE decoder, pretraining
    layers.py           # PhraseDecoder (RoPE + multi-scale attention)
    multilingual_vae.py # MultilingualVAEGenerator
    pretrain/           # VAE pretraining pipeline
    config.py           # GeneratorConfig
  translator/           # Self-supervised LLM translation pipeline
    romanize.py         # IPA → ASCII romanization
    corpus.py           # CorpusGenerator (expression game → text corpus)
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
  data/                 # Corpus datasets, loaders, IPA conversion
  embeddings/           # Embedding store (300K sentences, 2048 clusters)
  faculty/              # FacultyConfig + LanguageFaculty compositor
  expression/           # Expression dataclass, encoder (legacy)
```

## Pretrained Decoders

| Model | Dataset | Val CE | max_seq_len | Notes |
|-------|---------|--------|-------------|-------|
| v7 | 11.6M full constituency | 0.0082 | 109 | z_var regularization, current production model |
| v5-leaf-27 | 4M leaf phrases | 0.0071 | 27 | Fast iteration, shorter expressions |

## Expression Game Results

**Diffusion z-generator** is the current approach (GRU deprecated for new work):

- **95.4% accuracy** at 100% hard negatives (16-way, 2048 clusters, 300K corpus)
- **299,990/300,000 unique** surface expressions (99.997%)
- 4 phrases per expression, ~50 tokens total with v7 decoder
- Trained via hidden-state discrimination (referential game)
- Surface diversity solved by diffusion noise schedule (no auxiliary losses needed)

## Commands

```bash
# Pretrain VAE decoder
poetry run lfm pretrain configs/pretrain_vae_v7.yaml

# Train expression game (diffusion, YAML config)
poetry run lfm agent expression configs/expression_leaf_phase1.yaml

# Train dialogue game (multi-turn self-play)
poetry run lfm agent dialogue configs/dialogue_phase1.yaml

# Generate romanized IPA corpus for LLM training
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

# Generate translation pairs (for evaluation)
poetry run lfm translate generate-pairs \
  --expression-checkpoint data/expression_game_v7-diffusion/best.pt

# Visualizations
poetry run lfm visualize all --checkpoint data/models/v7/vae_resume.pt
poetry run lfm visualize surface-diversity --checkpoint data/models/v7/vae_resume.pt

# Explore expressions
poetry run lfm explore expression-sample --checkpoint data/expression_game/best.pt
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

- Rename `segment` → `phrase` throughout codebase (segments are phrase constituents)
- Migrate expression game's `_multiseg_decode` to use shared `ExpressionDecoder` class

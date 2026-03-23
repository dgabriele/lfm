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
    layers.py           # LinguisticDecoderLayer (RoPE + multi-scale attention)
    multilingual_vae.py # MultilingualVAEGenerator
    pretrain.py         # VAE pretraining pipeline
    discriminator.py    # StructuralDiscriminator (diagnostic)
    tokenizer.py        # SubwordTokenizer (sentencepiece)
    config.py           # GeneratorConfig
  data/                 # CorpusDataset, collation, loaders
    loaders/            # Leipzig corpus loader, IPA converter, phonetic distance
  embeddings/           # LLM embedding games, sampler, prefetcher, losses, metrics
  training/             # TrainingLoop, TrainingPhase, Callbacks
  utils/                # Tensor helpers, sampling (Gumbel, straight-through)
  phonology/            # Legacy: phonotactic prior pretraining (research artifact)
```

## Pretraining Results

20 epochs on 560K IPA sentences from 16 languages:
- **Val CE: 0.94** (PPL ≈ 2.6)
- **Reconstruction**: near-perfect through 256-dim latent bottleneck
- **Interpolation**: smooth typological transitions (Hungarian ↔ Polish ↔ Vietnamese)
- **σ=0.5 perturbation**: paraphrastic variation within language
- **TTR: 0.96**, rep_rate: 0.00, mean word length: 5.8

### Sample outputs (epoch 20):

**Reconstruction:**
```
orig:  mon văn kuən hut toj ɲiəw xi ciəm ka thɤj zan zɛɲ cɔ kak mon xak
dec:   văn ku mon hut toj xiən ciəm ɲiəw zɛɲ thɤj zan ka kak cɔ saŋ xak
```

**Interpolation (Polish → Vietnamese):**
```
0.00: prɛzɨdɛnt ʂtajn tɔ thɯjatkɔvali faɲit͡ʂnɨ dɔ druɡji...
0.50: tam kucamplɛt vɔŋ xi dɔ zɛɲ cɔ biət to kwok te saŋ bimɛ ɲiəw...
1.00: văn ku mon hut toj xi ɲiəw ciəm thɤj zan zɛɲ ka kak mon cɔ saŋ...
```

**Perturbation (σ=0.5):**
```
ɐkliɕmɨ d͡ʑakarta funkvɲidjijniz tɔ aktɛnliɕmɨ napravljennuu ɡɾinɛlʊs
```

**Random z:**
```
ia prebɪl pre momento pre ninlasikanlas sɛzt͡sɨ a tɯŋ prebɪlnɔɕt͡ɕi pre nin
```

## Agent Game Results

REINFORCE referential game with real LLM embeddings (all-MiniLM-L6-v2, 384-dim):
- **87.5% accuracy** at step 650 (chance = 12.5%, **7× above random**)
- Variable-length messages (17-22 tokens) via z-norm scaling
- Baseline accuracy: 73% and climbing

## Commands

- `poetry install` — Install dependencies
- `poetry install --with generator` — Install with sentencepiece
- `poetry install --with phonology` — Install with panphon (legacy)
- `poetry run pytest` — Run tests (61 tests)
- `poetry run ruff check src/` — Lint
- `poetry run ruff format src/` — Format

## Tech Stack

- Python 3.11+
- PyTorch (GPU compute)
- Pydantic v2 (configuration)
- Poetry (dependency management)
- sentence-transformers (embedding precomputation)
- sentencepiece (subword tokenization)
- epitran (IPA transcription)
- clean-text (corpus sanitization)

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

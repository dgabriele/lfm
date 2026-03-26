# Data Guide

This document describes the data layout, checkpoint structure, and acquisition process for LFM.

## System Dependencies

Before setup, ensure the following are installed:

| Dependency | Purpose | Install |
|-----------|---------|---------|
| Python 3.11+ | Runtime | System package manager |
| Poetry | Package management | `pip install poetry` |
| CUDA 11.8+ | GPU compute | NVIDIA driver + toolkit |
| NLTK data (cmudict) | English IPA conversion | `python -c "import nltk; nltk.download('cmudict')"` |
| g2p-en | English grapheme-to-phoneme | Installed via Poetry |
| epitran | Non-English IPA transcription | Installed via Poetry |

**Note**: `epitran` requires language-specific data files that are downloaded automatically on first use. The first IPA conversion run may be slower due to this initialization.

## Directory Structure

```
data/
  leipzig/                       # Source corpus (Leipzig Corpora Collection)
    {lang}_{source}_{year}_{size}/
      {lang}_{source}_{year}_{size}-sentences.txt
  embeddings/                    # Agent game embeddings (sentence-transformers)
    embeddings.npy               # (N, 384) float32 sentence embeddings
    cluster_labels.npy           # (N,) int32 k-means cluster assignments
    cluster_index.json           # Per-cluster sample indices
    metadata.json                # Embedding model name, dim, count
  models/
    v1/                          # Model version directory (self-contained)
      vae_decoder.pt             # Best decoder-only checkpoint
      vae_resume.pt              # Full training state (for resume)
      spm.model                  # Sentencepiece BPE model
      spm.vocab                  # Sentencepiece vocabulary
      preprocessed_cache.pt      # Tokenized corpus cache
      training_history.json      # Per-session config + epoch history
      config.yaml                # Frozen config snapshot
    v2/                          # Next model version
      ...
  archived/                      # Old checkpoints, not actively used
```

## Checkpoint Files

### `vae_decoder.pt` — Decoder-only checkpoint

Saved when validation CE improves. Used by the agent game and visualizations. Contains:

| Key | Description |
|-----|-------------|
| `latent_dim` | Latent space dimensionality (e.g. 256) |
| `vocab_size` | Sentencepiece vocabulary size (8000) |
| `decoder_hidden_dim` | Decoder model dimension (512) |
| `decoder_num_layers` | Number of decoder layers (4) |
| `decoder_num_heads` | Number of attention heads (8) |
| `max_seq_len` | Maximum sequence length (96) |
| `latent_to_decoder` | State dict: z → decoder memory projection |
| `token_embedding` | State dict: token embedding table |
| `pos_embedding` | State dict: positional embedding (Identity if RoPE) |
| `decoder` | State dict: LinguisticDecoder (multi-scale attention + RoPE) |
| `output_head` | State dict: decoder hidden → logits projection |
| `z_mean` | (latent_dim,) running mean of encoder z distribution |
| `z_std` | (latent_dim,) running std of encoder z distribution |
| `train_loss` | Training loss at save time |
| `val_loss` | Validation loss (the metric that triggered saving) |
| `spm_hash` | SHA-256 prefix of the sentencepiece model for consistency verification |

### `vae_resume.pt` — Full training state

Saved every epoch for resumable training. Contains everything in the decoder checkpoint plus:

| Key | Description |
|-----|-------------|
| `epoch` | Next epoch to resume from |
| `global_step` | Total optimizer steps completed |
| `best_val_loss` | Best validation loss seen so far |
| `spm_hash` | SPM hash for consistency check on resume |
| `modules` | Dict of state dicts for all encoder + decoder modules |
| `optimizer` | AdamW optimizer state (momentum buffers, etc.) |
| `scheduler` | CosineAnnealingLR scheduler state |
| `scaler` | AMP GradScaler state |

### `preprocessed_cache.pt` — Tokenized corpus

Caches the full IPA conversion + tokenization pipeline to skip expensive reprocessing on resume:

| Key | Description |
|-----|-------------|
| `token_ids_list` | List of token ID lists, one per sentence |
| `languages` | List of ISO 639-3 language codes, aligned with token_ids_list |
| `vocab_size` | Sentencepiece vocabulary size |
| `spm_hash` | SPM hash — cache is invalidated if SPM changes |

### `training_history.json` — Session log

Append-only JSON recording each training session (start/resume to completion/kill):

```json
{
  "sessions": [
    {
      "start_epoch": 0,
      "end_epoch": 40,
      "started_at": "2026-03-24T14:46:08+00:00",
      "ended_at": "2026-03-25T02:30:00+00:00",
      "config": { ... full config snapshot ... },
      "best_val_loss": 0.68,
      "spm_hash": "8867185a9ad69a96"
    }
  ]
}
```

## Pre-generated Datasets (HDF5)

The dataset generation system creates reusable, pre-processed HDF5 datasets
from corpus sources. Pre-generated datasets skip inline sanitization and IPA
conversion during pretraining, making experiments faster and reproducible.

### Dataset Layout

```
data/datasets/<name>/
  manifest.yaml             # Dataset metadata (languages, config, stats)
  samples.h5                # Accepted samples (HDF5 with LZ compression)
  rejected.h5               # Rejected samples with reasons (for inspection)
```

### HDF5 Schema (`samples.h5`)

| Dataset | Type | Description |
|---------|------|-------------|
| `seq` | int64 | Global sequence number |
| `language` | string | ISO 639-3 code |
| `source` | string | Corpus source name |
| `source_file` | string | Original filename |
| `raw` | string | Original text |
| `ipa` | string | IPA transcription |
| `ipa_length` | int32 | Character length of IPA |

### Generating Datasets

```bash
# Install dataset dependencies
poetry install --with datasets

# Generate from Leipzig corpus (default settings)
lfm dataset generate --source leipzig

# Custom: specific languages, spell out numbers
lfm dataset generate --source leipzig \
  --languages eng deu pol hin ara \
  --max-samples 50000 \
  --sanitize-number-policy spell_out

# Strict script purity for Hindi
lfm dataset generate --source leipzig \
  --languages hin \
  --sanitize-max-foreign-script-ratio 0.1

# Skip LLM quality gate (faster)
lfm dataset generate --source leipzig --no-llm-gate

# List installed datasets
lfm dataset list
lfm dataset list --detail
```

### Using in Pretraining

Set `dataset_path` in the pretraining config to skip inline preprocessing:

```yaml
# configs/pretrain_vae.yaml
dataset_path: data/datasets/leipzig
```

Or via Python:

```python
config = VAEPretrainConfig(
    dataset_path="data/datasets/leipzig",
)
```

### Sanitization Pipeline

The dataset generator applies configurable rule-based filters:

| Filter | Default | Description |
|--------|---------|-------------|
| `number_policy` | `spell_out` | `reject`/`strip`/`keep`/`spell_out` |
| `symbol_policy` | `spell_out` | Greek/math symbols: `reject`/`strip`/`keep`/`spell_out` |
| `alpha_ratio_min` | 0.7 | Min ratio of alphabetic characters |
| `max_foreign_script_ratio` | 0.3 | Max ratio of foreign-script words |
| `max_word_repetition_ratio` | 0.5 | Reject degenerate repetitive text |
| `require_terminal_punctuation` | true | Require sentence-final punctuation |

### LLM Quality Gate

An optional stage that uses a small LM (Qwen2.5-0.5B) to validate sanitized
text quality. Each sample receives an accept/fix/reject verdict. Rejected
samples are saved to `rejected.h5` for inspection.

Disable with `--no-llm-gate` for faster generation.

## Corpus Acquisition

### Leipzig Corpora Collection

The default corpus source. Provides sentence-level text for 200+ languages.

**Setup via CLI:**

```bash
poetry run lfm setup data --corpus leipzig
```

**Manual setup:**

1. Visit https://wortschatz.uni-leipzig.de/en/download/
2. Download 100K sentence files for the 16 target languages
3. Extract to `data/leipzig/`

**Target languages (16, typologically diverse):**

| Typology | Languages |
|----------|-----------|
| Fusional | Polish, Russian, German, Spanish, Portuguese, Czech, English, Hindi |
| Agglutinative | Turkish, Finnish, Hungarian, Estonian, Korean |
| Isolating | Vietnamese, Indonesian |
| Introflexive | Arabic |

### Embeddings

Pre-computed sentence embeddings for the agent game:

```bash
poetry run python scripts/precompute_embeddings.py
```

Produces 10K sentence embeddings from all-MiniLM-L6-v2 (384-dim) with k-means clustering for hard negative sampling.

## Consistency Verification

The SPM hash mechanism prevents accidental mismatches between checkpoints, caches, and sentencepiece models. On resume, the training script verifies that the checkpoint's `spm_hash` matches the current `spm.model` file. If they differ, training aborts with an error message.

The preprocessed cache also stores `spm_hash` and is regenerated if the SPM model changes.

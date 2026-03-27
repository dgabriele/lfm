# Data Guide

This document describes the data layout, checkpoint structure, dataset generation pipeline, and acquisition process for LFM.

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
  datasets/                      # Pre-generated HDF5 datasets
    leipzig/                     # Base dataset (full sentences only)
      manifest.yaml
      samples.h5
      rejected.h5
    leipzig-constituents/        # Constituency-augmented dataset (~5.75M samples)
      manifest.yaml
      samples.h5
      rejected.h5
      _constituency_checkpoint.jsonl  # Resumable parsing checkpoint
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
    v4/                          # Constituency-augmented model (latent_dim=384)
      ...
  archived/                      # Old checkpoints, not actively used
```

## Checkpoint Files

### `vae_decoder.pt` — Decoder-only checkpoint

Saved when validation CE improves. Used by the agent game and visualizations. Contains:

| Key | Description |
|-----|-------------|
| `latent_dim` | Latent space dimensionality (256 for v1, 384 for v4) |
| `vocab_size` | Sentencepiece vocabulary size (8000) |
| `decoder_hidden_dim` | Decoder model dimension (512) |
| `decoder_num_layers` | Number of decoder layers (4) |
| `decoder_num_heads` | Number of attention heads (8) |
| `max_seq_len` | Maximum sequence length (96) |
| `encoder_num_layers` | Number of encoder layers (1 for v1, 3 for v4) |
| `encoder_pooling` | Pooling strategy (`mean` or `attention`) |
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
| `scheduler` | CosineAnnealingLR scheduler state (per-step cosine) |
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
      "config": { "...full config snapshot..." },
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

| Dataset | Type | Compression | Description |
|---------|------|-------------|-------------|
| `seq` | int64 | gzip | Global sequence number |
| `language` | string | lzf | ISO 639-3 code |
| `source` | string | lzf | Corpus source name |
| `source_file` | string | lzf | Original filename |
| `raw` | string | lzf | Original text |
| `ipa` | string | lzf | IPA transcription |
| `ipa_length` | int32 | gzip | Character length of IPA |

### Generation Pipeline

The `DatasetGenerator` runs a multi-stage pipeline:

```
1. Load        — CorpusLoader.load_detailed() → list[RawSample]
2. Sanitize    — Rule-based filters (SanitizeConfig), multiprocessing
3. LLM Gate    — Optional small LM quality filter (accept/fix/reject)
4. Constituents — Optional constituency parsing (augments with sub-phrases)
5. IPA Convert — epitran (non-English) + CMU dict (English), multiprocessing
6. Balance     — Per-language min/max sample caps
7. Write       — HDF5 (samples.h5 + rejected.h5) + manifest.yaml
```

### Constituency Augmentation

The `--extract-constituents` flag enables phrase-level dataset augmentation via
Stanza constituency parsing. Each sentence is parsed into a tree, and
sub-phrases (NP, VP, PP, ADJP, ADVP, clauses) are extracted as separate
training samples alongside the full sentence.

**Purpose**: Teach the decoder to produce variable-length output — from short
noun phrases (10 tokens) to full sentences (96 tokens). Without this, decoders
trained on full sentences only never learn to produce EOS at short lengths.

**Supported languages** (those with Stanza constituency models):

| ISO 639-3 | Language | Stanza code |
|-----------|----------|-------------|
| deu | German | de |
| eng | English | en |
| spa | Spanish | es |
| ind | Indonesian | id |
| por | Portuguese | pt |
| tur | Turkish | tr |
| vie | Vietnamese | vi |

Unsupported languages (ara, ces, est, fin, hin, hun, kor, pol, rus) pass
through with full sentences only — no constituents are extracted.

**Extracted labels** (Penn Treebank / universal):

| Label | Description |
|-------|-------------|
| S, SBAR, SBARQ, SQ, SINV | Clauses |
| NP, NP-TMP | Noun phrases |
| VP | Verb phrases |
| PP | Prepositional phrases |
| ADJP, ADVP | Modifier phrases |

**Checkpointing**: Constituency extraction is expensive (hours for 1M+
sentences). The generator saves a JSONL checkpoint
(`_constituency_checkpoint.jsonl`) after extraction. On restart, if the
checkpoint exists, it is loaded directly — skipping re-parsing.

**Parallelism**: Each supported language runs in a separate worker process with
its own Stanza pipeline. GPU is used if available (2 workers to fit ~8GB VRAM),
otherwise parallel CPU workers. Signal handlers (SIGTERM/SIGINT) ensure clean
subprocess shutdown.

**Leipzig-constituents dataset stats** (current):

| Metric | Value |
|--------|-------|
| Total samples | 5,750,822 |
| Languages | 16 |
| Rejected | 499,761 |
| Largest | Portuguese (1.17M), English (1.03M) |
| Smallest | Korean (51.8K) |

### Generating Datasets

```bash
# Install dataset dependencies
poetry install --with datasets

# Generate base dataset (full sentences only)
lfm dataset generate --source leipzig

# Generate constituency-augmented dataset
lfm dataset generate --source leipzig \
  --output data/datasets/leipzig-constituents \
  --extract-constituents \
  --max-samples 5000000

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

# Constituency with custom minimum phrase length
lfm dataset generate --source leipzig \
  --extract-constituents \
  --min-constituent-length 15

# List installed datasets
lfm dataset list
lfm dataset list --detail
```

### Using in Pretraining

Set `dataset_path` in the pretraining config to use a pre-generated dataset:

```yaml
# configs/pretrain_vae_v4.yaml
dataset_path: data/datasets/leipzig-constituents
```

Or via Python:

```python
config = VAEPretrainConfig(
    dataset_path="data/datasets/leipzig-constituents",
)
```

When `dataset_path` is set, the pretraining pipeline loads IPA directly from
the HDF5 file via `DatasetReader` — skipping inline corpus loading,
sanitization, and IPA conversion.

### Sanitization Pipeline

The dataset generator applies configurable rule-based filters (`SanitizeConfig`):

| Filter | Default | Description |
|--------|---------|-------------|
| `number_policy` | `spell_out` | `reject`/`strip`/`keep`/`spell_out` |
| `symbol_policy` | `spell_out` | Greek/math symbols: `reject`/`strip`/`keep`/`spell_out` |
| `alpha_ratio_min` | 0.7 | Min ratio of alphabetic characters |
| `max_digit_ratio` | 0.0 | Max digit ratio (0.0 = reject any digits) |
| `max_foreign_script_ratio` | 0.3 | Max ratio of foreign-script words |
| `max_word_repetition_ratio` | 0.5 | Reject degenerate repetitive text |
| `max_bigram_repetition_ratio` | 0.4 | Reject repetitive bigrams |
| `require_terminal_punctuation` | true | Require sentence-final punctuation |
| `min_line_length` | 20 | Minimum character length |
| `max_line_length` | 500 | Maximum character length |
| `min_word_count` | 3 | Minimum word count |
| `strip_urls` | true | Remove URLs |
| `strip_emails` | true | Remove email addresses |
| `strip_phone_numbers` | true | Remove phone numbers |

### IPA Conversion

Multilingual text-to-IPA using `epitran` (rule-based G2P for non-English) and
the CMU Pronouncing Dictionary (English). Language-specific processing:

| Language | Pre-filter | Post-process |
|----------|-----------|--------------|
| Hindi | Strip Latin loanwords, reject >50% code-mixed | Strip leaked Devanagari, word-final schwa deletion |
| Arabic | Strip Latin loanwords, reject >50% code-mixed | Strip leaked Arabic script |
| Others | — | General IPA cleanup (strip non-IPA characters) |

Conversion runs in parallel via `multiprocessing.Pool` (90% of CPU cores).

### LLM Quality Gate

An optional stage that uses a small LM (Qwen2.5-0.5B) to validate sanitized
text quality. Each sample receives an accept/fix/reject verdict:

- **Accept**: Sample passes quality check
- **Fix**: Minor issues correctable by the LLM (returns corrected text)
- **Reject**: Unfixable quality issues (garbled text, wrong language)

Rejected samples are saved to `rejected.h5` for inspection.
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

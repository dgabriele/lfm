# Pretrained Decoder Approach (synth package)

## Problem Being Solved

The current VAE-based decoders reconstruct accurately but fail at coherent generation:
- Mindless repetition and filler tokens when sampling or interpolating
- No long-range grammatical structure (pronoun reference, agreement, phrase composition)
- Token-level CE on short sequences teaches phonotactics, not grammar

Root cause: the VAE bottleneck compresses easy content first (content words, local patterns).
Long-range structure is rounding error in the gradient and gets dropped.

## Core Idea

Grammar lives in a transformer body, not in embedding/output layers.
Subject-verb agreement, anaphora, head direction, scope -- these are encoded in attention
patterns and hidden-state geometry. Surface form (tokenizer, vocabulary, orthography)
lives only in input/output layers.

Swap the lexical surface: keep the grammar machinery, make the surface unrecognizable.

## Base Model: mT5-large (1.2B)

mT5-large is encoder-decoder, 1.2B params, trained on 101 languages.

Why encoder-decoder over decoder-only:
- The encoder naturally processes a source representation; the decoder cross-attends to it
- At inference the mT5 encoder is replaced by a small embedding projection
- Cross-attention conditioning is principled, not bolted-on prefix tokens
- 101-language training means structural priors are genuinely multilingual, not English-dominated

Why 1.2B over smaller:
- Grammar transfer works at small scale; conditioning reliability does not
- At 160-560M, the conditioning adapter cannot reliably map 300K distinct embeddings
  to 300K meaningfully distinct outputs
- 1.2B provides enough capacity for both grammar and fine-grained conditioning

## Alien Vocabulary

A custom subword inventory V_alien of 8K-16K tokens:
- Constructed from CV/CVC syllable templates following universal phonotactic constraints
- Latin characters with a small set of diacritics (circumflex, grave, acute, tilde, umlaut)
- No IPA required: IPA was specific to the VAE decoder's phonemic training objective
- The goal is alien-looking, plausible-sounding tokens that the downstream LLM has no
  prior knowledge of

## Two-Phase Training

### Phase 1: Cipher Fine-Tuning (grammar transfer)

Data source: DepTreeVAE dataset `raw` field (11.6M English constituents).
This is the existing dep_tree_vae training corpus -- filtered, validated English sentences
and phrase constituents (NP, VP, PP, S, SBAR etc.) from Wikipedia and diverse registers.
No IPA or tree structure needed -- just the `raw` text field.

Training task: English text -> alien cipher version of the same text.
The cipher is a deterministic word/morpheme-level mapping into V_alien.
The mT5 encoder processes English, the decoder generates the alien form.
The transformer body re-anchors its grammatical knowledge to alien surface tokens.

After Phase 1: the decoder generates grammatically coherent alien text conditioned
on English encoder input.

### Phase 2: Embedding Conditioning (replace encoder)

Data source: the 300K sentence embedding store (existing, from sentence-transformers).
Each entry: (source_embedding, alien_text) where alien_text = cipher(source_sentence).

Training task: freeze decoder weights from Phase 1. Replace the mT5 encoder entirely
with a learned embedding projector:
  source_embedding (384-dim or 768-dim) -> n_prefix hidden states (model_dim)

The decoder's cross-attention already knows how to attend to encoder hidden states.
We are teaching the projector to produce encoder states that the decoder can use.

Auxiliary length head: small MLP, source_embedding -> predicted token count.
Used at inference to bound generation length.

After Phase 2: the full conditioned generator works.
Input: source embedding. Output: one alien sentence expressing that embedding's content.

## OOD Behavior

When a source embedding is out-of-distribution (unusual domain, different modality,
random vector):
- The decoder's autoregressive prior is always active; it defaults toward common
  patterns in its alien-language distribution rather than producing garbage
- OOD embeddings produce generic but grammatically coherent alien sentences
- Contrast with VAE OOD behavior: decoder has no prior, produces repetition or garbage
- Worst case is "semantically generic but grammatical filler" -- acceptable in a
  pretraining corpus; garbage is not

## Data Flow Summary

  DepTreeVAE raw field (11.6M English)
      -> cipher() -> alien sentences
      -> Phase 1: train mT5-large decoder on (English -> alien)

  300K sentence embeddings + cipher(source sentences)
      -> Phase 2: train embedding projector on (embedding -> alien)

  Trained model (embedding projector + frozen alien decoder)
      -> generate corpus: for each embedding, generate one alien sentence
      -> downstream LLM (Qwen 0.5B) trains from scratch on this corpus

## Implementation: src/lfm/synth/

Quarantined package. Zero imports from the rest of lfm (except lfm.data.dataset.reader
for reading the DepTreeVAE dataset raw field, which is stable and read-only).

Files:
  config.py        -- SynthConfig (all settings, pydantic)
  vocab.py         -- AlienVocab: build CV/CVC token set + tokenizer
  cipher.py        -- WordCipher: deterministic English word -> alien token mapping
  model.py         -- SynthLM: mT5-large with swapped decoder vocab + embedding projector
  trainer.py       -- two-phase trainer (Phase 1 cipher, Phase 2 embedding conditioning)
  generator.py     -- CorpusGenerator: embedding -> alien sentence at scale
  cli.py           -- entry points for lfm synth {build-vocab, train, generate-corpus}

## What Is Preserved vs Lost vs Gained

Preserved:
- Grammatical coherence (anaphora, agreement, phrase structure, long-range dependencies)
- Alienness of surface form to downstream LLM
- Conditionability on arbitrary source embeddings
- Multilingual structural variety (101-language mT5 base)

Lost:
- The emergence aesthetic: grammar no longer emerges from phonotactic constraints alone
- Full architectural control (we rely on mT5 body as a black box)

Gained:
- Coherent generation without repetition/collapse -- guaranteed by decoder prior
- Grammar for free without training from scratch
- Graceful OOD degradation (generic filler, not garbage)
- Much faster path to a usable UNMT pretraining corpus

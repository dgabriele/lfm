# SynthLM Architecture

## Core Intuition

Grammar is already solved — we are re-skinning it.

mT5 was trained on 101 languages. Its transformer body has internalized deep structural priors: what a sentence shape looks like, how clauses nest, how agreement propagates, how length scales with content complexity. That knowledge lives in the weights, not in the vocabulary. The vocabulary is just a surface coating.

SynthLM exploits this. We keep the transformer body and replace the surface. The result is a model that generates grammatically coherent alien text conditioned on any continuous source embedding.

---

## Why Not VAE-Based Generation?

Three VAE architectures all exhibited the same failure mode: token-level CE on phonemic sequences teaches phonotactics, not grammar. The bottleneck compresses content first and drops long-range syntactic dependencies early. The decoder never learns clause structure, agreement, or reference — only local phoneme patterns. The output is phonotactically plausible but grammatically incoherent (phrase repetition, no pronoun reference, no subordination).

The fundamental problem: a VAE decoder trained from scratch has to learn grammar *and* phonotactics *and* content conditioning simultaneously from limited signal. Grammar is the hardest part and loses out.

A pretrained decoder sidesteps this entirely. The grammar is already baked in.

---

## Pipeline

```
Any continuous embedding (384-dim, 768-dim, etc.)
  │
  ▼
[Phase 2] EmbeddingProjector
  Maps embedding → n_prefix fake encoder hidden states
  │
  ▼
mT5 Decoder (frozen after Phase 1)
  Cross-attends to projector output
  Autoregressively generates alien tokens
  │
  ▼
Alien token sequence
  Decoded with WordLevel tokenizer → alien syllable string
```

---

## Two-Phase Training

### Phase 1 — Cipher Fine-Tuning

**Goal:** Transfer mT5's grammatical knowledge to the alien vocabulary.

**Procedure:** Feed English text through the frozen mT5 encoder. Train only the decoder's embed_tokens and lm_head (plus decoder body) to produce the alien cipher equivalent of the English input.

**Intuition:** The decoder is doing what it always did — reading encoder hidden states and producing structured output — but now the output symbols are alien syllables. After Phase 1 the decoder generates grammatically shaped alien text from English context. It has re-anchored its structural habits to the new vocabulary without forgetting the habits themselves.

**What trains:** decoder body + alien embed_tokens + lm_head  
**What freezes:** mT5 encoder

---

### Phase 2 — Embedding Conditioning

**Goal:** Replace the mT5 encoder with a source-embedding projector.

**Procedure:** Freeze the entire decoder. Train only the EmbeddingProjector and LengthHead on (source_embedding, alien_text) pairs. The projector maps a flat embedding to n_prefix fake encoder hidden states; the decoder cross-attends to these as if they were normal encoder output.

**Intuition:** The decoder never notices the swap — it just reads hidden states and generates. Phase 2 only teaches *what* to say; the decoder already knows *how* to say it. This is why Phase 2 converges fast: it is a low-dimensional regression problem sitting on top of a well-initialized generator.

**What trains:** EmbeddingProjector + LengthHead  
**What freezes:** entire mT5 (encoder + decoder)

---

## Alien Vocabulary

The alien vocabulary is a deterministic set of CV and CVC syllables constructed from 17 consonants and 30 vowel variants (standard Latin + common diacritics, no IPA). Vocab size is configurable (default 8K); syllables are seeded for reproducibility.

The WordCipher maps each English word to 1–3 alien syllables via SHA-256 hash of the lowercased word. The mapping is deterministic and stable across runs given the same vocab seed. The cipher is used only to generate training targets for Phase 1 — not at inference time.

**Design constraints:**
- No IPA characters: avoids spurious content-policy triggers in downstream API calls
- Standard Latin diacritics: readable, copy-pasteable, tokenizer-friendly
- Space-separated syllables: each syllable is exactly one WordLevel token

---

## OOD Behavior

For embeddings outside the training distribution, the mT5 decoder's autoregressive prior produces generic-but-grammatical alien text rather than garbage. This is a direct consequence of using a pretrained model: the structural prior acts as a regularizer, and the projector falls back to producing average encoder states when it cannot interpret the embedding confidently.

Contrast with the VAE approach, where OOD inputs produce phonotactically incoherent noise because the decoder has no structural prior to fall back on.

---

## Model Components

| Component | Params (mT5-small) | Params (mT5-large) | Trains in |
|-----------|--------------------|--------------------|-----------|
| mT5 encoder | ~117M | ~600M | Neither phase |
| mT5 decoder + alien embed/lm_head | ~183M | ~600M | Phase 1 only |
| EmbeddingProjector | ~0.5M | ~2M | Phase 2 only |
| LengthHead | ~66K | ~66K | Phase 2 only |

---

## Key Files

```
src/lfm/synth/
  config.py      — SynthConfig (pydantic)
  vocab.py       — AlienVocab + WordLevel tokenizer builder
  cipher.py      — WordCipher (SHA-256 hash mapping)
  model.py       — SynthLM, EmbeddingProjector, LengthHead
  trainer.py     — CipherTrainer (Phase 1), ConditioningTrainer (Phase 2)
  generator.py   — CorpusGenerator (batched inference)

configs/
  synth_local_smoke.yaml     — mT5-small, 200+100 steps (smoke test)
  synth_local_extended.yaml  — mT5-small, 10K+2K steps (local validation)
```

---

## Open Questions

1. **How many Phase 2 steps to convergence?** The projector is low-dimensional but the target space is high-dimensional alien text. Need to measure how quickly lm_loss plateaus under Phase 2 training.

2. **Does Phase 1 length matter?** mT5-small needs far fewer steps to transfer grammar than mT5-large (fewer parameters to retune). The right Phase 1 step count for mT5-large is unknown.

3. **Projector capacity vs. n_prefix_tokens tradeoff.** More prefix tokens give the decoder more context but make the projector harder to train. Current default is 8.

4. **Does the cipher choice matter?** The cipher is a convenient way to produce consistent alien targets from English text. Any consistent word-level mapping should work. The grammar transfer is independent of the specific cipher.

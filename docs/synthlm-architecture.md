# SynthLM Architecture

## What It Is (One Paragraph)

SynthLM takes any continuous vector embedding — from a language model, a vision encoder, a protein model, anything — and generates a grammatically complete sentence in an invented language. The sentence is opaque (not recognizable as any human language) but structurally sound: it has article-like short words, content-word-like longer words, natural length, and coherent internal structure. A downstream LLM trained on a large corpus of such sentences can learn to interpret them via unsupervised machine translation — revealing what the source encoder "meant" without pre-mapping it to human vocabulary.

---

## Where It Fits

The broader LFM goal is to peer into how non-human systems (neural networks) carve up the world, without collapsing their representations into predefined human categories. The pipeline is:

```
Source encoder (any domain)
  → continuous embedding
  → SynthLM generates alien sentence
  → large corpus of (embedding, alien sentence) pairs
  → LLM pretrained on alien corpus
  → LLM learns to interpret alien language via UNMT
  → interpretable description of source encoder's perspective
```

SynthLM is the bridge: it converts the continuous embedding space of any encoder into a structured natural-language surface that an LLM can learn from.

---

## The Core Insight

Grammar is already solved — we are re-skinning it.

Qwen2.5-0.5B was pretrained on hundreds of billions of tokens. Its transformer body has deeply internalized what sentences look like: how clauses nest, how length scales with content, how function words distribute. That knowledge lives in the weights, not the vocabulary.

SynthLM exploits this. We keep the transformer body and replace the vocabulary with an invented alien syllable set. After a short fine-tuning pass, the model generates grammatically coherent alien text. Then we add a small learned projector that maps any input embedding to a set of prefix tokens that condition what the model says. The result: arbitrary embeddings expressed as grammatically complete alien sentences.

---

## Two-Phase Training

### Phase 1 — Alien Language Model

**Goal:** Give the transformer body a fluent prior over the alien vocabulary.

**How:** Standard causal next-token prediction on cipher-encoded English text. Each English word is deterministically mapped to 1–3 alien syllables (SHA-256 hash of the lowercased word). The model sees only alien tokens and predicts the next one — no English input, no cross-modal task. This is just domain adaptation: the body already knows how to model text; it just needs to re-anchor those habits to the alien token set.

**What trains:** `_alien_emb` (alien token embeddings), `_alien_head` (alien LM head), body layers (gentle nudge, low lr).

**What this is NOT:** The previous architecture attempted seq2seq translation (English BPE tokens → alien tokens). That failed because it required the body to process English-shaped input in Phase 1, which then conflicted with Phase 2's prefix conditioning. The alien LM approach produces a body with no prior about what precedes the alien sequence — which is exactly what Phase 2 needs.

**Diagnostic targets:** `lm_acc` 0.45–0.65 (theoretical ceiling ~0.65), `rep_rate` < 0.05, `entropy` > 4.5.

---

### Phase 2 — Embedding Conditioning

**Goal:** Teach the frozen alien LM to respond to source embeddings.

**How:** Prepend 8 learned prefix token embeddings (produced by `PrefixProjector` from the source embedding) to the alien sequence, then run standard teacher-forced LM training. The body, having no prior about what occupies those prefix positions, integrates the prefix as genuine conditioning signal. A `LengthHead` is also trained to predict sentence length from the source embedding, used at inference to set the generation budget.

**What trains:** `PrefixProjector` (source_dim → d_model × n_prefix), `LengthHead` (source_dim → scalar).

**What freezes:** Everything from Phase 1 — alien embeddings, alien head, body.

**Why Phase 2 converges (and why the old arch didn't):** Phase 1 produces a body that has no entrenched expectation about what comes before the alien sequence. Phase 2 prepends a prefix — an extension, not a replacement. In the old seq2seq arch, Phase 1 conditioned the body on English BPE tokens in the source slot. Phase 2 then replaced those with a learned prefix, a distribution shift the frozen body could not accommodate. Phase 2 produced zero learning. The current arch eliminates this entirely.

---

## Alien Vocabulary and Cipher

**`AlienVocab`** (`vocab.py`): 2000 CV/CVC syllable tokens from 17 consonants × 30 Latin+diacritic vowel variants. Fully deterministic from `(vocab_size, seed)`. Special tokens: `[PAD]=0, [UNK]=1, [BOS]=2, [EOS]=3`. EOS is appended to every sequence by the tokenizer post-processor. Tokenizer is a HuggingFace `PreTrainedTokenizerFast` with `WhitespaceSplit` pre-tokenizer (each syllable is exactly one token).

**`WordCipher`** (`cipher.py`): Maps each English word to 1–3 alien syllables via SHA-256 hash of the lowercased word. Words ≤ 2 chars → 1 syllable, 3–5 chars → 2 syllables, ≥ 6 chars → 3 syllables. The mapping is stable across runs given the same vocab. Used only to generate Phase 1 training targets — not at inference time.

**Example:**
```
EN: The economy is growing rapidly this year
AL: Hámzog sâznãrùz fê lïbâjje wàrzédpól sàkâ tévuk [EOS]
```

The alien surface is fully opaque. A downstream LLM has no way to recognize it as English in disguise, which is the point.

---

## Model Components

| Component | Module | Trains in |
|-----------|--------|-----------|
| `CausalDecoderBackend` wrapping Qwen2.5-0.5B | `backend.py` | Phase 1 (body at low lr, alien emb/head at full lr) |
| `PrefixProjector` (source_dim → d_model × n_prefix) | `model.py` | Phase 2 only |
| `LengthHead` (source_dim → scalar) | `model.py` | Phase 2 only |

**Qwen2.5-0.5B:** 0.5B params, 896-dim hidden, 24 layers. Loaded in fp32, alien emb/head randomly initialized. Body fine-tuned at `body_lr=3e-5`; alien emb/head at `phase1_lr=1e-4`.

**Source embedding:** Currently Qwen2.5-0.5B mean-pool of last hidden layer over input tokens (896-dim, L2-normalized). This means Phase 2 conditions on embeddings from the same model family as the generator body — the projector maps within a familiar geometry.

---

## Inference (Phase 2 generation)

```python
# source_embedding: (B, 896) — any encoder's output
context = projector(source_embedding)           # (B, 8, 896) prefix
max_len = length_head(source_embedding) + slack  # predicted budget

for _ in range(max_len):
    hidden  = body(context)
    next_id = alien_head(hidden[:, -1]).argmax(-1)
    context = cat([context, alien_emb(next_id)])
    if all(next_id == EOS): break
```

Greedy decoding. The prefix is the only conditioning — there is no English input at inference time.

---

## Key Files

```
src/lfm/synth/
  config.py      — SynthConfig (pydantic): all hyperparameters
  vocab.py       — AlienVocab (syllable set) + tokenizer builder
  cipher.py      — WordCipher (SHA-256 word → alien syllables)
  backend.py     — DecoderBackend ABC + CausalDecoderBackend (Qwen wrapper)
  model.py       — SynthLM, PrefixProjector, LengthHead
  trainer.py     — AlienLMTrainer (Phase 1), ConditioningTrainer (Phase 2)
  generator.py   — CorpusGenerator (batched Phase 2 inference → corpus file)
  CHANGELOG.md   — architecture decisions, what was tried, quantitative targets

configs/
  synth_local_qwen.yaml   — Qwen2.5-0.5B on RTX 3060 Ti

scripts/
  generate_qwen_embeddings.py  — extract Qwen mean-pool embeddings from Leipzig corpus
```

---

## What the Output Corpus Looks Like

Each line of the generated corpus is one alien sentence conditioned on one source embedding:

```
Hámzog sâznãrùz fê lïbâjje wàrzédpól sàkâ tévuk
Gôt sòjgà pamwãr húfpíghód sof hámzog hèrvàjhë
Dòvèhfü mínüvtìj gôt dergì wimäblá tá kãhkä lòspev hímíz
```

Variable length. Function-word-like short tokens. Content-word-like 3-syllable clusters. Full sentence structure. The LLM pretraining target is next-token prediction on this corpus, exactly like any language model pretraining — the alien surface is the language being learned.

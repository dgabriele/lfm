# SynthLM Changelog

Structured record of architectural approaches, outcomes, and decisions.
Each entry: what was tried → what happened → why → what changed.

---

## ARCH-1: Seq2seq Phase 1 (English BPE → alien syllables)

**Status: ABANDONED**
**Phase 1 result: 80–85% TF accuracy**
**Phase 2 result: zero learning, no convergence**

### What it was
Decoder-only Qwen2.5-0.5B doing cross-modal translation in Phase 1.
Input: English BPE tokens (native Qwen tokenizer).
Target: alien syllable tokens (WordLevel alien tokenizer).
Body processed [English tokens | alien tokens shifted] → predicted next alien token.

### Why Phase 2 failed
Body internalized a strong prior: "English tokens occupy source positions; I generate alien tokens conditioned on them."
Phase 2 replaced English tokens with 8 learned prefix embeddings — a distribution shift the frozen body could not accommodate.
PrefixProjector gradients flowed back through a frozen body that was actively working against prefix conditioning.
Result: projector had no leverage; loss did not move.

### Why we changed Phase 1 (not the real reason at the time)
At the time we cited low free_acc (~0.013–0.024 even after scheduled sampling).
free_acc is actually irrelevant for our goals — see DECISION-1.
The real reason Phase 1 needed to change was Phase 2 failure due to distribution shift.

---

## ARCH-1a: Scheduled Sampling addition (on top of ARCH-1)

**Status: ABANDONED with ARCH-1**

Added SS to try to close the tf_acc / free_acc gap.
- free_acc: 0.004 → 0.024 (marginal)
- tf_acc: plateaued at ~0.35, unmoved

Root cause was not exposure bias but the fundamental memorization nature of the SHA-256 cipher
(no generalization possible, word-level→BPE mismatch, length mismatch).
SS could not fix a task that has no learnable structure beyond memorization.

---

## ARCH-2: Alien LM Phase 1 (current)

**Status: ACTIVE**
**Phase 1 result: loss 6.4→4.1 in 1800 steps; lm_acc=0.362 at step 2000**

### What it is
Phase 1: pure causal next-token prediction on cipher-encoded English text.
Input: alien syllable tokens only — no English tokens anywhere.
Body learns: "given alien token context, predict the next alien token."
Body has no source-position prior, no expectation of what precedes the alien sequence.

### Why prefix conditioning is clean (applies to game approach)
Phase 1 trains on alien tokens with no prefix — the body has no prior on what occupies prefix
positions. Prepending learned prefix tokens is pure extension. This applies equally to the
supervised Phase 2 design (now abandoned) and to the contrastive game approach (current direction):
PrefixProjector gradients flow through a body that is genuinely open to prefix conditioning.

### Convergence vs ARCH-1
ARCH-1 after 20,000 steps: loss ~4.83, tf_acc plateau ~0.355.
ARCH-2 after 1,800 steps: loss 4.15, lm_acc 0.362 — already past ARCH-1 ceiling.
Each step ~31s/200 vs comparable hardware. Alien LM is standard domain adaptation; ARCH-1 was memorization.

### Target diagnostics for Phase 2 readiness
With the 232K word-type vocab, lm_acc is next-word prediction over a large vocabulary — theoretical
ceiling is low (~0.10–0.25). Loss is the primary metric. rep_rate <0.05 and entropy staying high
are equally load-bearing. A fluent diverse prior matters more than raw top-1 accuracy.

---

## DECISION-4: Phase 2 (supervised CE) abandoned — use Phase 1 in contrastive game

**Status: ACTIVE DIRECTION as of 2026-04-28**

### What Phase 2 was
Train a PrefixProjector (2-layer MLP: source_embedding → 8 prefix tokens) via cross-entropy
against `cipher(sentence)` — supervised next-token prediction on the exact alien encoding of the
source sentence.

### Why we are not doing it

**Training signal ambiguity (primary):**
CE against `cipher(sentence)` treats one specific surface form as ground truth. But the source
embedding is a Qwen mean-pool — it encodes semantic content, not a specific sentence. Any alien
paraphrase that preserves the semantics is equally valid, but CE penalises all of them. The model
is optimising toward a single arbitrary target among many valid ones.

**Bijective cipher ceiling:**
The cipher maps English words → alien words deterministically. Training a model to reproduce
cipher(sentence) from sentence_embedding is, at best, learning a noisy inversion of the cipher
table — not genuine semantic grounding. An LLM trained on such a corpus risks learning cipher
decoding rather than semantic interpretation.

**Previous Phase 2 failure (ARCH-1):**
Phase 2 never converged in the seq2seq architecture (ARCH-1). The current decoder-only setup has
better theoretical odds (same-model embeddings, simpler projector, MSE-preserved body geometry)
but the supervised CE approach still has the fundamental ambiguity problem above.

### What we do instead: contrastive game

Use Phase 1 alien LM as a frozen voice box in the existing contrastive discrimination game
framework, directly analogous to how the frozen PhraseDecoder is used in the expression/dialogue
games.

**Architecture:**
- PrefixProjector (same as Phase 2): source_embedding → 8 prefix tokens
- Frozen Phase 1 alien LM: prefix tokens condition autoregressive generation
- Alien token hidden states mean-pooled through frozen Qwen body → alien_embedding (in Qwen's space)
- Contrastive loss: maximize cosine(alien_embedding, source_embedding) vs distractors

**Why this is better:**
- Directly optimises discriminability — what UNMT actually needs
- Eliminates the single-target ambiguity: any alien text whose Qwen encoding matches the source
  embedding is rewarded equally
- The game drives semantic grounding rather than surface reproduction
- The Phase 1 LM's learned phrase-level structure is preserved (frozen body); word order is free
  to organise itself, but clause/phrase patterns remain as information channels
- Round-trip consistency is the objective: Qwen(generated_alien) ≈ source_embedding

**Differentiability:** straight-through token embeddings (`embed_tokens_straight_through`,
already in agents/components.py).

**Key constraint preserved:** Phase 1 LM stays frozen during game training. This is what keeps
the alien language structurally coherent — the LM's autoregressive distribution enforces phrase
patterns; the game only shapes which patterns are activated via prefix conditioning.

---

## DECISION-1: free_acc is not tracked in Phase 1

Phase 1 is never used for standalone generation — all generation goes through Phase 2 prefix conditioning.
The prefix provides strong conditioning that substitutes for exact autoregressive context matching.
The alien cipher is English in disguise; Qwen's pretrained sequence dynamics are robust.
Exposure bias, if it matters at all, is a Phase 2 concern — measure it there against held-out embeddings.
Phase 1 diagnostics: **lm_acc** (distribution quality), **entropy** (vocabulary breadth), **rep_rate** (no degeneracy).

---

## DECISION-2: English-only embeddings for PoC

Multilingual embeddings (LaBSE, paraphrase-multilingual-mpnet) would produce a richer,
less English-centric semantic space — important for the long-term UNMT goal of not
collapsing source encoder ontology to English categories.

English-only chosen for now because:
- Cipher is invertible: any generated alien sentence can be decoded back to verify correctness.
- Architecture proof does not depend on embedding richness.
- Multilingual is a one-line config change once end-to-end pipeline is validated.

---

## DECISION-3: Why cipher-as-English is acceptable

The downstream LLM will not recognize the alien surface as English.
What matters is grammatical coherence of generated sentences, not semantic novelty of the cipher.
The VAE approach (earlier, separate arch) failed because it generated multi-phrase expressions
with no intra-sentence coherence — phrases referenced unrelated things.
The cipher approach guarantees full-sentence grammatical structure because
the source text is grammatically complete English.

---

## CHANGE-1: Cipher word concatenation (morphological boundary fix)

Syllables within a word are now concatenated directly: "economy" → `sâznãrùz` (one space-delimited token)
instead of `sâz nã rùz` (three separate tokens). Word boundaries are preserved via spaces.

Why: the old representation lost word-boundary information — all tokens appeared monosyllabic,
destroying the length-to-word-class signal (short=function word, long=content word) that UNMT
relies on for syntactic role discovery.

Tokenizer consequence: the WordLevel vocab is now built from corpus-derived word types
(all unique alien words found in the training corpus) rather than individual syllables.
`AlienVocab.build_tokenizer(words)` takes the sorted word list; `BuildVocabCommand` scans
the corpus and passes it. Vocab size is now corpus-derived (~50-100K), not fixed at 2000.

---

## SESSION-2026-04-28 digest

- Killed old 300K MiniLM embedding store; generated 1M Qwen2.5-0.5B mean-pool embeddings
  (896-dim, L2-normalized) from Leipzig English corpora (news 2023, Wikipedia 2016, news 2020).
  ~35 minutes at batch=512.
- Fixed cipher surface representation: syllables now concatenated within words (sâznãrùz)
  instead of space-separated (sâz nã rùz). Preserves word-boundary morphological signal for UNMT.
- Rebuilt tokenizer as corpus-derived WordLevel vocab (232K word types) to match new format.
- Config updated: source_embedding_dim 384→896, dataset/store dirs point to data/embeddings_qwen.
- Added checkpoint-boundary diagnostics: 5 English/alien pairs (ground-truth cipher vs model generation).
- lm_acc diagnostic thresholds invalidated by vocab change (2K→232K); loss is now primary metric.
- Phase 1 training restarted from scratch on new corpus.

---

## Surface structure reference

Cipher rules: word ≤2 chars → 1 syllable, 3–5 chars → 2 syllables, ≥6 chars → 3 syllables.
Syllables: CV and CVC tokens from Latin + diacritics (bdfghjklmnprstvwz × aeiou variants).
Vocab size: 2000 (config), seed 42.
EOS appended by tokenizer post-processor.

Example (English → alien cipher):
```
EN: The economy is growing rapidly this year
AL: Hám zog sâz nã rùz fê lï bâj je wàr zéd pól sà kâ té vuk [EOS]
```

Target failure modes (diagnostics should catch these):
- rep_rate high: degenerate repetition (e.g., Hám zog Hám zog Hám zog)
- entropy low: vocabulary collapse
- lm_acc not rising by step 2000–5000: body not learning alien distribution

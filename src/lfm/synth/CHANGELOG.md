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

### Why Phase 2 transition is clean
Phase 2 prepends 8 learned prefix tokens to alien context.
Body has never seen anything in those prefix positions → no conflicting prior.
PrefixProjector gradients flow through a body that is genuinely sensitive to prefix content.
This is extension, not replacement.

### Convergence vs ARCH-1
ARCH-1 after 20,000 steps: loss ~4.83, tf_acc plateau ~0.355.
ARCH-2 after 1,800 steps: loss 4.15, lm_acc 0.362 — already past ARCH-1 ceiling.
Each step ~31s/200 vs comparable hardware. Alien LM is standard domain adaptation; ARCH-1 was memorization.

### Target diagnostics for Phase 2 readiness
Theoretical lm_acc ceiling ~0.55–0.65: word-boundary positions are bounded by English LM uncertainty,
within-word continuations approach ~0.9. Targets: lm_acc <0.35 bad, 0.45–0.55 good, 0.55–0.65 excellent, >0.65 suspect overfit.
rep_rate <0.05 and entropy >4.5 are equally load-bearing — a fluent diverse prior matters more than raw accuracy.

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

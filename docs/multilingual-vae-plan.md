# Multilingual AR DepTreeVAE — Extension Plan

This document captures the architectural and measurement adjustments
required to extend the current English-only AR DepTreeVAE to a
multilingual setting with typologically diverse languages (the v7
12-language corpus: eng, deu, por, rus, tur, fin, hun, kor, vie, ind,
ara, hin).

The current model was developed and validated on English depth-4
dependency trees. The training-time loss terms, the
well-formedness regularizer, and the linguistic-quality measurement
framework were all built around English statistical distributions. Some
pieces transfer cleanly. Some break silently. One actively misbehaves.

This plan separates the three categories and proposes the smallest
viable change for each.

---

## 1. What transfers cleanly (no work)

These metrics and components are language-agnostic by construction:

- **chrF** — character n-gram F-score. Works on Finnish, Vietnamese,
  Korean equally. Already in the periodic checkpoint digest.
- **Distinct-1, Distinct-2, Distinct-3** — counting unique n-grams is
  not language-specific. Already logged.
- **Reconstruction CE** — token-level loss, agnostic to which tokens.
- **Topology loss** — operates on z-distance vs hidden-distance
  correlation, no linguistic content.
- **KL with free_bits, z_var floor** — VAE regularization, language-
  agnostic.

No changes needed.

## 2. What needs per-language treatment (moderate work)

These metrics produce meaningful numbers on each individual language but
break when aggregated across a mixed-language corpus. The numbers
become a weighted average of incommensurable per-language distributions.

| Metric | Why it breaks across languages |
|---|---|
| Mean syllables/word | English ~1.5, Finnish ~5, Vietnamese ~1.0 |
| Mean chars/word | German compounds vs Vietnamese morphemes |
| Short-word fraction | Different morphological types have different distributions |
| Mono-syllabic fraction | Tonal isolating languages skew this hard |
| Bigram coverage | Bigram inventories don't overlap between languages |

**Required changes:**

1. Each sample needs a language ID (one of 12 or whatever set is used).
   The v7 multilingual data pipeline already has this; the AR DepTreeVAE
   data pipeline currently doesn't preserve it from cache → trainer.
2. `lingdiag.py` and the periodic checkpoint digest should accept the
   per-sample language tag and stratify all language-sensitive metrics
   by it. Output: per-language tables of mean_syllables, mean_chars,
   mono_syl_pct, etc.
3. Bigram coverage needs to be computed against per-language reference
   sets, not a corpus-wide set.

These are presentation-layer changes — the model itself doesn't need
modification for these.

## 3. What actively misbehaves cross-lingually (architectural work)

### 3.1 The `corpus_kl_weight` regularizer

The current implementation pulls the model's batch-marginal output
distribution toward a single training-corpus unigram. In a 12-language
training run, that unigram is the mixture of 12 different unigrams. A
model generating clean Finnish would have a unigram that doesn't match
the mixture (it lacks English function words), so KL > 0 and the
gradient pushes toward producing English function words mixed into
Finnish output.

**This is the only place in the current architecture that would cause
cross-lingual interference if reused naively.**

**Required changes:**

1. **Precompute per-language unigrams.** Modify `compute_corpus_unigram.py`
   to optionally read a language tag from each sample and produce
   per-language probability vectors. Output shape changes from `(V,)` to
   `(num_languages, V)` indexed by language ID. Save as
   `unigram_by_lang.npy` plus a `languages.json` mapping ID to ISO code.

2. **Modify model to load per-language unigrams.** In
   `DepTreeVAE.__init__`, when `corpus_kl_weight > 0` and
   `corpus_unigram_path` points at a per-language file, load as a buffer
   shaped `(L, V)` instead of `(V,)`.

3. **Modify `_decode_and_loss` to apply per-sample KL.** The forward
   needs the per-sample language ID. KL becomes:

       for each sample b:
           target = corpus_unigram[language_id[b]]
           marginal_b = (probs_b * mask_b).sum(0) / mask_b.sum()
           kl_b = (marginal_b * (log marginal_b - log target)).sum()
       loss = corpus_kl_weight * mean(kl_b)

   Rather than computing one batch-level marginal, compute per-sample
   marginals (or per-language-group marginals if the batch has multiple
   samples of the same language).

4. **Pass language ID through the data pipeline.** Trainer's batch dict
   needs a `language_id` field; collation needs to populate it from the
   cache. Encoder input may also benefit from a learned language
   embedding added to z.

### 3.2 The skeleton decoder

UD dependency relations are nominally universal, but the *distribution*
over relations varies sharply with morphological type. Finnish heavily
uses `case` (case markers on nouns); English uses `prep` and `obj` more.
Korean and Japanese use particles tagged differently than English.

A single skeleton decoder learning across all 12 languages must capture
this variance from z alone. Two options, in increasing expense:

1. **Language conditioning at the skeleton input.** Add a language
   embedding to `z_struct` (or whatever the skeleton head reads) before
   the projection. Smallest change.
2. **Per-language skeleton heads.** Replace the single
   `skeleton_decoder` with one head per language (or per typology
   family). More expensive but cleaner. Probably overkill.

Recommend option 1 for the first multilingual run.

### 3.3 Decode-time hyperparameters

The autotuned values in the current config (`eos_boost=8.0`,
`expected_len=10`, `ngram_block=[2,3,4]`) were tuned on English. They
need re-tuning per language:

- `expected_len` should be a per-language default reflecting that
  language's typical content-token count per sentence.
- `eos_boost` should reflect the per-language length distribution
  variance.
- N-gram blocking might over-block in agglutinative languages where
  legitimate token repetition is higher.

Cleanest: extend the `DecodeAutotuner` to accept a per-language
validation batch and produce per-language settings. Stored as a dict in
config (`per_language_decode: { eng: {...}, fin: {...}, ... }`). Used at
inference based on the language ID accompanying the latent.

## 4. Architectural primitives that need re-examination

### 4.1 BPE / tokenizer

Currently a single SentencePiece model with 8K vocab trained on English
IPA. For 12 languages, needs to be either:
- One multilingual SPM trained on combined IPA from all 12 languages
  (~32-64K vocab to capture the morphology of agglutinative languages
  without excessive fragmentation).
- Per-language SPMs with a shared decoder. More complex.

The v7 frozen multilingual VAE already solved this — its vocab is
trained on the 11.6M phrase corpus. The AR DepTreeVAE should reuse the
same SPM rather than re-train.

### 4.2 Latent dimensionality

384 dimensions was chosen for English. For 12 typologically diverse
languages, expect that effective rank should grow. Either:
- Increase total_dim to 512 or 768.
- Add language-conditioning so the latent doesn't have to encode
  language identity itself (frees capacity for content).

I'd recommend the latter — explicit language conditioning is cheaper
than a wider latent.

### 4.3 Word dropout schedule

`word_dropout: 0.3 → 0.0` over 4 epochs. With multilingual training the
effective epoch length grows; the schedule should anneal over more
total steps, not more epochs.

## 5. Measurement framework adjustments

### 5.1 Add per-language stratification to `lingdiag.py`

The script currently produces a single posterior + prior generation set
and aggregates across all of it. For multilingual it needs:

- Sample a balanced N-per-language posterior set.
- Sample priors at the same balance.
- Report per-language tables (one block per language) alongside the
  current aggregate.
- Compare posterior to *that-language's* training distribution, not the
  aggregate.

### 5.2 Cross-lingual interference detection

A specific multilingual failure mode worth measuring: does generating
from a `language_id=fin` z accidentally produce English tokens? Add to
`lingdiag.py`:

- For each sample, conditioning on language ID L, compute the fraction
  of generated tokens that appear in *that language's* training
  unigram top-K (vocabulary purity).
- Cross-tabulate: rows = conditioned language, columns = "language of
  most-generated-tokens." If diagonal dominates, generation is
  language-clean. Off-diagonal mass = cross-lingual interference.

This metric directly catches the failure mode that section 3.1 is
designed to prevent.

## 6. Ordering and milestones

Recommended sequence for the multilingual extension:

1. **Data preparation.** Convert v7 multilingual corpus to dep-tree
   depth-filtered cache with per-sample language tags. ~1 day of data
   engineering.

2. **Per-language unigram precompute.** Modify
   `compute_corpus_unigram.py` to produce `(L, V)` tensor + language
   mapping. Tests on the depth-filtered cache.

3. **Model architectural changes** (in order of necessity):
   1. Language ID flowing through forward + loss. Required for any
      per-language conditioning.
   2. Per-language KL target in `corpus_kl_weight`. Eliminates the
      misbehavior in 3.1.
   3. Language embedding added to z (or fed into skeleton head).
      Improves performance.

4. **Measurement updates.** `lingdiag.py` per-language stratification +
   cross-lingual interference metric. Periodic digest gets per-language
   chrF/distinct.

5. **Training.** Restart from scratch; the architectural changes
   require a new run.

6. **Validation.** Run `lingdiag.py` per-language. Bar to clear: each
   language individually has chrF/distinct/mean-syllables roughly
   matching its training distribution. Cross-lingual interference
   metric should show diagonal > 90%.

## 7. Open questions

- Whether to retain the AR DepTreeVAE as the multilingual decoder or
  switch back to the v7 frozen multilingual decoder. The v7 decoder
  was already trained on 11.6M multilingual phrases and worked at 98%
  agent-game accuracy. The AR DepTreeVAE's added structure (skeleton,
  per-role memory) may not be worth the multilingual complexity unless
  empirically validated.
- Whether dependency-tree filtering at depth ≤ 4 is uniform across
  languages. Some languages (e.g., German with separable verb prefixes
  or Korean with extensive verb morphology) may have systematically
  deeper trees that get filtered out unevenly.
- Whether "syllables/word" is the right structural metric for tonal
  languages (Vietnamese, Mandarin), where the syllable count is fixed
  by the writing system but morpheme density varies independently.

## 8. Summary table: changes needed for multilingual

| Component | Change required | Cost |
|---|---|---|
| chrF, Distinct-N | none | 0 |
| Reconstruction CE | none | 0 |
| KL / topo / z_var | none | 0 |
| Mean syllables/chars | per-language stratification at measurement layer | low |
| Bigram coverage | per-language reference sets | low |
| `corpus_kl_weight` | per-language unigram + sample-wise KL target | medium |
| Skeleton decoder | language conditioning at input | medium |
| Decode hyperparameters | per-language autotune | low |
| SPM tokenizer | reuse v7's multilingual SPM | low |
| Latent dim | optionally widen, or rely on language conditioning | low |
| Data pipeline | add language ID to cache + collation | low |
| `lingdiag.py` | per-language stratification + interference metric | low |

The dangerous one is `corpus_kl_weight`. Everything else is either
already language-agnostic or a presentation-layer change. The
multilingual extension is not a fundamental rewrite, but the KL
regularizer would silently degrade quality if reused naively, so its
fix has to be the first architectural piece touched.

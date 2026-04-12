# Neuroglot Phoneme Alphabet: Design Rationale

**Status:** Accepted (v1)
**Artifact:** `data/phoneme_alphabet_multi.json` (50 phonemes)
**Generator:** `scripts/design_phoneme_alphabet_multi.py`
**Related diagnostics:** `scripts/measure_english_stability.py`, `scripts/diagnose_alphabet_alienness.py`

This document records the design journey that produced the current
Neuroglot phoneme inventory. It is written after four iterations, three
of which produced alphabets we explicitly rejected. The purpose is to
make the failure modes of earlier approaches legible so future work can
either improve on them or avoid repeating them.

---

## 1. Problem statement

LFM's pipeline ends with an LLM (Qwen 2.5 0.5B) that is fine-tuned to
interpret emergent-language utterances (Neuroglot) into English
descriptions of what the source neural network perceives. The LLM is
the only component that reads Neuroglot as surface text, so
tokenization is the sole interface between the VAE's continuous latent
space and the LLM's learned representations.

Earlier experiments rendered Neuroglot as IPA or as romanized Latin
text (see `src/lfm/translator/romanize.py`). This produced a measured
**cold-read** Spearman correlation of **ρ = 0.115** between source
embedding geometry and the LLM's latent geometry over Neuroglot
surface (see memory: `project_qwen_latent_results.md`). The
diagnosis: Qwen's BPE tokenizer fragments arbitrary IPA / romanized
output unpredictably. The same phoneme sequence tokenizes one way
word-initially, another way after a space, another way when one
preceding character changes. Topographic structure present in the
continuous latent is shattered at the tokenizer before any learned
layer sees it.

The dominant requirement is therefore **deterministic, context-stable
tokenization**: two Neuroglot words that differ in a single phoneme
must differ in a predictable, local way in Qwen's token ID sequence.
Without this, no amount of downstream training can recover the
topology.

## 2. Goal criteria

A good Neuroglot alphabet must satisfy:

1. **Single-token phonemes.** Each phoneme maps to exactly one Qwen
   token in both space-prefixed (`Ġphoneme`) and bare forms.
2. **Concatenation stability.** When phonemes are concatenated to
   build words, BPE respects the phoneme boundaries in the large
   majority of contexts. We operationalize this as left/right stability
   fractions against a partner set.
3. **Context stability.** A word's token sequence is invariant across
   different surrounding sentences.
4. **Semantic neutrality.** Phonemes do not invoke strong English (or
   code) priors. If a Neuroglot "word" is a real English morpheme Qwen
   recognizes, fine-tuning must fight an existing meaning rather than
   writing into unused representational space.
5. **Damage isolation.** Full fine-tuning perturbs embeddings and
   layer weights. Perturbation should land preferentially on
   capabilities Neuroglot does not need — *not* on Qwen's English
   generation, which is the critical downstream capability.
6. **Architectural leverage.** The tokens should have non-degenerate
   pretrained representations so Qwen's attention / FFN patterns for
   word-like composition fire naturally when reading Neuroglot.
7. **Linguistic plausibility.** Phonemes should be short (2–3 chars)
   and combine into pronounceable-looking strings.

Criteria 4, 5, and 6 are in tension: criterion 6 wants tokens Qwen
*learned*; criteria 4 and 5 want tokens Qwen *doesn't actively use*.
The four design iterations below are progressively better attempts to
resolve this tension.

## 3. Design evolution

We tried four approaches in sequence. The first three failed in
instructive ways.

### 3.1 Approach A — ASCII 2-/3-char pairs with frequency filter

**Script:** `scripts/design_phoneme_alphabet.py` (v1)

**Method.** Enumerate all 2- and 3-letter lowercase-ASCII strings
that Qwen tokenizes to a single token in space-prefixed form. Filter
for BPE concatenation stability (≥ 0.85 left + right product). Drop
two-letter English words, known code acronyms, and hand-curated
English morphemes. Finally, drop anything above the 50th percentile of
token frequency in a mixed English-prose + Stack-code corpus.

**Failure mode.** The stability filter disproportionately selects
tokens that BPE learned *because they are common English morphemes*:
suffixes like `-ion`, `-ure`, `-ful`, `-ery`, roots like `act`, `app`,
`serv`, `cept`. These are stable under concatenation exactly because
BPE saw them often enough to elevate them to single tokens. The
frequency filter is weak defense — morphemes like `-ith`, `-org`, or
`-opp` have low corpus frequency as standalone tokens yet carry
strong semantic priors when Qwen encounters them in morphologically
decomposable contexts.

We maintained growing hand-curated blacklists (`TWO_LETTER_WORDS`,
`CODE_ACRONYM_BLACKLIST`, `ENGLISH_MORPHEME_BLACKLIST`). Each run
surfaced new contamination. The blacklist approach has no principled
stopping point — there is always another morpheme. The fundamental
problem is that the **population from which we sampled (ASCII letter
combinations) is dense with English)**. Filtering cannot escape the
distribution.

### 3.2 Approach B — Vocab-wide rare-token scan

**Script:** `scripts/design_phoneme_alphabet_v2.py`

**Method.** Instead of enumerating ASCII letter combinations, scan
Qwen's entire vocabulary (≈151K tokens across all scripts). Require
single-token status in both space-prefixed and bare forms. Score each
candidate on three rarity signals:

1. prose rate (English corpus)
2. code rate (Stack-smol-xl)
3. LM-head output-weight L2 norm (proxy for how much pretraining
   signal reached that token)

A candidate must pass a percentile cutoff on **all three** signals,
then also satisfy concat stability ≥ 0.85.

**Failure mode.** The rare-everywhere intersection is dominated by
**code-like / technical tokens** — file extensions, rare programming
identifiers, symbol-heavy tokens — and by genuinely exotic-script
tokens that tokenize inconsistently when concatenated with Latin
partners. Widening the script mix introduced new instabilities at
script boundaries that the stability test (which used in-pool
partners) did not catch.

Lesson: *rare overall* is not the right target. We want rare **in
the contexts that matter** (English output), not rare in some global
pretraining-weighted average.

### 3.3 Approach C — Vestigial tokens (bottom percentile of LM-head norm)

**Script:** `scripts/design_phoneme_alphabet_vestigial.py`

**Method.** The hypothesis: tokens whose LM-head output row has very
small L2 norm were trained to rarely-emit during pretraining. These
are "vestigial" — they exist in vocab but carry little learned
function. If we write Neuroglot meaning into their embeddings via
fine-tuning, we destroy nothing, because there was nothing to destroy.
We constrained the pool to Latin-script word-like strings for
script-mode consistency, applied a 15th-percentile norm cutoff, and
added a defensive corpus-rarity check.

**Failure mode.** Vestigial tokens are, almost by definition,
**representationally noisy**. They were rarely produced, which means
Qwen's attention/FFN pathways have not learned to treat them as
word-like units. Downstream, Qwen does not engage its compositional
machinery when reading them — they behave more like unknown glyphs
than like lexical units. This trades criterion 5 (damage isolation)
against criterion 6 (architectural leverage), and loses on 6 harder
than it wins on 5.

Lesson: we don't want *untouched* tokens. We want tokens Qwen
**learned well, but for purposes we are willing to sacrifice**.

### 3.4 Approach D (chosen) — Multilingual Latin-script from non-dominant languages

**Script:** `scripts/design_phoneme_alphabet_multi.py`
**Artifact:** `data/phoneme_alphabet_multi.json`

**Method.** Select tokens characteristic of Latin-script languages
Qwen was exposed to in moderate-but-not-dominant proportion, while
excluding tokens characteristic of the languages we cannot afford to
damage.

Source corpora (positive signal — tokens must be *common* in ≥ 1 of
these):

| Language  | Family        | Leipzig corpus                          |
|-----------|---------------|-----------------------------------------|
| Czech     | IE Slavic     | `ces_news_2022_100K`                    |
| Polish    | IE Slavic     | `pol_news_2023_100K`                    |
| Finnish   | Uralic        | `fin_news_2022_100K`                    |
| Estonian  | Uralic        | `est_news_2022_100K`                    |
| Hungarian | Uralic        | `hun_news_2022_100K`                    |
| Turkish   | Turkic        | `tur_news_2023_100K`                    |
| Indonesian| Austronesian  | `ind_news_2022_100K`                    |

Exclusion corpora (negative signal — tokens must be *rare* in ALL):

| Corpus       | Rationale                                           |
|--------------|-----------------------------------------------------|
| English      | Critical downstream output language                 |
| German       | High Qwen resource, protected                       |
| Spanish      | High Qwen resource, protected                       |
| Portuguese   | High Qwen resource, protected                       |
| Stack code   | Protects technical English generation               |

Filter rule:

```
max_exclusion_rate ≤ 2.0 / M   AND   max_source_rate ≥ 5.0 / M
```

Followed by concat stability ≥ 0.5 against a partner set of real
English words plus other candidate phonemes (this is the usage pattern
that matters — Neuroglot will be abutted to English in carrier
sentences). Constrained to bare lowercase, length 2–3, Latin script.

**Why this works.** The source languages give Qwen enough exposure
that its attention/FFN machinery treats the selected tokens as
word-like units (architectural leverage satisfied). The exclusion
filter guarantees these tokens have minimal direct footprint in
English / German / Spanish / Portuguese / code — so the embedding
rewrite during Neuroglot fine-tuning lands preferentially on
Czech/Finnish/Turkish/etc. fluency, which Neuroglot does not need.

## 4. Stability threshold calibration

Earlier approaches demanded stability ≥ **0.85**. This turned out to
be over-engineered. `scripts/measure_english_stability.py` tests ~100
common English single-token words (`the`, `and`, `water`, ...) under
the same left/right stability protocol used on our candidates, with a
partner set mirroring realistic usage.

Result (paraphrased from script output):

- Median left × right product stability for English: **≈ 0.70**
- 25th percentile:                                    **≈ 0.59**
- 10th percentile:                                    **≈ 0.50**

Real English words tokenize only ~70% consistently across random
concatenations. Requiring 0.85 from Neuroglot phonemes demands *more
stability than natural language itself exhibits*, at the cost of
aggressively shrinking the candidate pool and biasing it toward
aberrantly over-segmented morphemes (exactly the failure of Approach
A).

**Decision:** set threshold to **0.5** — the 10th percentile of
English. This is the principled floor: "no worse than a rare but
real English word". Combined with the multilingual pool of Approach D,
100% of selected phonemes achieve perfect **context stability** in
the final validation (see §6).

## 5. Damage-isolation rationale and its limits

The strongest argument for Approach D is damage isolation. But this
argument has a firm upper bound that the design doc should state
clearly.

**What token choice *does* control.** Fine-tuning reassigns the input
embedding and LM-head row for each Neuroglot phoneme token. If we
choose tokens rarely used by English, the direct embedding-level
overwrite does not touch Qwen's English vocabulary. Writing into
`Ġopr` does not damage `Ġthe`.

**What token choice *does not* control.** Full fine-tuning perturbs
all shared weights — FFN layers, attention projections, layer norms.
These are shared across all tokens. Gradients from the Neuroglot
objective reshape internal circuits regardless of which token IDs the
objective uses. A phoneme choice that isolates direct embedding
damage still leaves Qwen's shared machinery vulnerable to objective
drift.

The bigger lever for cognitive preservation is therefore the
**fine-tuning protocol**, not the token inventory:

- **Interleaved training** — mix Neuroglot objective with an English
  preservation objective every N steps.
- **Cognitive-preservation eval gates** — hold-out English tasks
  that must stay within a tolerance of baseline to accept a
  checkpoint.
- **Lower learning rate** on shared weights vs. on new embedding
  rows (or LoRA-style factorization isolating new capability).
- **Early stopping** on preservation metric drift.

The phoneme alphabet is a *necessary* but *insufficient* step. It
removes one specific damage vector (direct embedding collision on
high-value English tokens); the FT protocol must handle the rest.

## 6. Alphabet composition

Final inventory from `data/phoneme_alphabet_multi.json`:

```
ith iv  opp jav opr uz  ogr jak uk  ust
oz  ell och bib aby vak ott biz uv  ply
baz ilk nd  fab kok ipt ark bij uy  agg
wik wyn bak fid zag pow psz jos ug  ked
hud fon het bab vy  lik py  jan jar hed
```

**50 phonemes**, length distribution **40 × 3-char + 10 × 2-char**.

**Source language distribution** (dominant contributor per phoneme):

| Language  | Count |
|-----------|-------|
| Turkish   | 10    |
| Czech     | 9     |
| Hungarian | 8     |
| Finnish   | 7     |
| Polish    | 7     |
| Indonesian| 7     |
| Estonian  | 2     |

**Validation metrics** (5000 random 3-phoneme words):

- **Unique words generated:** 4911
- **Tokenization length:** 85.7% → 3 tokens, 14.3% → 4 tokens,
  0.06% → 2 tokens
- **Context stability:** **100%** (200/200 words tokenize identically
  across `" the <W> ends here."` vs `" a very <W> appears!"`)

Context stability is the metric that matters downstream — the LLM
must see the same token sequence for the same phoneme string
regardless of surrounding text. At 1.00, this is a qualitative
improvement over IPA/romanized surface, where context stability was
the bottleneck that produced the ρ = 0.115 result.

## 7. Alienness diagnostic

`scripts/diagnose_alphabet_alienness.py` tests whether Cyrillic or
other non-Latin scripts would give a "more alien" (lower-prior)
substrate than rare Latin tokens. It compares four cohorts:

1. ASCII-phoneme (our inventory, 3-phoneme words)
2. ASCII-random (random 8-char lowercase)
3. Cyrillic-random (random 6-char distinct-Cyrillic)
4. English-real (50 real English words, control)

For each, measures per-token NLL and next-token entropy in a generic
carrier sentence.

**Finding:** Cyrillic is *not* substantially more alien than rare
ASCII phonemes. Qwen has enough Russian pretraining that Cyrillic
triggers a "Russian word" prior rather than a "no prior" response,
and the multi-byte tokenization of random Cyrillic actually lowers
per-token NLL relative to rare Latin combinations. The assumption
that "another script = less prior" is empirically wrong for this
model.

**Implication:** The right optimization target is not "most alien
script" but "minimal damage to critical English output capability
while retaining architectural leverage". Approach D optimizes
exactly that.

## 8. Open questions and future work

1. **Corpus choice for frequency cutoffs.** The prose corpus
   (`data/translator/english_corpus.txt`) and Stack code sample are
   small. A larger, balanced prose-plus-code-plus-multilingual corpus
   might give a cleaner exclusion cutoff and let us raise the
   source-rate minimum (currently 5/M), pulling in more distinctly
   minority-language tokens. Candidates like `ith` are borderline —
   they survive the 2.0/M English cutoff but sit closer to the edge
   than we'd prefer.

2. **Tokenization length tail.** 14.3% of 3-phoneme words tokenize
   to 4 tokens rather than 3. This is acceptable (context stability
   is 100%), but a stricter inventory — perhaps at the cost of size —
   could push 3-phoneme words to 100% exactly-3-tokens. Worth
   revisiting if cold-read ρ is still limited by sub-word drift
   after the first training run.

3. **Cognitive-preservation eval gates.** Not implemented. Should
   be built alongside the first Neuroglot FT run so we can measure
   English degradation continuously and abort runs that damage
   critical capabilities. The alphabet minimizes direct embedding
   damage; the gate catches shared-weight damage.

4. **FT protocol design.** Interleaved training with English
   preservation mini-batches; LR separation between new embedding
   rows and shared weights; LoRA vs full FT comparison. These are
   the levers with larger cognitive-preservation effect than the
   token inventory alone.

5. **Alphabet size.** 50 is a guess, not a derivation. The VAE has
   2048 latent clusters and 11.6M phrase constituents; if the
   downstream objective needs more distinct surface units to
   disambiguate clusters, 80–100 phonemes may be warranted. Inverse
   concern: the more phonemes we pull in, the deeper into
   contaminated territory we reach.

6. **Reproducibility.** The source and exclusion corpus files live
   outside the repo (`data/leipzig/...`). Pinning the exact Leipzig
   corpus versions in the alphabet artifact would make the design
   fully reproducible from the JSON alone. Current artifact records
   only the method string.

---

## References

- `src/lfm/translator/romanize.py` — the original IPA/romanized
  surface approach whose cold-read ρ motivated this redesign.
- `/home/daniel/.claude/projects/-home-daniel-projects-lfm/memory/project_qwen_latent_results.md`
  — empirical record of the ρ = 0.115 bottleneck.
- `scripts/design_phoneme_alphabet.py` — Approach A (rejected).
- `scripts/design_phoneme_alphabet_v2.py` — Approach B (rejected).
- `scripts/design_phoneme_alphabet_vestigial.py` — Approach C
  (rejected).
- `scripts/design_phoneme_alphabet_multi.py` — Approach D
  (accepted).
- `scripts/measure_english_stability.py` — calibration of 0.5
  stability threshold.
- `scripts/diagnose_alphabet_alienness.py` — rejection of
  multi-script hypothesis.
- `data/phoneme_alphabet_multi.json` — final artifact.

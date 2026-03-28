# LFM Roadmap

Planned improvements and research directions, roughly prioritized.

---

## 1. Unsupervised constituency induction for all languages

**Status:** Planned

Currently, constituency parsing for variable-length training data is available for 7/16 languages (de, en, es, id, pt, tr, vi) via Stanza. The remaining 9 languages (ar, cs, et, fi, hi, hu, ko, pl, ru) only contribute full sentences, limiting the decoder's ability to produce short phrases in those phonotactic regimes.

**Approach:** Train compound PCFGs (Kim et al., 2019) on the raw Leipzig text for each unsupported language. These learn constituency structure from distributional patterns alone — no annotations needed. The induced trees reflect each language's actual compositional structure rather than projecting English-centric categories.

**Why it matters:** The expression system requires the decoder to produce variable-length sub-expressions across all 16 languages. Without short-phrase training data for Arabic, Hindi, Russian, etc., the decoder can only produce full sentences in those phonotactic patterns, limiting the expression tree's capacity to the 7 parsed languages.

**Alternative:** Dependency subtree extraction (Stanza has dependency parsers for all 16 languages). Headed phrases from dependency trees approximate constituency well enough for training data augmentation.

---

## 2. Learned tree depth

**Status:** Infrastructure in place, not yet trained

The `ExpressionGenerator` supports learned expand/leaf decisions via sigmoid logits, with `min_depth` forcing expansion at shallow depths. The halt decision is a Bernoulli action trained via REINFORCE — high reward when correct with a shallow tree incentivizes the agent to use the minimum structure needed. Initial experiments should validate with `min_depth=1` before enabling fully learned depth.

---

## 3. Cross-leaf context during generation

**Status:** Not yet implemented

In the continuous z-switching decode, each leaf's z is generated independently by the `ExpressionGenerator` before decoding begins. The tree topology is decided top-down, and leaf z vectors are projected from their local hidden context — with no knowledge of what other leaves will produce.

A richer approach: condition later leaf z vectors on earlier leaves' decoded output. After the first leaf is decoded (via the continuous AR pass), its hidden states could feed back into the tree generator to inform subsequent leaf z projections. This would let the agent avoid redundancy ("don't repeat what the first leaf already said") and build coherent multi-constituent expressions.

This is naturally compatible with the continuous z-switching architecture — the KV cache already carries context from earlier leaves. The missing piece is feeding that context back to the z-generation stage.

---

## 4. Multi-agent self-play

**Status:** Not started

The current referential game has one sender and one receiver with fixed roles. Multi-agent self-play would have agents alternate roles, developing shared conventions. Different agents might converge on different expression grammars for the same referential task — different tree topologies, different z-space dialects — and the diversity of emergent grammars is itself an interesting research output.

---

## 5. Decoder fine-tuning on emergent language

**Status:** Not started

After agents converge on a communication protocol, the decoder could be fine-tuned on the agent's actual z→output distribution (rather than the original encoder's). This would sharpen the decoder's output for the specific z region the agent uses, potentially improving reconstruction quality and EOS behavior without full retraining.

---

## 6. Translation from expression trees

**Status:** Translation infrastructure exists for flat IPA→English

The existing `lfm translate` pipeline translates flat IPA sequences to English. With the expression system, the decoder produces one continuous IPA stream with z-switch boundaries derived from the tree structure. The translator could receive this stream as-is (a "run-on" alien sentence) or with tree-derived segmentation markers. Whether markers help or hinder translation quality is an empirical question — the translator may discover the structure from phonotactic cues alone.

Bidirectional translation (English → alien IPA) is also supported by the architecture. The same LLM that learns IPA→English can learn the reverse direction, enabling full human-agent dialogue mediated by the emergent language.

---

## 7. Emotional tone and subjective state expression

**Status:** Speculative research direction

Train the decoder on emotionally valenced first-person corpora — angry, calm, kind, neutral variations of the same content. This would create a **tonal manifold** in z-space: regions whose phonological patterns correlate with emotional categories (prosodic structure, word choice, sentence length all differ across emotional registers).

The scientifically interesting test: give agents persistent state across communication rounds, then observe whether their expressions drift into emotion-correlated z regions when internal state changes (high loss, adversarial pressure, prediction error) — *without* being trained on emotional labels. If agents express internal state through a linguistic channel that has emotion-correlated structure, without being told to, that's a genuine finding about the relationship between internal dynamics and linguistic expression.

**Why it matters:** This is a controlled test of the embodied cognition hypothesis applied to language. Cognitive science claims emotional language is grounded in bodily states. LFM lets you test the inverse: if a system has linguistic structure with emotional dimensions (from training data) coupled to internal states (from task dynamics), does emotion-correlated language emerge *without* a body? The answer — yes or no — is an empirical contribution to a philosophical debate, with every variable controlled and measurable.

**Prerequisites:** First-person narrative training data, persistent agent state across rounds, structural analysis tools for z-space tonal clustering.

**Measurement:** Mutual information between agent persistent state and expression structural features (tree topology, z-vector distribution, phonotactic patterns), computed entirely in z-space with zero reference to English. Nonzero MI = something about the agent's internal state is being expressed. Translation to English is a separate interpretive step — the structural signatures exist independently in the alien IPA.

---

## 8. Phonetic feature-aware decoding

**Status:** Panphon integration built, not yet used in decoding

The phonetic embedding module (`generator/phonetic_embeddings.py`) maps BPE tokens to articulatory feature vectors via panphon. This could be used at decode time to bias the decoder toward phonetically coherent output — e.g., when the decoder is uncertain between two tokens, prefer the one that maintains articulatory continuity with the preceding context (coarticulation).

---

## 9. Non-human acoustic data (whale, bird song)

**Status:** Speculative

Map non-human vocalizations to IPA via acoustic-to-articulatory inversion: click trains → plosives, tonal sweeps → vowel formant transitions, frequency bands → place of articulation. The transcription would be lossy but systematic — the same acoustic features that distinguish /p/ from /t/ in human speech can assign IPA labels to whale click categories.

The decoder, trained on human + whale IPA, would learn "whale phonotactics" alongside human phonotactics. The expression system could then generate whale-like segments when encoding marine biology data.

**Risk:** Mixing radically different phonotactic systems may degrade human language quality. Safer approach: train on human languages first, then test whether the frozen decoder can decode whale-mapped IPA at all. If it produces valid output, the mapping is compatible enough to include in training. If not, the decoder rejects it — which is useful diagnostic information about the mapping quality.

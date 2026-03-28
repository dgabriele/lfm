# LFM Roadmap

Planned improvements and research directions, roughly prioritized.

---

## 1. Unsupervised constituency induction for all languages

**Status:** Planned

Currently, constituency parsing for variable-length training data is available for 7/16 languages (de, en, es, id, pt, tr, vi) via Stanza. The remaining 9 languages (ar, cs, et, fi, hi, hu, ko, pl, ru) only contribute full sentences, limiting the decoder's ability to produce short phrases in those phonotactic regimes.

**Approach:** Train compound PCFGs (Kim et al., 2019) on the raw Leipzig text for each unsupported language. These learn constituency structure from distributional patterns alone — no annotations needed. The induced trees reflect each language's actual compositional structure rather than projecting English-centric categories.

**Why it matters:** The tree-structured agent communication requires the decoder to produce variable-length sub-expressions across all 16 languages. Without short-phrase training data for Arabic, Hindi, Russian, etc., the decoder can only produce full sentences in those phonotactic patterns, limiting the tree's expressivity to the 7 parsed languages.

**Alternative:** Dependency subtree extraction (Stanza has dependency parsers for all 16 languages). Headed phrases from dependency trees approximate constituency well enough for training data augmentation.

---

## 2. Learned tree depth (halt signal)

**Status:** Infrastructure in place, not yet trained

The `TreeSender` supports learned expand/leaf decisions via sigmoid halt logits, but initial experiments should validate the architecture with `min_depth=1` (forced root expansion) before enabling fully learned depth. The halt decision is a Bernoulli action trained via REINFORCE — high reward when correct with a shallow tree incentivizes the agent to use the minimum structure needed.

---

## 3. Bottom-up tree generation

**Status:** Not started

Current tree generation is top-down: root context → children. An alternative is bottom-up: generate atomic leaf expressions first, then learn which leaves to compose and in what order. This mirrors how linguistic competence works — you know the words before you know the sentence structure.

Bottom-up would allow the agent to first decide "what to say" (leaf z selection) and then "how to structure it" (composition order), cleanly separating content from syntax.

---

## 4. Variable branching factor

**Status:** Infrastructure supports it (max_children configurable)

The `TreeSender` currently uses binary branching (expand creates left + right children). Natural language has variable branching — a noun phrase can have 1, 2, or 5 modifiers. Extending to categorical branching (0..K children per node) is a config change, but the REINFORCE credit assignment becomes harder with more structural actions per node.

---

## 5. Cross-sibling attention during generation

**Status:** Partially implemented (sibling context in old tree.py, removed in constituency refactor)

Children of the same parent should know about each other to avoid redundancy ("don't repeat what your sibling already said"). Each child's z could be conditioned on the previous siblings' decoded hidden states. This was in the original tree-of-expressions design but removed when switching to constituency-style (leaves only). Should be re-added for the leaf generation step — later leaves conditioned on earlier siblings' content.

---

## 6. Multi-agent self-play

**Status:** Not started

The current referential game has one sender and one receiver with fixed roles. Multi-agent self-play would have agents alternate roles, developing shared conventions. The tree communication architecture is well-suited for this — different agents might converge on different tree grammars for the same referential task, and the diversity of emergent grammars is itself an interesting research output.

---

## 7. KV cache for tree generation

**Status:** KV cache exists for single-sequence decode, not yet adapted for tree

The `LinguisticDecoder` has KV caching (12x speedup for single-statement decode). For tree generation, each node is decoded independently — the KV cache could be shared across sibling nodes that have the same prefix (they share the parent's context). This would reduce decode cost for deep trees.

---

## 8. Decoder fine-tuning on emergent language

**Status:** Not started

After agents converge on a communication protocol, the decoder could be fine-tuned on the agent's actual z→output distribution (rather than the original encoder's). This would sharpen the decoder's output for the specific z region the agent uses, potentially improving reconstruction quality and EOS behavior without full retraining.

---

## 9. Translation from emergent tree language

**Status:** Translation infrastructure exists for flat IPA→English

The existing `lfm translate` pipeline translates flat IPA sequences to English. Extending it to tree-structured output would require the translator to process multiple leaf expressions plus their compositional structure. A tree-to-sequence model (or simply concatenating leaves with structural delimiters) could feed into the existing LLM translation pipeline.

---

## 10. Emotional tone and subjective state expression

**Status:** Speculative research direction

Train the decoder on emotionally valenced first-person corpora — angry, calm, kind, neutral variations of the same content. This would create a **tonal manifold** in z-space: regions whose phonological patterns correlate with emotional categories (prosodic structure, word choice, sentence length all differ across emotional registers).

The scientifically interesting test: give agents persistent state across communication rounds, then observe whether their expressions drift into emotion-correlated z regions when internal state changes (high loss, adversarial pressure, prediction error) — *without* being trained on emotional labels. If agents express internal state through a linguistic channel that has emotion-correlated structure, without being told to, that's a genuine finding about the relationship between internal dynamics and linguistic expression.

**Why it matters:** This is a controlled test of the embodied cognition hypothesis applied to language. Cognitive science claims emotional language is grounded in bodily states. LFM lets you test the inverse: if a system has linguistic structure with emotional dimensions (from training data) coupled to internal states (from task dynamics), does emotion-correlated language emerge *without* a body? The answer — yes or no — is an empirical contribution to a philosophical debate, with every variable controlled and measurable.

**Prerequisites:** First-person narrative training data, persistent agent state across rounds, structural analysis tools for z-space tonal clustering.

**Measurement:** Mutual information between agent persistent state and expression structural features (tree topology, z-vector distribution, phonotactic patterns), computed entirely in z-space with zero reference to English. Nonzero MI = something about the agent's internal state is being expressed. Translation to English is a separate interpretive step — the structural signatures exist independently in the alien IPA.

---

## 11. Phonetic feature-aware decoding

**Status:** Panphon integration built, not yet used in decoding

The phonetic embedding module (`generator/phonetic_embeddings.py`) maps BPE tokens to articulatory feature vectors via panphon. This could be used at decode time to bias the decoder toward phonetically coherent output — e.g., when the decoder is uncertain between two tokens, prefer the one that maintains articulatory continuity with the preceding context (coarticulation).

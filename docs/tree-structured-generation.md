# Tree-Structured Generation (v14 plan)

**Status:** design sketch.  Not implemented.  Assumes v13's
phrase-type-tagged decoder is trained and the agent game has been
re-run on it, so we have a measured topology baseline against which
to evaluate this architecture.

## Summary

Replace the current "K parallel z vectors → K independent phrases"
generation pipeline with a **two-stage tree-structured generator**:

1. **Stage 1 — Tree scaffold.**  A model conditioned on the input
   embedding produces a constituency tree: an ordered set of typed
   phrase nodes (`S`, `NP`, `VP`, `PP`, `SBAR`, …) with parent/child
   edges.  Interior nodes are structural; leaf nodes each carry a
   content-latent vector.
2. **Stage 2 — Leaf decoding.**  For each leaf, its content-latent
   is passed through the **frozen v13 phrase decoder**, producing a
   surface-text span wrapped in its phrase-type tag.  Leaves are
   concatenated in tree order to form the full expression.

The tree itself is learned; its leaves are decoded by the existing
decoder infrastructure.  No change to the decoder.

## Why this is likely better than v13's K-parallel-z

* **Syntactic well-formedness by construction.**  Concatenating a
  tree's leaves in order produces a syntactically valid utterance
  by definition.  The current architecture has no such guarantee —
  two sibling z's can decode to a VP and an NP that don't compose.
* **Hierarchical compositional semantics.**  A VP node's content-
  latent is generated conditioned on its parent S and sibling NP.
  That is proper compositionality — the VP "knows" which sentence
  it is inside and which subject it is attached to.  Flat-K
  generation has only parallel siblings, forcing the decoder to
  reconstruct hierarchy from uncorrelated z's at generation time.
* **Matches the v13 training corpus.**  v13 deliberately preserves
  constituency structure and phrase-type tags.  Tree-structured
  generation is the architecture for which that signal was
  prepared.
* **Controllable structure.**  You can prompt the generator with
  constraints on tree shape — "one declarative S containing a WH-
  clause" — and the rest fills in.  Current architecture has no
  structural prompt.
* **Interpretable intermediate state.**  The tree is a readable
  artifact.  Humans can point at a node and correct it ("that
  should be an NP, not a complement clause"), feeding directly
  into the perception-mediated-training framework.

## Architecture

Two candidate generation flavors for the tree scaffold:

### Flavor A: Autoregressive over BFS-ordered tree

A transformer decoder emits one tree-node token per step, in
breadth-first traversal order.  Each token is a tuple:

```
(phrase_label, is_leaf, content_latent)
```

`content_latent` is only meaningful when `is_leaf=True`.  The
decoder maintains an open-slot queue (initialized with "need one
root") and pops/pushes as it emits.  Stops when the queue is empty.

* **Pros:** standard transformer-decoder plumbing; teacher-forcing
  training is straightforward; variable-depth trees handled
  naturally.
* **Cons:** sequential — wall-time scales with tree size.
* **Training:** cross-entropy on phrase labels + is_leaf, regression
  on leaf content-latents (target = the v13 encoder's embedding of
  that leaf's surface).
* **Inference:** beam search / sampling over tree topology,
  deterministic content-latent generation.

### Flavor B: Diffusion over flattened tree tensor

Pad each tree to a maximum of `M` BFS positions.  Each position
holds `(label_logits, is_leaf_flag, content_latent)`.  Noise the
whole tensor with a **position-conditional schedule** that keeps
high SNR on root/near-root positions longer than on leaves — so
large-scale structure commits first, fine detail last.

* **Pros:** parallel denoising steps, fast inference; naturally
  non-autoregressive; explicit control over which level of the
  tree commits when.
* **Cons:** padding + masking is fiddly; variable depth becomes a
  ragged-tensor problem; research territory.
* **Training:** DDPM-style with the coarse-to-fine noise schedule.
* **Inference:** N diffusion steps → well-formed tree.

### Recommendation

Build **Flavor A first** to validate the tree-gen → leaf-decode
pipeline on downstream metrics.  If topology ρ improves over v13's
flat-K baseline, invest in Flavor B for inference speed and the
coarse-to-fine structural control it enables.

## Training data

The v13 constituency corpus is already exactly the right shape.
Each training example becomes:

1. Source embedding (from a sentence encoder, e.g. SBERT) — the
   conditioning input.
2. Constituency tree for the same sentence (already parsed with
   Stanza for v13) — the target structure.
3. Each leaf's wrapped surface text — used to compute target
   content-latents via the v13 encoder.

The existing `samples.h5` + per-chunk `constituents_*.txt` files
give us trees implicitly via the parent-seq column.  We'd add a
small conversion script that materializes the actual tree objects
per sentence.

## Implementation staging

1. **v13 baseline.**  Finish v13 pretrain.  Train the agent game on
   v13 (flat-K diffusion as now).  Measure topology ρ + downstream
   dialogue accuracy.  This is the number to beat.
2. **Tree materialization.**  Script: for each v13 sentence, build
   a tree object from the parent-seq links in the constituent data
   and serialize it.  Verify integrity on a sample.
3. **Flavor-A generator.**  Transformer decoder in
   `lfm/agents/tree_generator.py`, trained on (source_emb, tree)
   pairs with the loss described above.
4. **Leaf-decoding integration.**  A new `lfm/agents/games/tree.py`
   that plays the dialogue / expression game using the tree
   generator + frozen v13 decoder stack.
5. **Eval.**  Re-run the interpretation-topology eval on v14 vs
   v13.  The primary question is whether ρ goes up.

## Relation to perception-mediated training

Tree-structured generation makes the correction-learning loop
richer: the human can correct not only *what* concept is being
expressed but *how* it is syntactically structured.  Example
corrections:

* "That should be a relative clause, not a coordinated sentence."
* "Express this as a noun phrase, not a full clause."
* "Split the NP into two smaller NPs conjoined with 'and'."

These corrections target tree topology, not surface words.  The
z-generator (tree-model) receives gradient on structural choices,
not just word choices.  This is a richer communication channel
than word-level correction alone — closer to how a language
teacher actually refines a student's phrasing.

## Open questions

* **Does K-parallel-z already hit the ceiling on topology ρ?**  If
  v13's flat-K approach approaches SBERT's upper bound, tree
  structure adds complexity for marginal gain.  Measurement
  determines whether v14 is worth building.
* **How deep should trees be?**  Full Penn Treebank depth is
  expensive (often 10+ levels).  Truncating to depth 3–4 may
  capture most structural benefit with much less compute.
* **How to handle very short utterances (single NP)?**  Tree
  generator needs a graceful degenerate case: produce a single
  leaf-only tree.  Not hard but a design decision.
* **Inference cost.**  If tree generation adds N = tree-size AR
  steps, multi-turn dialogue games become slower.  Flavor B
  addresses this but requires the noise-schedule work.

## Why v14 is the right time, not v13

We need the v13 phrase-tag-aware decoder trained first.  The
decoder provides the surface-form capacity that the tree generator
builds on top of.  Trying to train both simultaneously adds joint-
optimization complexity that isn't necessary if we do them in
sequence.  v13 is also the cleaner baseline against which to
measure v14's structural gains.

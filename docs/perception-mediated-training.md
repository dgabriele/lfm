# Perception-Mediated Training

**Status:** strategic thesis. This is the paradigm LFM is infrastructure
for — the corpus work, the VAE, the dialogue game, the ortho decoder
are all preparation for training that happens here, not there.

## The reframing

Conventional ML supervises **outputs**. You feed the system inputs, you
correct its outputs, and whatever internal perceptions it formed along
the way are an uninspectable byproduct. If it perceives a vortex as a
dog, you only notice when downstream behavior is wrong, and fixing it
means collecting more labeled (vortex, correct-output) pairs.

**Perception-mediated training** supervises the **representation** directly:

1. The system has a voice — via LFM, its internal perception is
   continuously articulated in a form a human can read through an LLM.
2. When the articulation is wrong ("you're calling this a dog, it's a
   vortex"), a correction is delivered in natural language.
3. The correction is encoded in the LLM's semantic space and used as a
   target to adjust the system's articulation — specifically, the
   z-generator that maps its internal representations to language.
4. The gradient reshapes the *region* of representation-space that maps
   to "dog-like" articulations to instead map near "vortex-like"
   articulations. Generalization is automatic because the correction
   targets a semantic category, not a single instance.

The training signal is **referential**: it corrects what the system is
*pointing at*, not what it is *outputting*.

## Preconditions — why LFM is the enabling infrastructure

You can't correct a perception you can't see. Without a faithful
articulation of what the system is perceiving, there is no surface
on which correction can operate. Three LFM properties make this
possible:

- **Topology-preserving language**: similar internal states articulate
  as similar (LLM-interpretable) phrases. Without this, corrections
  don't generalize — each one would only fix a single point in
  representation space.
- **LLM-readable surface**: the articulation lands in a semantic space
  (Qwen's pretrained English embeddings) that humans can interact with
  naturally and that provides a fixed coordinate system for correction
  targets.
- **Differentiable correction path**: z-generator → frozen decoder →
  straight-through token probs → LLM embedding layer → hidden state.
  A cosine-distance loss from current-articulation to
  target-articulation backpropagates to the z-generator and actually
  moves the representation-to-articulation mapping.

## Mechanism

One correction cycle, concretely:

1. The neural system under training produces an internal state `h`
   (e.g. a pooled attention output, a latent activation, a trajectory
   summary) for some stimulus.
2. LFM articulates `h` as a short English phrase *s*. The human reads it.
3. Human says: "No, what you're looking at is a *vortex*."
4. The correction is encoded as `e_target = LLM("vortex in context")`.
   The current articulation's LLM interpretation is `e_current = LLM(s)`.
5. **Update**: `loss = 1 − cos(e_current, e_target)`, backprop through
   straight-through tokens to the z-generator only (decoder + LLM
   frozen).
6. Optional regularizer: replay of unrelated `h'` inputs with their
   previous `s'` articulations as anchors, so one correction doesn't
   drag the rest of the semantic geometry.

Key design:

- Only the z-generator (optionally with a LoRA adapter) is trained.
  The frozen VAE decoder stays frozen; its phonotactic/orthographic
  prior is what makes the articulations well-formed in the first place.
- Corrections accumulate into a small `(h, target_phrase)` dataset and
  get applied in light fine-tuning passes. One-shot online updates
  tend to thrash.
- Framed as a preference optimization, this reduces to **DPO on the
  z-generator with LLM-embedding distance as the reward signal**.

## What this changes methodologically

**Data efficiency.** Each correction carries orders of magnitude more
information than a conventional labeled example. Instead of 1000
(input, label) pairs teaching "vortex is not dog" by statistical
pressure, one correction pins the whole "dog-like" region of
representation space to a new English neighborhood.

**Evaluation shifts.** The metric of interest is not test-set accuracy
on a held-out distribution. It is **teachability**: how few corrections
does it take to align the system's perceptions to a target referential
frame? This is close to measuring sample efficiency of a dialogue rather
than sample efficiency of a dataset.

**Alignment becomes continuous.** You are not attempting to anticipate
every edge case at training time and bake a frozen policy. You are
maintaining an ongoing referential relationship with the system.
Misperceptions get corrected as they surface; the system's semantic
geometry stays up to date with your expectations.

**Teaching resembles pedagogy, not engineering.** Joint attention,
naming, gentle correction, refinement. The same loop a child uses to
learn what things are and how to talk about them. This is a
philosophical win as much as a technical one — it suggests the
bottleneck in deploying neural systems isn't data but *dialogue*.

## Generalization beyond LFM

Any neural system with:
- An internal state you want to supervise
- An LFM-style articulation layer projecting that state into a
  topology-preserving, LLM-readable surface
- A differentiable path from target-embedding back to the state

…can be trained this way. Examples we're already building toward or
will build toward:

- **LIGO gravitational-wave analyzer**: articulates what it thinks it
  sees in the strain data; gets corrected by a domain expert. See
  `docs/ligo-plan.md`.
- **Biological signal decoders** (EEG, neural recordings): articulate
  perceived patterns; clinicians correct.
- **Multimodal perceivers**: articulate what they're attending to in
  an image or scene; corrections refine attention and grounding.

LFM is the shared infrastructure that makes all of these systems
teachable by the same mechanism.

## Open research questions

- **How local is the correction?** We'd like one correction on
  "vortex" to fix the whole category without unlearning adjacent
  correct perceptions (e.g. "tornado"). Replay + LoRA may be
  sufficient; may need explicit locality regularizers.
- **How to avoid mode collapse toward a fixed vocabulary of
  English-common words?** If the optimizer finds that "system" or
  "thing" has low average cosine distance to many targets, corrections
  may push toward generic articulations rather than specific ones.
- **How to compose multiple corrections?** "It's a vortex, but not
  exactly a tornado — it's more of a Rayleigh-Bénard instability."
  Requires the correction gradient to respect finer-than-category
  structure. Possibly solved by multi-target DPO or by richer target
  encodings than single-phrase embeddings.
- **When does the agent push back?** A mature version of this paradigm
  should support the system saying "I see why you'd call it a vortex,
  but the frequency signature isn't consistent with one" — i.e., the
  system has a stake in the referential negotiation. Requires tracking
  the system's confidence in its own articulation, not just producing
  one.
- **Offline bootstrap vs. online pedagogy.** How much conventional
  pretraining is needed before perception-mediated training becomes
  effective? Probably: enough to get articulations that are
  topology-consistent but possibly categorically wrong. Past that
  threshold, corrections do the heavy lifting.

## Intellectual framing: neural onomasiology

The paradigm slots neatly into an existing linguistic tradition.
**Onomasiology** asks: given a concept, what are the ways to lexicalize
it?  It was always the weaker sibling of semasiology (form-to-meaning)
because the "concept" side was ill-defined — taxonomies, primitives,
gut feeling.

LFM is **general neural onomasiology**: the hand-wavy concept is
replaced by a concrete, arbitrary neural representation — whatever
latent state the source system happens to produce.  No metaphysics
required; the "concept" *is* the activation pattern.

Under this framing:

- The frozen VAE decoder is the **lexicalization function**.  Different
  source system → different lexicalizations of the same conceptual
  content.
- **Subword composition** is onomasiology's word-formation mechanism
  (nominalization, compounding, derivation) realized neurally.
  `hopperator = hop + per + ator` is exactly the kind of productive
  coining classical onomasiologists described in natural languages —
  here, emerging from statistical pressure on the decoder.
- **Topology preservation** is the onomasiological invariant: related
  concepts map to related lexicalizations.  Cognitive onomasiologists
  argued this must underlie natural language cross-linguistically; the
  topology diagnostic (ρ) measures it directly.
- **Perception-mediated training** is onomasiological pedagogy.  You
  correct a lexicalization; the concept → form mapping updates.  That
  is how children acquire vocabulary.
- **Cross-system comparison becomes tractable.**  Two different source
  systems (LIGO detector, biosignal decoder, an LLM's own hidden state)
  each lexicalize their perceptions through the same LFM.  What they
  name, and the conceptual structures their naming reveals, becomes
  directly comparable.  This is empirical cross-system onomasiology —
  previously impossible because no shared expressive medium existed.

Positioning-wise, "general neural onomasiology" is sharper and lands in
an intellectual tradition readers can immediately place.

## Why this is the actual thesis

Re-read the project statement in CLAUDE.md: "LFM gives any neural system
a voice. The output has the structural properties of natural language
because the decoder learned how these emerge from universal phonotactic
constraints. But the *content* is determined by whatever representation
was fed in."

The corpus-generation experiments, the dialogue games, the ortho-vs-IPA
bake-off — these are all about *producing a voice of sufficient
quality*. That's the input. The output is: now you can teach any
neural system by talking to it. Perception-mediated training is what
the voice is *for*.

# Why the Language-First Approach Is Underexplored

An analysis of why structured emergent language — rather than mathematical formalization, latent space alignment, or direct NLP classification — remains unexplored as a medium for deriving novel insights from unsupervised empirical representations of dynamical systems.

---

## The question

Why hasn't anyone tried to give agents a language faculty for communicating about dynamical systems, and then translate that language via a pretrained multilingual LLM, rather than using direct mathematical analysis, latent space alignment, or classical NLP parsing?

The answer reveals how deeply certain disciplinary assumptions have calcified — and why no single existing research community would arrive at this approach on its own.

## 1. The math-is-sufficient assumption blocks the idea at the source

Physics and dynamical systems theory have been spectacularly successful for 400 years using mathematical formalization as the sole representational medium. Equations are universal, precise, and perspective-independent. To a physicist, the proposition that a neural agent might perceive regularities in a dynamical system that are easier to express in a structured alien language than in mathematics sounds like mysticism.

But this assumption breaks down exactly where complexity science lives — in systems with so many interacting degrees of freedom that no closed-form description captures the full structure. A system like Lenia with 34 Sobol dimensions, multi-ring kernels, 3 coupled channels, and CFL-adaptive substeps is precisely such a system. The agent isn't replacing math — it's perceiving empirical regularities in a space too high-dimensional for humans to survey, and it needs a medium to externalize those perceptions. Math would work if you knew what to formalize. Language works when you don't.

## 2. The alignment paradigm makes translation seem redundant

In ML interpretability, the universal approach is: map internal representations onto human concepts. Linear probes, TCAV, concept bottleneck models, CLIP-style contrastive alignment — all of these project the agent's world into our ontological space. The community's implicit assumption is that *understanding* means *reduction to familiar categories*.

Translation is the opposite bet: the agent's categories are valuable precisely because they're unfamiliar, and the goal is to access them on their own terms. This is a fundamentally different epistemological commitment. Most ML researchers would not even frame it as a choice — alignment IS interpretability, in their view.

The idea that alignment destroys the very information you're seeking requires a chain of beliefs: that the agent's native ontology contains novel structure, which requires that the agent can perceive things humans can't, which requires taking the agent's perspective seriously as a legitimate empirical viewpoint. That's a philosophical commitment most engineers don't make.

## 3. Emergent communication research has been pessimistic for good reason

Since Lazaridou et al. (2017), the empirical results have been consistent: agents develop degenerate, non-compositional codes. Chaabouni et al. showed that "compositionality" in emergent protocols is often an artifact of measurement. This killed enthusiasm for the idea that useful language-like structure can emerge from communication pressure alone.

LFM's response — impose structural constraints as inductive biases rather than hoping structure emerges — goes against the dominant paradigm of pure emergence. The emergent communication community wants to prove that structure can emerge from nothing. LFM says: structure won't emerge from nothing, but it will emerge from the right constraints plus communication pressure, just like human language emerges from the interaction of universal-grammar-like biases and communicative need. This is essentially the Chomsky-vs-Tomasello debate resolved in a third direction that neither camp would endorse.

## 4. The disciplinary fragmentation is extreme

To conceive of this approach, you need simultaneous expertise in:

- Continuous dynamical systems (PDEs, cellular automata, complexity science)
- VQ tokenization with roundtrip fidelity (representation learning)
- Discrete diffusion / generative modeling (D3PM, denoising processes)
- Emergent communication (Lewis signaling games, referential games)
- Linguistic typology (morphosyntactic universals, WALS typological database)
- LLM translation capabilities and limitations
- Philosophy of science (perspectival realism, situated epistemology)

These communities don't talk to each other. Each has a piece: physicists have the dynamical systems, VQ researchers have the tokenization, the emergent communication community has the agent games, linguists have the typological universals, and LLM researchers have the translation capability. But nobody has assembled the full pipeline because nobody stands at the intersection.

## 5. "Just use English" poisons the well

If you want interpretable agent communication, the path of least resistance is: train the agent to output English. This is what every LLM-based agent system does. It works, it's easy, and the output is immediately human-readable.

The problem — that English imposes English's ontology, its way of carving up the world, its implicit conceptual categories — is invisible unless you're specifically looking for non-human perspectives. If you're doing RL for game playing or robotic control, English output is fine because you don't care about the agent's internal ontology. You only care about the agent's ontology when the agent might be seeing something you can't see — and that requires a specific kind of problem (complex dynamical system) plus a specific epistemic goal (novel scientific insight). Most AI research has neither.

## 6. The objectivity assumption makes perspective seem like contamination

Science values objectivity — perspective-independent truth. An approach that explicitly preserves perspectival descriptions violates the foundational assumption that good science eliminates perspective.

The deeper reason the language-first approach is underexplored is that it requires taking seriously the idea that a situated observer's perspective on a dynamical system could contain valid empirical content that is not reducible to the observer-independent description. That's perspectival realism — a real position in philosophy of science (Massimi, Giere) — but it's not how working scientists think about their own practice.

The word "bias" cuts both ways: the field rightly avoids biasing agent semantics with human concepts, but it also avoids treating agent perspectives as epistemically legitimate. Those are different things, and conflating them forecloses the approach entirely.

## 7. The grounding pipeline didn't exist until now

Before the Spinlock grounding pipeline, the idea of agents communicating about dynamical systems in structured language was purely theoretical. There was no clean, measurable path from:

1. Continuous physical dynamics
2. Discrete tokens with verified roundtrip fidelity
3. A generative model that learns the joint distribution over all token families
4. An agent that can "imagine" (inverse generation) and "perceive" at multiple temporal resolutions

The grounding pipeline makes fidelity measurable at every stage. LFM can only exist because this pipeline exists. And specific architectural features — such as the denoising trajectory functioning as temporal unfolding, where intermediate denoising states correspond to coarse-to-fine perceptions of progressively longer rollouts — map directly onto information structure in the emergent language: what the agent perceives first (parameters, at low noise) becomes given/background information; what resolves last (long-horizon dynamics) becomes focus/new information. These are exactly the information-structural distinctions that syntax modules in LFM are designed to learn.

## 8. Why classical NLP methods don't apply

A natural objection: why not use dependency parsing, constituency parsing, PoS tagging, or other classical NLP methods to learn rules from agent representations?

These methods are discriminative — they label existing structure. They operate on discrete symbols that already have meaning (words in a human language). And they assume the existence of grammar rules that can be described symbolically.

None of these conditions hold for physical dynamical systems:

- There are no pre-existing symbols — the VQ tokenizer *discovers* the discrete vocabulary from continuous dynamics.
- The mapping from parameters to behavior is many-to-many and continuous — there are no grammar rules to extract.
- Physical consistency is a global constraint (all token positions must jointly describe a valid system), not a tree of local decisions.
- The goal is *generation* of novel consistent configurations, not *classification* of existing ones.

A parser can label what's already there. A generative model with language-like structure can imagine what could be there — and express its imagination in a form that's translatable.

## Summary

The language-first approach is underexplored not because it's wrong, but because there's no institutional home for it. It requires simultaneously rejecting three dominant assumptions:

1. That mathematical formalization is sufficient for describing complex dynamical systems
2. That interpretability means alignment with human concepts
3. That scientific objectivity requires eliminating perspective

Each of these rejections is independently defensible. Holding all three at once puts you outside every existing research community — which is precisely why the approach has gone unexplored, and why it might see things that existing approaches can't.

---

## Further reading

- [Spinlock](https://github.com/dgabriele/spinlock) — The grounding pipeline: VQ tokenization with roundtrip fidelity, D3PM discrete diffusion for inverse generation, and the denoising trajectory as temporal unfolding.
- [D3PM Architecture and Training Dynamics](https://github.com/dgabriele/spinlock/blob/main/docs/d3pm-architecture.md) — Technical details of the graded noise schedule, multi-truncation roundtrip loss, and how the denoising trajectory recapitulates physical causality.

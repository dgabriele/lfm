# Qwen Latent Targets: A Proposed New Direction

**Status:** Proposal, pending two sanity checks. Working document for the next session.

## Summary

Replace sentence-transformer embeddings with Qwen 0.5B hidden states as the dialogue game's perception targets. The input to the agent and the reader at evaluation time would then share a single latent space, eliminating the cross-space bridging that has blocked every prior interpretation attempt.

## The Problem Being Solved

The existing pipeline uses sentence-transformer embeddings (from a model trained on English NLI, ~384 dim) as the perception targets the agent must learn to discriminate. At inference time, we try to use a pretrained LLM (Qwen 2.5 0.5B) to interpret the agent's Neuroglot output into English. This has failed across multiple experiments:

- **UNMT** (unsupervised NMT bridging Neuroglot and Qwen's token space) — did not produce signal.
- **LLM-pressure training** (adding a Qwen-perplexity auxiliary loss to the dialogue game) — did produce Qwen-plausible surface English fragments, but downstream interpretation showed no signal at N=100.
- **Forced-choice retrieval** with Qwen 0.5B as judge — below-chance accuracy.
- **SBERT cosine** on Qwen interpretations vs sources — zero signal at N=100.
- **RLAIF with Qwen 7B as subjective judge** — blocked on 7B not fitting in 8GB VRAM.

### Root cause

Sentence-transformer embedding space and Qwen's internal representation space are fundamentally disjoint. Every approach that tries to bridge them is doing semantic-space translation through narrow lossy channels, and Qwen 0.5B at the receiving end has coarse narrative-attractor generation that collapses input-dependent signal into a handful of stock themes (war, horror film, fantasy, etc.) regardless of input.

## The Proposed Direction

Train the dialogue game using **Qwen 0.5B's own hidden states** as the perception targets, instead of sentence-transformer embeddings. Specifically:

1. Feed a corpus (e.g. the existing 300K Leipzig English sentences) through Qwen 0.5B.
2. Extract the last-token residual at the final transformer layer as the "embedding" for each sentence.
3. Cluster these Qwen-native embeddings via the same k-means pipeline currently used (approximately 2047 clusters for hard-negative sampling).
4. Train the dialogue game's discrimination task on these targets.

The agent's objective becomes: "produce a Neuroglot document such that the receiver can pick, from 16 Qwen hidden states, the one that was originally shown." This is the same discrimination task as now, just operating in a different target space.

## Why This Is Potentially a Significant Simplification

### The perception target and the reader share a single latent space

At inference time, to "interpret" a Neuroglot document, we just feed it to Qwen and extract its resulting hidden state. Compare to the target hidden state via cosine similarity. No English generation step, no SBERT, no judge, no narrative hallucination.

The whole evaluation pipeline becomes:

```
source     → Qwen  → target hidden state      (in Qwen latent space Q)
target     → agent → Neuroglot
Neuroglot  → Qwen  → produced hidden state    (in Q)
metric     = cos(target, produced)
```

The five-stage current pipeline collapses to three stages, with both endpoints in the same space.

### Training signal becomes well-defined

Rather than RLAIF with a subjective judge, we can optimize a direct geometric reconstruction objective in Qwen's latent space. The discrimination game already does this when given proper targets; we just swap which embeddings feed in.

### The agent learns to name Qwen's ontology

Qwen's internal representations encode learned linguistic/semantic structure from next-token prediction training. A Neuroglot that discriminates Qwen hidden states is, by construction, a language whose distinctions reflect Qwen's own distinctions about the world. This doubles as a novel language-based interpretability probe of Qwen: for any input to Qwen, the agent emits a linguistic label for the region of latent space that input activates.

### RLAIF machinery becomes unnecessary

No judge model, no PPO loop, no reward modeling. The existing dialogue game trainer is essentially reusable as-is with the new target space.

## Caveats and Open Questions

1. **Qwen's latent space may not have good discriminative cluster structure.** Sentence transformers are contrastively trained so similar sentences have similar embeddings; Qwen is trained for next-token prediction and its hidden states reflect that objective. The cluster structure may be weaker or differently-shaped. Sanity-check script `scripts/qwen_latent_sanity.py` (in progress) tests this directly via k-means on 1000 Leipzig sentences.

2. **Layer and pooling choice.** Last-layer last-token is a reasonable default for causal LMs but not obviously optimal. Middle layers sometimes capture more abstract structure. Pooling over all tokens vs last-token is another axis. Needs a brief sweep.

3. **Dimensionality mismatch.** Qwen 0.5B hidden dim is 896; current dialogue game is wired for 384-dim sentence-transformer outputs. Mild architectural change: widen the z-generator's input projection and related shapes. Expected effort: small code change and one retraining pass.

4. **Hidden-state input coupling is unverified.** Even if the target space has good structure for English-input clustering, it's unknown whether feeding Neuroglot to Qwen produces hidden states that vary meaningfully per input. If Qwen collapses all unfamiliar inputs to a near-constant "I don't recognize this" attractor, the geometric reconstruction objective would be pinned to a constant target and training would fail. The sanity-check script tests this directly by feeding 20 Neuroglot samples from the baseline agent to Qwen and measuring the variance of resulting hidden states.

5. **Interpretation as an optional side output.** Once a Neuroglot corresponds to a known Qwen hidden state, we can ask Qwen to verbalize what that state represents ("you just read this passage; what was it about"). This is a known-lossy operation — LLMs often cannot articulate their own representations well — but it is **decoupled from training**. Training succeeds based on geometric reconstruction. Verbalization is a bonus for human inspection.

6. **LIGO generalization is still an open problem.** To apply this to non-linguistic modalities like LIGO gravitational-wave data, we need a way to project LIGO signals into Qwen's latent space. That requires a multimodal projection layer (similar to how LLaVA maps CLIP image features into LLaMA's token space), which in turn requires some form of paired data or contrastive training. The current proposal is clean for the sentence-embedding test-bed but does not automatically generalize. This trade-off is worth acknowledging up front.

7. **The "ontology collapse" concern is unchanged in direction but changed in kind.** The user has previously objected to paired training on the grounds that it collapses the agent's perception onto pre-existing English semantic categories. Using Qwen hidden states as targets IS committing to Qwen's English-derived latent geometry. But the alternative currently in use — sentence-transformer embeddings — is also English-derived. Neither approach escapes the English substrate. The meaningful difference is not more or less English-imposing; it is whether the input and the reader share coordinates. This makes the proposal the honest choice for the test-bed given the constraint that we are using an English pretrained LLM as the reader.

## Two Cheap Sanity Checks Before Committing

Both implemented in `scripts/qwen_latent_sanity.py`.

### Check 1 — Cluster structure

Encode 1000 Leipzig sentences through Qwen 0.5B, extract last-layer last-token hidden states, run k-means with k=20, inspect whether cluster members are semantically coherent to human eye. Also report pairwise cosine statistics (mean, std, min, max) to diagnose whether the space is collapsed.

### Check 2 — Input coupling

Using the baseline 98.3% dialogue game checkpoint, generate Neuroglot documents for 20 diverse Leipzig sources. Feed each Neuroglot back into Qwen, extract its resulting hidden state. Report pairwise cosine statistics for these Neuroglot-induced hidden states versus the same statistics for the natural English source sentences fed directly into Qwen.

If the Neuroglot-induced states have much smaller variance than the natural-English states, Qwen is collapsing them all to a single "unknown input" attractor and the approach will fail. Also measure `cos(Neuroglot_i, source_i)` versus `cos(Neuroglot_i, source_j)` for `j ≠ i` as a direct geometric-retrieval signal at the Qwen latent level.

## Decision Criterion

Commit to a full dialogue-game retraining with Qwen-native targets if **all three** hold:

- Check 1 shows coherent clusters.
- Check 2 shows reasonable variance in Neuroglot-induced hidden states (Neuroglot std within ~50% of source std).
- The Neuroglot→source geometric retrieval shows any above-chance signal.

If either check fails hard, the approach is dead and we pivot elsewhere — most likely toward retraining the phrase decoder on Qwen-BPE text or accepting that evaluation requires an external API judge.

## Implementation Plan If Sanity Checks Pass

### Phase 1: one-time offline extraction

- Feed all 300K Leipzig sentences through Qwen 0.5B (last-layer last-token).
- Save as a new EmbeddingStore at `data/embeddings_qwen/`.
- Re-cluster to produce `cluster_index.json`, `cluster_labels.npy`.
- Takes ~30–60 minutes on the local GPU; one-off.

### Phase 2: dialogue game configuration changes

- New config `configs/dialogue_v7_qwen_targets.yaml`.
- Point `embedding_store_dir` at `data/embeddings_qwen/`.
- Update any hardcoded `dim=384` assumptions to `hidden_dim=896` (or whatever Qwen's layer width actually is).
- Expected changes: z-generator input projection, context transformer input dim, a few configuration constants.

### Phase 3: training

- Train from scratch on the Qwen-target dataset using the known-good `dialogue_v7_phase1.yaml` hyperparameters as a starting point.
- Monitor discrimination accuracy; expect similar dynamics to the baseline run.
- Target: >85% accuracy at 100% hard negatives, matching or approaching the 98.3% from the current baseline.
- Training time: similar to the baseline (~1–3 hours local).

### Phase 4: evaluation

- Generate Neuroglot documents from the trained agent on held-out source sentences.
- Feed those documents back into Qwen; extract resulting hidden states.
- Compare against ground-truth target hidden states via cosine similarity.
- Report mean cosine gap (correct vs random) and retrieval recall@K at various K.

### Phase 5 (optional): human-readable interpretation

- For selected Neuroglot samples, prompt Qwen with "you just read: [Neuroglot]; describe what kind of text this reminds you of" and collect the generated descriptions.
- Manual inspection: are descriptions consistent across similar Neuroglot, different across dissimilar?

## Risks and Alternative Interpretations If This Fails

If the sanity checks pass but training fails to achieve good discrimination, possible causes:

- Qwen's hidden states cluster too tightly, making fine discrimination hard.
- Dimensionality mismatch handled poorly.
- The frozen phrase decoder's latent z space isn't expressive enough to distinguish many Qwen regions.

If the sanity checks pass AND training succeeds but geometric retrieval shows no signal in evaluation, that would be a strong surprise meaning Qwen's encoding of Neuroglot doesn't recover the training target even though the receiver-side network (trained on Qwen targets) can match them. Debugging path:

- Check whether Qwen's encoding of Neuroglot is different when the model is in train vs eval mode.
- Check whether dropout/attention quirks are affecting reproducibility.
- Consider whether a separate "Neuroglot-to-Qwen" encoder head is needed vs relying on Qwen's default reading.

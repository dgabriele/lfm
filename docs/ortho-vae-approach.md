# Orthographic VAE (v12) — Alternative to the IPA Path

**Status:** active development. This document is a living scratchpad; update
as v12 training, debugging, and evaluation progress.

## TL;DR

The original LFM stack (v7) has the frozen VAE decoder emit IPA, which is then
run through a learned phoneme-to-grapheme (p2g) model to produce
English-approximate spelling before an LLM reads it.

**v12** skips the IPA layer entirely: the decoder emits English BPE subwords
directly, so an LLM (Qwen) can re-tokenize and read the output zero-shot with
no p2g layer in between.

This is a **different topology** for the emergent language, not an
optimization of the old one. It trades one theoretical property for another.

## The two topologies

| | **v7 — IPA decoder + p2g + LLM** | **v12 — orthographic BPE decoder + LLM** |
|---|---|---|
| Surface vocab | ~50 IPA symbols + syllable markers | 8K English BPE pieces |
| Output surface | phonological pseudo-speech | English subwords concatenated as text |
| **Topology is inherent to** | the surface (phonological feature space) | **not the surface** — to the LLM's pretrained English semantic space |
| Closeness in surface space means | phonetic similarity (voiced↔unvoiced, close vowels) | nothing (BPE ids are arbitrary integers) |
| Closeness in interpretation space means | whatever Qwen learned to map from pseudo-speech — learned from scratch on the LFM corpus | whatever English words mean to Qwen — immediate, free, pretrained |
| Lossy stages before interpretation | 3 (z → IPA → spelling → interpretation) | 2 (z → BPE → interpretation) |
| p2g required | yes (~65% word accuracy, v11) | no |
| Preserves alien-ontology framing | yes (Qwen has no prior on IPA) | no (Qwen reads English as English) |
| Supports fine-grained phonological interpolation in surface space | yes | no |
| Supports direct semantic-manifold routing via Qwen embeddings | no (requires learning the mapping) | yes (Qwen's embeddings are the coordinate system) |

### Why v12 is more operationally direct for LFM's goal

The LFM mission is to let any neural system express its internal structure in
a form a human can interpret via an LLM. The practical criterion (established
earlier during discussion of the Qwen-latent topology eval) is **topology
preservation through to the LLM's English output** — not preservation of the
source system's literal referents.

Under that criterion:

- v7 has to learn a mapping from z to a topology-preserving pseudo-speech
  *and* then Qwen has to learn (or be fine-tuned) to interpret that
  pseudo-speech in a topology-preserving way. The v7 Qwen-latent experiment
  measured this end-to-end and found ρ ≈ 0.09 — topology is the weak link.
- v12 inherits Qwen's pretrained English semantic topology for free. The
  decoder's job is reduced to *routing source-z neighborhoods to English
  word neighborhoods*, which Qwen then reads natively.

The "alien ontology" property v7 preserves is philosophically interesting
but doesn't pay off on the topology metric we actually evaluate against.

### Caveat: BPE's topology is borrowed, not inherent

The v12 vocabulary of 8K BPE ids is *not* intrinsically organized — the ids
themselves carry no similarity structure. The topology only emerges once
Qwen (or any comparable English-tokenizer-aware model) maps each piece to
its pretrained embedding.

Practical consequence: **surface edit distance on BPE pieces is meaningless
for v12**. Any topology-correlation diagnostic for v12 must measure "surface
closeness" as the cosine distance between learned piece-embedding means, not
as edit distance.

## Dataset + tokenizer

- Source corpus: `data/datasets/english-diverse-constituents/constituents.txt`
  — 11.6M English constituents from 750K Wikipedia paragraphs, cleaned of
  HTML, citation brackets, curly quotes, `@-@` wikitext artefacts,
  non-Latin script lines, and parser "unknown" placeholder lines.
- Tokenizer: SentencePiece BPE, **8000 vocab**, trained on a random
  500K-line subset of the cleaned corpus.
- Build script: `scripts/build_v12_english_ortho_dataset.py`.
- Cleaning script (also backported into the constituent builder):
  `scripts/clean_english_constituents.py`.
- Output dataset: `data/datasets/english-constituents-v12/`
  (`samples.h5` 4.4GB + `spm.model` / `.vocab`).

## Training config

`configs/pretrain_vae_v12_english_ortho.yaml`.

- Architecture: v7_english clone — latent_dim 256, decoder 512 × 4 layers,
  8 heads, RoPE, weight-shared layers, multi-scale attention windows,
  `num_memory_tokens=8`.
- Batch: `batch_size: 240 × grad_accum: 2` (effective 480). Reduced from
  the original 320/1 after a GPU OOM on 24GB 3090 with length-boosted
  batches.
- Regularization carried from v7_english:
  `unlikelihood_weight: 0.15`, `scheduled_sampling_target: 0.15`,
  `length_boost_threshold: 40`, `length_boost_factor: 15`.
- 3 epochs over ~9M (post-drop) tokenized sequences,
  `max_seq_len` auto-scales to 286.

## Known debug / memory issues (fixed)

The v7 pretrain pipeline's `MultilingualCorpusDataset` pre-padded every
sample to `max_seq_len` as `int64` in `__init__`. On 11.6M samples this
allocated ~23GB of padded tensors before training even started and
triggered an OOM on the 32GB-RAM vast instance. The docstring already
claimed "padding is deferred to the collate function" — the implementation
just didn't. Fix:

- Store variable-length `int32` tensors (no upfront padding).
- Add `pad_collate` / `pad_collate_indexed` in `lfm.data.corpus` that pad
  to max-in-batch inside the DataLoader.

Memory goes from ~23GB to ~1.5GB for the padded-tensor representation.

## Evaluation — what to measure

Because v12 surface topology is borrowed from Qwen, the topology metric
must be:

1. Sample N source embeddings with known input topology.
2. Decode each to BPE surface text.
3. Run each surface through Qwen, take the final-layer hidden state
   (or SBERT of Qwen's English interpretation).
4. Correlate pairwise interpretation-space distances with pairwise
   input-space distances.

A result significantly better than v7's ρ ≈ 0.09 would be the primary
win condition for v12.

## Open questions

- **Mode collapse risk**: BPE argmax can funnel many source inputs to the
  same common English words ("the", "of"). Need to watch the surface
  uniqueness diagnostic and dialogue-game accuracy to catch this.
- **Qwen prior as asset vs obstacle**: if the source system's structure
  genuinely doesn't fit Qwen's English geometry, v12 will smear source
  distinctions that v7 could have preserved in phonological space. No
  clean way to tell without a head-to-head on the topology metric.
- **Fine-tuning budget**: if v12 topology is already good zero-shot, we
  may not need to fine-tune Qwen at all. If not, fine-tuning onto the
  decoder's output should be cheap since the surface is already English.

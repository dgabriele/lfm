# Expression Game — Progress Log

Tracking experiments, results, and design decisions for the expression game system. Goal: produce IPA expressions from input embeddings that are (1) surface-diverse enough for LLM translation training and (2) discriminative enough to carry semantic information.

---

## Key Metrics

| Metric | What it measures | Target |
|--------|-----------------|--------|
| **acc** | Receiver discrimination accuracy (16-way, 100% hard negatives) | >95% |
| **sdiv** | Per-batch unique IPA token sequences / batch size | 100% |
| **gdiv** | Running global unique / total seen | ~100% |
| **Pairs unique** | Unique IPA across full 10K embedding store (eval) | >9,500/10K |
| **segs** | Mean phrases per expression | Variable (content-dependent) |
| **expr_len** | Mean total tokens per expression | Longer = better for LLM |

---

## Experiments (chronological)

### 1. GRU + PonderNet (baseline)
- **Config**: GRU z-sequence, PonderNet cumulative halt, hidden-state primary
- **Result**: 97.3% acc, segs=2.5, expr_len=22
- **Surface diversity**: 1,519/10,000 unique (15.2%)
- **Why it failed**: `_input_proj` collapsed to 5.8% of z-space. Hidden-state path discriminated via cross-attention nuances without needing diverse tokens. PonderNet cumulative product prevented phrase expansion.

### 2. GRU + ZDistributionLoss
- **Config**: Added moment-matching loss to keep z vectors spread across decoder's training distribution (zcov target=1.0)
- **Result**: 74% acc (plateau), zcov=1.0, segs=2.0
- **Surface diversity**: 1,519/10,000 (no improvement)
- **Why it failed**: zcov=1.0 spread z vectors but `_input_proj` mapped many different embeddings to z regions that produce similar tokens. Distribution matching necessary but not sufficient.

### 3. GRU + ZDiversityLoss
- **Config**: Hinge loss on intra-expression z cosine similarity
- **Result**: 97.7% acc, z_sim=0.20 (down from 0.97)
- **Surface diversity**: Still ~1,519/10,000
- **Why**: z vectors diverse within expression but all expressions still in same z region

### 4. Surface bottleneck (dual-path)
- **Config**: Straight-through token re-embedding as primary loss, hidden-state annealed
- **Result**: Plateaued at 25-32% with leaf decoder, 75% with v7 decoder
- **Why limited**: Leaf decoder's 18 tokens insufficient for surface discrimination. v7's 50+ tokens worked but was still climbing when killed.

### 5. Independent phrase gates (replaced PonderNet)
- **Config**: Per-position sigmoid gate (not cumulative product), biased closed
- **Result**: Segs still stuck at 1.7-2.5 regardless of initialization
- **Why**: No mechanism gets clear gradient signal to expand phrases. Gate gradient through per_pos_weight scaling is too indirect.

### 6. DiffusionZGenerator (breakthrough)
- **Config**: T=4 flow-matching denoiser, all K positions refined simultaneously
- **Result with leaf decoder**: 93.8% acc (phase 1, hidden-state), **9,994/10,000 unique** pairs
- **Result with v7 decoder**: 95% acc (phase 1, hidden-state), longer expressions
- **Why it works**: Noise schedule forces full z-space exploration by construction. No `_input_proj` to collapse. All phrases co-adapt via self-attention.

### 7. Phase 2: surface primary (leaf decoder)
- **Config**: Resume from phase 1 checkpoint, switch to surface-only loss
- **Result**: 5% → 89% in 200 steps, segs still 2.5, expr_len=24
- **Why phrases don't grow**: Surface loss at 89% is already low (~0.06). Marginal gain from extra phrase doesn't overcome gate weights learned in phase 1.

---

## Architectural Findings

### Hidden-state path is too powerful
The decoder's cross-attention hidden states carry so much z information that the receiver can discriminate at >95% accuracy with only 18 tokens. This makes surface diversity and phrase expansion unnecessary from the game's perspective, even though both are critical for downstream LLM translation.

### Surface diversity ≠ hidden-state accuracy
A model can have 97% hidden-state accuracy but only 15% surface diversity. The metrics are orthogonal. Surface diversity must be measured directly (sdiv, gdiv, full 10K pair eval).

### z-space coverage is necessary but not sufficient
ZDistributionLoss (zcov=1.0) ensures z vectors span the decoder's training distribution. But the decoder may map different z regions to similar tokens (many-to-one). The diffusion generator solves this because the noise schedule naturally explores regions the decoder maps distinctly.

### Phrase count is inelastic
Neither PonderNet, independent gates, nor zero-bias initialization successfully increased phrase count above 2.5 in response to accuracy pressure. The gradient signal from "I need more tokens" is always too weak to flip a gate. Phrase count is effectively a hyperparameter, not a learned quantity.

---

## Current Design Problem

**The receiver compares a message vector against raw embeddings via dot product.** The IPA tokens don't need to be discriminative — the learned message encoder projection does all the work. This misaligns the training objective with the downstream use case (LLM sees only IPA tokens).

**Proposed fix**: IPA-to-IPA receiver. Encode all 16 candidates through the frozen decoder, compare IPA representations pairwise. The game directly mirrors the LLM's task: distinguish inputs based on their IPA surface forms.

---

## Pretrained Decoders Available

| Model | Dataset | Val CE | max_seq_len | Surface div (random z) |
|-------|---------|--------|-------------|----------------------|
| v5-leaf-27 | 4M leaf phrases | 0.0071 | 27 | 100% (64/64) |
| v7 | 11.6M full constituency | 0.0082 | 109 | 100% (2000/2000) |

---

### 8. IPA-to-IPA receiver (current)
- **Config**: Diffusion + leaf decoder, IPAEncoder (shared, warm-started from decoder embeddings), precomputed IPA cache for candidates, surface-only, hard CE loss
- **Result at step 2100**: 81.2% IPA-to-IPA accuracy (still climbing), sdiv=100%, gdiv=100%, segs=2.5, expr_len=26
- **Key insight**: IPA encoder learns to read token-level representations fast (5% → 81% in 500 steps) when denoiser is pretrained from phase 1
- **Limitation**: Hard cross-entropy treats discrimination as binary. Doesn't encode degrees of semantic similarity — "the dog ran" should be closer to "the cat ran" than to "quantum mechanics."

---

## Next Steps

1. **Soft topology loss** — replace hard CE with KL divergence against input embedding cosine similarities. IPA similarity structure mirrors semantic structure.
2. **Run with v7 decoder** — longer expressions, more LLM-learnable structure
3. **Evaluate with full pair generation** — 10K pairs, count unique, measure Zipf statistics
4. **Train LLM translator** — self-supervised next-token prediction on IPA corpus

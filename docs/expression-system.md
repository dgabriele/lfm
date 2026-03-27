# Expression System

Learnable tree-structured expression generation through the LFM linguistic bottleneck.

---

**Contents**

1. [Motivation](#motivation)
2. [Core Insight](#core-insight)
3. [Architecture](#architecture)
   - [ExpressionGenerator](#expressiongenerator)
   - [Expression](#expression)
   - [ExpressionEncoder](#expressionencoder)
4. [Continuous Z-Switching Decode](#continuous-z-switching-decode)
5. [Integration](#integration)
6. [Downstream Applications](#downstream-applications)
7. [Design Decisions](#design-decisions)

---

## Motivation

A single latent vector z decoded through the frozen LFM decoder produces a
single utterance — a flat sequence of IPA tokens. This is sufficient for
simple referential communication (agent game accuracy ~95%), but it has
fundamental limitations:

- **Fixed granularity**: One z = one utterance. No way to express compositional
  structure (e.g., "the red dog" as a sub-expression within a larger statement).
- **No variable length control**: The v1 decoder always produces ~96 tokens
  because it was trained exclusively on full sentences. It never learned EOS
  at short positions.
- **No hierarchical composition**: Human language is recursively structured.
  Noun phrases embed in verb phrases embed in clauses. A flat z → tokens
  pipeline cannot represent this.

The expression system addresses all three by introducing a **learned tree
structure** between the agent's embedding and the decoder's output.

## Core Insight

Algebraic expressions are trees. Linguistic expressions are trees. The
expression system makes this connection concrete: an agent's message is a
binary constituency tree where the topology is learned and each leaf carries
a latent z vector decoded through the frozen LFM decoder. The tree structure
is not imposed — it emerges from communication pressure via REINFORCE,
because compositional decomposition is the most efficient way to communicate
through the linguistic bottleneck.

The key innovation is **continuous z-switching**: instead of decoding each
leaf independently (producing disjoint IPA fragments), one continuous
autoregressive pass runs through all leaves in tree order, switching the
cross-attention memory at segment boundaries while the KV cache persists.
This produces phonotactically coherent output — natural coarticulation and
prosodic bridging across constituents — because the decoder's language model
state carries context from one segment to the next.

```
    Tree:        ○ (root — learned topology)
                / \
               ○   z₃
              / \
            z₁   z₂

    Decode:  [BOS ðʌ kwɪk braʊn | fɑks dʒʌmpt | oʊvɝ ðʌ leɪzi dɑɡ EOS]
                  memory=z₁       memory=z₂     memory=z₃
                  (continuous KV cache — no breaks)
```

## Architecture

The expression system has three components:

### ExpressionGenerator

`lfm.expression.ExpressionGenerator` — learns tree topology and decodes
through the frozen LFM decoder.

**Topology generation** (top-down):
1. Root context: MLP projects the agent's input embedding into a hidden state.
2. At each node, a learned `expand_head` decides: expand (create two children)
   or stop (become a leaf). Below `min_depth`, expansion is forced.
3. Leaf nodes project their hidden context through `leaf_proj` to produce
   (μ, σ) → sample z via reparameterization.

**Continuous decode** (left-to-right):
4. Leaves are collected in in-order traversal (left-to-right tree ordering).
5. One autoregressive pass runs through all leaves, switching the cross-attention
   memory vector at segment transitions (EOS or max tokens per leaf).
6. The KV cache carries continuously across z-switch boundaries.

The expand/leaf decisions are discrete (sampled from Bernoulli) and trained
via REINFORCE. The leaf z vectors are continuous and trained via the
reparameterization trick through the receiver's gradient.

### Expression

`lfm.expression.Expression` — the data structure holding everything:

| Field | Shape | Description |
|-------|-------|-------------|
| `is_leaf` | (B, N) bool | Which nodes are leaves |
| `active` | (B, N) bool | Which nodes are active |
| `depth` | (N,) int | Precomputed per-node depth |
| `leaf_z` | (B, N, D) | Leaf latent vectors |
| `leaf_mu` | (B, N, D) | Leaf means (for REINFORCE) |
| `tokens` | (B, T) long | Continuous decoded token sequence |
| `states` | (B, T, H) | Decoder hidden states |
| `lengths` | (B,) int | Total valid tokens |
| `segment_boundaries` | (B, L) int | Token position of each segment start |
| `leaf_order` | (B, L) int | Node indices in left-to-right order |

Nodes are indexed in BFS order (root=0, left child of i = 2i+1, right child = 2i+2).

### ExpressionEncoder

`lfm.expression.ExpressionEncoder` — composes a decoded expression into a
fixed-size message vector for downstream use.

1. **Segment pooling**: Mean-pool decoder hidden states within each segment
   (bounded by z-switch points).
2. **Tree-guided composition**: Bottom-up Merge from leaves to root. Each
   internal node's representation is a learned function of its children's
   representations, plus a depth embedding for level-specific behavior.
3. **Shape embedding**: A deterministic hash of the tree topology is mapped
   through a learned embedding, giving the receiver a disentangled signal
   about tree structure separate from compositional content.

The Merge operation mirrors syntactic Merge in generative linguistics: the
meaning of a constituent is a function of the meanings of its sub-constituents
and how they combine.

## Continuous Z-Switching Decode

The defining feature of this architecture. Why it matters:

**Without z-switching** (independent per-leaf decode): Each leaf is decoded
in isolation. The decoder starts fresh for each segment. The output is
disjoint IPA fragments stitched together — phonotactically incoherent at
boundaries, with no prosodic continuity.

**With z-switching** (this system): One continuous AR pass. The KV cache
accumulates context from all previous segments. When the memory vector switches
from z₁ to z₂, the decoder "knows" what came before because the self-attention
history is intact. This enables:

- **Coarticulation**: Final phonemes of one segment blend naturally into the
  initial phonemes of the next.
- **Prosodic bridging**: Intonation and rhythm carry across constituent
  boundaries, producing utterance-level coherence.
- **Sandhi**: Phonological rules that apply across word/phrase boundaries
  (e.g., liaison in French, linking-r in English) emerge naturally.

This requires no changes to the pretrained decoder. The decoder was trained on
continuous text — it already handles long sequences with internal structure.
The z-switching mechanism simply changes what the decoder is "talking about"
mid-sequence while preserving its language model state.

## Integration

The expression system is designed for plug-and-play integration with any
agent architecture that produces fixed-size embeddings.

```python
from lfm.expression import ExpressionGenerator, ExpressionEncoder
from lfm.faculty import LanguageFaculty

# Load pretrained LFM decoder
faculty = LanguageFaculty.from_pretrained("data/models/v4/")

# Create expression system
expr_gen = ExpressionGenerator(
    generator=faculty.generator,
    input_dim=384,           # your agent's embedding dim
    latent_dim=384,          # LFM latent dim
    hidden_dim=512,          # internal hidden dim
    max_depth=3,             # max tree depth
    min_depth=1,             # forced expansion depth
    max_tokens_per_leaf=96,  # max tokens per segment
)

expr_enc = ExpressionEncoder(
    hidden_dim=512,
    output_dim=384,
    max_depth=3,
)

# In your agent's forward pass:
expression = expr_gen(agent_embedding)   # topology + continuous decode
message = expr_enc(expression)           # fixed-size message vector

# Use message for whatever downstream task:
# - Referential game (score against candidates)
# - Reconstruction (decode message back to prediction)
# - Translation (feed to LLM for interpretation)
```

**What's frozen vs. learned:**
- Frozen: The LFM decoder (all parameters in `faculty.generator`)
- Learned: `ExpressionGenerator` (root_proj, expand_head, leaf_proj) and
  `ExpressionEncoder` (segment_enc, merge, depth_embed, shape_embed)

**Training**: The tree topology decisions (expand/leaf) are discrete and
trained via REINFORCE. The leaf z vectors and encoder are continuous and
trained via standard backpropagation through the receiver/task loss.

## Downstream Applications

The expression system is a general-purpose **structured communication module**.
Any setting where an agent needs to express complex internal state through a
compositional, variable-length, linguistically structured channel:

**Referential games**: Discriminate between candidate inputs based on the
expression. Tree structure naturally decomposes when the input has
compositional structure (e.g., scene descriptions, multi-attribute objects).

**Scientific communication**: Agents observing complex systems (protein
structures, particle events, gene regulatory states) express observations
through the linguistic bottleneck. The tree structure may recapitulate
domain-specific decompositions (functional groups, decay chains, pathway
modules).

**Mathematical expression**: Algebraic expressions are literally trees. An
agent communicating mathematical content through the expression system would
develop a morphologically regular language where the tree topology mirrors
operator precedence and the leaf "words" denote operands.

**Multi-agent negotiation**: Agents with different observations of shared
state must reach agreement. The compositional structure allows
information-theoretically efficient communication — each constituent can
encode a different aspect of the state.

## Design Decisions

**Binary trees only**: The system uses strictly binary trees (each internal
node has exactly two children). This is a deliberate simplification that
matches linguistic constituency structure — in X-bar theory, all syntactic
Merge is binary. Non-binary branching can be represented by right-branching
chains.

**BFS node indexing**: Nodes are indexed in breadth-first order. This makes
the tree layout static (precomputed once) and enables batched operations
across the tree without variable-length indexing.

**In-order traversal for decoding**: Leaves are decoded left-to-right via
in-order traversal of the binary tree. This gives a natural left-to-right
ordering that mirrors how constituency trees linearize into surface strings.

**Shape embedding**: The tree topology is explicitly encoded as a separate
signal via hashed shape IDs. This disentangles what the tree says (leaf
content) from how it's structured (topology), allowing the receiver to use
both signals independently. Without this, the receiver must infer structure
from compositional content alone.

**No pretraining changes**: The expression system works with any frozen LFM
decoder checkpoint. The continuous z-switching mechanism exploits properties
the decoder already has (long-range context via KV cache, phonotactic
coherence from language model training). No decoder retraining is needed.

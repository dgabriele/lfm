"""Communication architectures for agent interaction.

Provides modular communication structures that compose with the frozen
decoder to produce linguistically structured messages.

**Leading approach: continuous z-switching tree decode**
(``continuous_tree.ContinuousTreeDecoder``)

The agent produces a constituency tree where only leaf nodes carry z
vectors.  Instead of decoding each leaf independently, one continuous
autoregressive decode runs with the cross-attention memory switching
between leaf z vectors at transition points.  The KV cache carries
across boundaries, producing natural coarticulation and prosodic
coherence.  See ``continuous_tree.py`` for details.

The per-leaf independent decode (``tree.TreeSender``) is retained
for comparison but is not the recommended path.
"""

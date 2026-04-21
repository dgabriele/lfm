"""DepTree Diffusion VAE — tree-structured diffusion decoder for dependency-parsed IPA.

Extends the DepTreeVAE with a non-autoregressive diffusion decoder that
generates content tokens conditioned on the dependency skeleton. The noise
schedule follows tree depth: root/high-level structure is denoised first,
leaf-level tokens last. This eliminates autoregressive repetition loops
while preserving linguistic coherence through hierarchical generation.
"""

from lfm.generator.dep_tree_diffusion.config import DepTreeDiffusionConfig
from lfm.generator.dep_tree_diffusion.model import DepTreeDiffusionVAE

__all__ = ["DepTreeDiffusionConfig", "DepTreeDiffusionVAE"]

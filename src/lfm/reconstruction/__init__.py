"""Reconstruction-based expression training.

Trains a z-generator to produce linguistically structured IPA
expressions that encode input embeddings recoverably.  An inverse
decoder learns to reconstruct the original embedding from the
surface-level IPA token representations — the same tokens an LLM
would see during pretraining.

Unlike the agent game approach (contrastive discrimination), this
optimizes for direct information preservation: the IPA output must
carry enough information for faithful embedding reconstruction.
"""

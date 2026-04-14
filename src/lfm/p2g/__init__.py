"""p2g (phoneme-to-grapheme) VAE — IPA → English orthography.

Word-level non-autoregressive VAE.  Encoder maps IPA character
sequence to a Gaussian latent z; decoder projects z to a length and
parallel per-position character logits for the English spelling.  KL
smooths the latent manifold so pseudoword IPAs decode to nearby real
words (the orthography-not-orthagrufi property).
"""

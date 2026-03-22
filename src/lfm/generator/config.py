"""Configuration for generator modules.

Defines the ``GeneratorConfig`` used to parameterize all generator
implementations (e.g. multilingual VAE decoder).
"""

from __future__ import annotations

from lfm.config.base import LFMBaseConfig


class GeneratorConfig(LFMBaseConfig):
    """Configuration for a generator module.

    These defaults target 6GB single-GPU training with a conservative decoder
    that intentionally relies on the latent code z.  For richer linguistic
    priors on larger GPUs, use ``decoder_hidden_dim=512``,
    ``decoder_num_layers=4``, ``decoder_num_heads=8``, ``latent_dim=256``.

    Attributes:
        name: Registry name of the generator implementation to use.
        latent_dim: Dimensionality of the VAE latent space.
        vocab_size: Subword vocabulary size from sentencepiece (excludes
            BOS/EOS special tokens).
        max_output_len: Maximum decoded sequence length.
        decoder_hidden_dim: Hidden size of the autoregressive decoder.
        decoder_num_layers: Number of transformer decoder layers.
        decoder_num_heads: Number of attention heads in the decoder.
        decoder_dropout: Dropout rate in decoder layers.
        kl_weight: Weight for KL divergence loss in ``extra_losses()``.
        kl_free_bits: Per-dimension free bits floor to prevent posterior
            collapse.  ``0.0`` disables.
        temperature: Gumbel-Softmax temperature for decoder sampling.
        temperature_min: Minimum temperature after annealing.
        temperature_anneal_steps: Steps to anneal temperature linearly.
        hard_sample: Whether to use straight-through Gumbel-Softmax
            (hard one-hot forward, soft gradient backward).
        pooling: How to aggregate variable-length input embeddings before
            projecting to latent space.  One of ``"mean"`` or ``"attention"``.
        pretrained_decoder_path: Path to a pretrained VAE decoder checkpoint.
            ``None`` means random initialization (valid during pretraining).
        spm_model_path: Path to a trained sentencepiece ``.model`` file.
            ``None`` disables text decode (``decode_to_text`` will raise).
        freeze_decoder: Whether to freeze decoder weights during agent
            training, preserving the pretrained linguistic prior.
    """

    name: str = "multilingual_vae"
    latent_dim: int = 128
    vocab_size: int = 8000
    max_output_len: int = 64
    decoder_hidden_dim: int = 256
    decoder_num_layers: int = 2
    decoder_num_heads: int = 4
    decoder_dropout: float = 0.2
    kl_weight: float = 0.1
    kl_free_bits: float = 2.0
    temperature: float = 1.0
    temperature_min: float = 0.5
    temperature_anneal_steps: int = 10000
    hard_sample: bool = True
    pooling: str = "mean"
    pretrained_decoder_path: str | None = None
    spm_model_path: str | None = None
    freeze_decoder: bool = True

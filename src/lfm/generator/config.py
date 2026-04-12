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
    latent_dim: int = 256
    vocab_size: int = 8000
    max_output_len: int = 96
    num_statements: int = 1
    """Number of independent statements per input.  Each gets its own z
    and decode.  Default 1 preserves backward compatibility."""
    decoder_hidden_dim: int = 512
    decoder_num_layers: int = 4
    decoder_num_heads: int = 8
    decoder_dropout: float = 0.2
    kl_weight: float = 0.1
    kl_free_bits: float = 0.5
    temperature: float = 1.0
    temperature_min: float = 0.5
    temperature_anneal_steps: int = 10000
    hard_sample: bool = True
    pooling: str = "mean"
    # Linguistic attention structure
    attention_head_windows: tuple[int, ...] = (3, 3, 7, 7, 15, 15, 0, 0)
    """Per-head sliding window sizes for multi-scale attention.  ``0`` means
    full causal.  Length must equal ``decoder_num_heads``.  Set all to ``0``
    to disable and use standard full causal attention."""
    attention_global_every: int = 7
    """Spacing of global attention positions (composition points)."""
    use_rope: bool = True
    """Use Rotary Positional Embeddings instead of learned absolute positions.
    Enables translation-invariant pattern learning."""
    share_decoder_layers: bool = True
    """Use N/2 unique layers each applied twice (literal recursion)."""
    num_memory_tokens: int = 1
    """Number of memory tokens for z → decoder cross-attention.
    1 = single memory vector (original behavior). K > 1 projects z into K
    memory tokens, giving the decoder richer z access across all positions."""

    pretrained_decoder_path: str | None = None
    spm_model_path: str | None = None
    freeze_decoder: bool = True

    # Phoneme-VAE surface formatting.  Applies only to PhonemeVAEGenerator.
    # Phonemes within a word are joined by this string; words are always
    # separated by spaces.
    #
    # Default "-".  Empirical rationale (see scripts/diagnose_separator_mode.py):
    #
    #   * "|" gives perfect phoneme-token alignment but triggers Qwen's
    #     code/table-mode attention (next-token mass shifts to pipe, 40%
    #     code-token mass), defeating the architectural-leverage argument
    #     for a language-shaped alphabet.
    #   * "-" preserves language-mode processing (next-token mass stays in
    #     the prose regime, 0% code-token mass) at the cost of ~24% of
    #     phonemes fragmenting (those where Qwen has a "-X" merge).
    #     Fragmentation is still deterministic per phoneme, so Qwen can
    #     learn the mapping during FT.
    #
    # The language-mode cost is strictly worse than the fragmentation cost:
    # deterministic fragmentation can be learned; mode shift breaks the
    # whole premise of using Qwen's language pathways for interpretation.
    phoneme_word_boundary: str = "-"
    phoneme_word_size: int = 3
    """How many phonemes per Neuroglot 'word' in the decoded surface form."""

    # Post-hoc VQ codebook for hybrid discrete+continuous agent interface.
    # Fitted to the pretrained encoder's z distribution via fit_vq_codebook.py.
    # When set, the agent's z is decomposed into a discrete VQ code (anchor)
    # plus a continuous residual (fine detail), both in-distribution for the
    # decoder.  Controls the blend via vq_residual_alpha.
    vq_codebook_path: str | None = None
    vq_residual_alpha: float = 1.0  # 0=pure VQ, 1=full residual passthrough

    # Vector Quantization (VQ-VAE mode)
    use_vq: bool = False
    vq_mode: str = "grouped"  # "residual" or "grouped"
    vq_num_levels: int = 4    # for residual mode
    vq_num_groups: int = 8    # for grouped mode
    vq_codebook_size: int = 512
    vq_commitment_weight: float = 0.25
    vq_entropy_weight: float = 0.1
    vq_balance_weight: float = 0.1
    vq_orthogonality_weight: float = 0.01
    vq_ema_update: bool = True
    vq_decay: float = 0.99

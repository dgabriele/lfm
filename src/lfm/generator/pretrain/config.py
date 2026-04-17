"""VAE pretraining configuration and constants."""

from __future__ import annotations

from typing import Any

from pydantic import model_validator

from lfm.config.base import LFMBaseConfig

# Fields that USED to exist on VAEPretrainConfig but have been
# removed.  Old YAMLs / saved checkpoint configs still reference
# them; we strip silently so legacy artifacts keep loading.
_DEPRECATED_FIELDS: frozenset[str] = frozenset({
    "use_adversarial",
    "adv_weight",
    "adv_lr",
    "adv_disc_hidden",
    "adv_disc_embed_dim",
    "adv_warmup_steps",
    "adv_free_run_len",
    "adv_spectral_norm",
})

# IPA vowels — used to strip trailing orphan consonants from decoded text
_IPA_VOWELS = set(
    "iyɨʉɯuɪʏʊeøɘɵɤoəɛœɜɞʌɔæɐaɶɑɒ"
    "ãẽĩõũɛ̃ɔ̃"  # nasalized vowels
)


class VAEPretrainConfig(LFMBaseConfig):
    """Configuration for VAE decoder pretraining.

    These defaults are conservative for 6GB single-GPU.  For larger GPUs,
    consider ``decoder_hidden_dim=512``, ``decoder_num_layers=4``,
    ``decoder_num_heads=8``, ``latent_dim=256`` for richer linguistic priors.

    Attributes:
        corpus_loader: Registry name of the corpus loader to use.
        corpus_loader_config: Keyword arguments passed to the corpus loader
            config constructor (e.g. ``{"data_dir": "data/leipzig"}``).
        corpus_paths: Legacy fallback — paths to plain text files/directories.
            Used only when ``corpus_loader`` is ``None``.
        spm_model_path: Path to a trained sentencepiece ``.model`` file.
            If ``None``, trains a new model from the corpus data.
        spm_vocab_size: Vocabulary size for sentencepiece training.
        latent_dim: Must match target ``GeneratorConfig.latent_dim``.
        encoder_num_layers: Number of transformer encoder layers.
        decoder_hidden_dim: Must match target
            ``GeneratorConfig.decoder_hidden_dim``.
        decoder_num_layers: Must match target
            ``GeneratorConfig.decoder_num_layers``.
        decoder_num_heads: Must match target
            ``GeneratorConfig.decoder_num_heads``.
        decoder_dropout: Dropout rate during pretraining.
        max_seq_len: Maximum token sequence length (truncated).
        kl_weight: Final KL weight after warmup.
        kl_free_bits: Per-dimension free bits floor.  Set to ``2.0`` to
            prevent posterior collapse with the conservative decoder.
        kl_warmup_steps: Steps to linearly anneal KL weight from 0.
        batch_size: Pretraining batch size per device.
        gradient_accumulation_steps: Accumulate gradients over this many
            batches before stepping.  Effective batch size is
            ``batch_size * gradient_accumulation_steps``.
        use_amp: Enable mixed precision training (``torch.amp``).
            Halves activation memory and speeds up training.
        lr: Learning rate.
        num_epochs: Number of pretraining epochs.
        val_fraction: Validation split fraction.
        seed: Random seed for reproducibility.
        device: Torch device string.
        output_path: Where to save the pretrained decoder checkpoint.
        repetition_penalty: Multiplicative penalty on logits for tokens
            that appeared in the recent ``repetition_window`` positions.
            ``1.0`` disables.  Applied during both teacher-forced training
            and free-run decoding to suppress degenerate loops.
        repetition_window: Number of recent positions to check for
            repeated tokens.
        topo_weight: Weight for topological regularization loss.
            Penalizes distance mismatches between latent pairs and their
            decoded output pairs, enforcing Lipschitz continuity.
        topo_sample_pairs: Number of random pairs per batch for topo loss.
    """

    @model_validator(mode="before")
    @classmethod
    def _strip_deprecated(cls, data: Any) -> Any:
        """Silently drop removed fields so old YAMLs/checkpoints still load."""
        if isinstance(data, dict):
            return {k: v for k, v in data.items() if k not in _DEPRECATED_FIELDS}
        return data

    dataset_path: str | None = None
    corpus_loader: str | None = "leipzig"
    corpus_loader_config: dict[str, Any] = {}
    corpus_paths: list[str] = []
    spm_model_path: str | None = None
    spm_vocab_size: int = 8000
    latent_dim: int = 256
    encoder_num_layers: int = 2
    decoder_hidden_dim: int = 512
    decoder_num_layers: int = 4
    decoder_num_heads: int = 8
    decoder_dropout: float = 0.1
    num_memory_tokens: int = 1
    max_batches_per_epoch: int | None = None
    """Cap batches per epoch for smoke testing.  None = full epoch."""
    length_boost_threshold: int = 0
    """BPE token length above which samples get boosted sampling weight.
    0 = disabled (uniform random).  E.g. 15 = samples with >=15 tokens
    get ``length_boost_factor`` extra weight."""
    length_boost_factor: float = 10.0
    """Sampling weight multiplier for samples above ``length_boost_threshold``."""
    checkpoint_every_steps: int | None = None
    """Save a resumable checkpoint every N steps.  None = end of epoch only."""
    syllable_aligned_bpe: bool = False

    # Linguistic attention structure
    attention_head_windows: tuple[int, ...] = (3, 3, 7, 7, 15, 15, 0, 0)
    attention_global_every: int = 7
    use_rope: bool = True
    share_decoder_layers: bool = True
    max_seq_len: int = 96
    """Max BPE token sequence length. 0 = auto-detect from dataset
    (max sample length + 2 for BOS/EOS)."""
    encoder_pooling: str = "mean"  # "mean" or "attention"
    kl_weight: float = 0.5
    kl_free_bits: float = 0.5
    kl_warmup_steps: int = 10000
    batch_size: int = 32
    gradient_accumulation_steps: int = 2
    use_amp: bool = True
    lr: float = 1e-3
    num_epochs: int = 20
    val_fraction: float = 0.1
    seed: int = 42
    device: str = "cuda"
    output_path: str = "data/vae_decoder.pt"

    # Latent variance regularization: smooth quadratic penalty that
    # pulls per-dimension z variance toward ``z_var_target``.  Provides
    # continuous gradient signal (not just a hard floor), preventing
    # the slow variance collapse that leads to gnorm explosion.
    # Loss = weight * mean((var_per_dim - target)^2).
    z_var_weight: float = 5.0
    z_var_target: float = 0.05
    z_var_floor: float = 0.01  # legacy, unused — kept for checkpoint compat

    # Word dropout (Bowman et al. 2016): randomly zero out decoder input
    # embeddings with this probability during training.  Forces the decoder
    # to rely on z for reconstruction instead of the teacher-forced input
    # highway, preventing the encoder from driving logvar → -∞.
    # Annealed linearly from word_dropout to word_dropout_min over
    # word_dropout_anneal_epochs epochs.
    word_dropout: float = 0.3
    word_dropout_min: float = 0.05
    word_dropout_anneal_epochs: int = 20

    # DIP-VAE off-diagonal covariance penalty: encourages z dimensions
    # to be statistically independent (disentangled).  Penalizes pairwise
    # correlation between dimensions without pushing mean toward zero.
    dip_weight: float = 0.1

    # Unlikelihood regularization against reduplication (Welleck et al., 2020).
    # At each decoder position, penalizes high probability for any token
    # that appeared in the previous `unlikelihood_window` target positions
    # unless that token IS the true next target.  Directly suppresses the
    # degenerate-tail failure mode (decoder falling into `tok tok tok ...`
    # local attractors) while leaving EOS emission intact.
    unlikelihood_weight: float = 0.0   # enable at 0.1–0.2
    unlikelihood_window: int = 4

    # Cosine LR decay: minimum LR at end of training.
    lr_min: float = 1e-4

    # Legacy topological regularization (disabled — replaced by z_var)
    topo_weight: float = 0.0
    topo_sample_pairs: int = 32

    # Scheduled sampling: anneal from 0 to target probability over warmup
    # epochs, starting at ``scheduled_sampling_start_epoch``.  At each
    # decoder position, replace ground truth input with the model's own
    # prediction with this probability.  Prevents exposure bias and
    # generation degeneration at long sequence lengths.  Must start late
    # enough that the model produces reasonable predictions to feed back.
    scheduled_sampling_target: float = 0.0
    scheduled_sampling_start_epoch: int = 5
    scheduled_sampling_warmup_epochs: int = 5

    # Contrastive learning (InfoNCE): encourages similar source sentences
    # to have similar z vectors.  Requires pre-computed sentence-transformer
    # embeddings aligned by index with the training corpus.
    contrastive_weight: float = 0.0
    contrastive_temperature: float = 0.07
    embeddings_path: str = ""

    # Fixed KL for distribution smoothness (no warmup/cycling).
    # Applied without free bits — raw KL(q||N(0,1)).
    kl_beta: float = 0.0

    # Phonetic embedding initialization and label smoothing.
    phonetic_init: bool = True
    phonetic_init_scale: float = 0.5
    phonetic_label_smoothing: float = 0.1

    # Vector Quantization (VQ-VAE mode).
    # Replaces continuous Gaussian latent with discrete codebook.
    use_vq: bool = False
    vq_mode: str = "grouped"  # "residual" or "grouped"
    vq_num_levels: int = 4    # for residual mode
    vq_num_groups: int = 8    # for grouped mode
    vq_codebook_size: int = 512
    vq_commitment_weight: float = 0.25
    vq_entropy_weight: float = 0.1
    vq_balance_weight: float = 0.1
    vq_orthogonality_weight: float = 0.01
    vq_noise_sigma: float = 0.0
    vq_ema_update: bool = True
    vq_decay: float = 0.99

    # Standard-VAE z-noise injection: during training only, add gaussian
    # noise ``randn_like(z) * z_noise_sigma`` to z between the encoder
    # reparameterization and ``latent_to_decoder``.  Teaches the decoder
    # to produce valid output from z values slightly off the encoder's
    # posterior manifold — directly addresses perturbation robustness
    # without relying on KL (which collapses in this setup).  Typical
    # values: 0.03–0.10 absolute (compare against observed z_std).
    # 0 disables the mechanism entirely (v7/v12 behavior).
    z_noise_sigma: float = 0.0

    # Tag-balance auxiliary loss: encourages matched open/close tag
    # counts in the decoder's predicted distribution.  Softmax mass
    # over all open-tag IDs should equal mass over close-tag IDs per
    # sample; squared difference × this weight is added to the total
    # loss.  Off-by-default (preserves v7/v12 behavior).  Tag IDs are
    # auto-discovered at trainer startup by scanning SPM pieces for
    # `<…>` / `</…>` patterns.
    tag_balance_weight: float = 0.0

    # Constituent context training: the encoder sees the full parent
    # sentence while the decoder is supervised only on the constituent
    # span.  Requires a constituency dataset with parent_seq fields
    # (generated with --extract-constituents).
    constituent_context: bool = False
    constituent_dataset_path: str = ""   # Path to constituency HDF5
    constituent_mix_ratio: float = 0.5   # Fraction of full sentences to mix in
    constituent_max_per_language: int | None = None  # Per-language constituent cap
    constituent_balance_by_length: bool = True  # Equalize across length buckets
    constituent_length_percentile: float | None = None
    """Keep only the shortest N% of constituents per language.
    Filters out long nested phrases, keeping leaf-like constituents
    for strong short-EOS training signal.  None = no filter.
    Suggested: 50.0 (keep bottom half by length per language)."""

    # Bag-of-words auxiliary loss: penalizes missing content tokens
    # regardless of position.  Complements CE (which is position-sensitive)
    # by ensuring the right vocabulary appears in the output.
    bow_weight: float = 0.0

    # Mid-epoch diagnostic interval: run reconstruction/interpolation/
    # perturbation diagnostics every N batches within each epoch.
    # 0 = diagnostics only at epoch boundaries (default).
    diagnostic_every: int = 0

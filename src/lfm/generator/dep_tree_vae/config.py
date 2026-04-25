"""Configuration for DepTreeVAE."""

from __future__ import annotations

from pydantic import Field

from lfm.config import LFMBaseConfig


# Universal Dependencies relation labels used in training data.
# Ordered for stable vocab indexing.
DEP_RELATIONS = (
    "acl",
    "acl:relcl",
    "advcl",
    "advmod",
    "amod",
    "appos",
    "aux",
    "aux:pass",
    "case",
    "cc",
    "cc:preconj",
    "ccomp",
    "compound",
    "conj",
    "cop",
    "csubj",
    "csubj:pass",
    "dep",
    "det",
    "det:predet",
    "discourse",
    "dislocated",
    "expl",
    "fixed",
    "flat",
    "flat:foreign",
    "goeswith",
    "iobj",
    "list",
    "mark",
    "nmod",
    "nmod:npmod",
    "nmod:poss",
    "nmod:tmod",
    "nsubj",
    "nsubj:pass",
    "nummod",
    "obj",
    "obl",
    "obl:npmod",
    "obl:tmod",
    "orphan",
    "parataxis",
    "punct",
    "reparandum",
    "root",
    "vocative",
    "xcomp",
)

NUM_DEP_RELATIONS = len(DEP_RELATIONS)
DEP_REL_TO_ID = {rel: i for i, rel in enumerate(DEP_RELATIONS)}


class LatentConfig(LFMBaseConfig):
    """Latent space configuration.

    The struct/content split is no longer enforced — both downstream modules
    (SkeletonDecoder, PhraseZProjector) read the full latent. ``struct_dim``
    and ``content_dim`` remain as input-width settings for those modules and
    should normally equal ``total_dim``.
    """

    total_dim: int = 256
    struct_dim: int = 256
    content_dim: int = 256


class SkeletonDecoderConfig(LFMBaseConfig):
    """Config for the lightweight skeleton (role sequence) decoder.

    Modes:
        ``parallel``: One-shot MLP predicts all positions + length.
        ``ar``: Autoregressive transformer decoder.
    """

    mode: str = "parallel"
    hidden_dim: int = 128
    num_layers: int = 2
    num_heads: int = 4
    max_roles: int = 40
    dropout: float = 0.1


class DisentanglementConfig(LFMBaseConfig):
    """Auxiliary losses for enforcing the struct/content split."""

    struct_cls_weight: float = 1.0
    content_bow_weight: float = 0.5
    adversarial_weight: float = 0.0
    gradient_reversal_scale: float = 0.0
    hsic_weight: float = 1.0


class DepTreeVAEConfig(LFMBaseConfig):
    """Top-level configuration for DepTreeVAE."""

    # Data
    dataset_path: str = ""
    spm_model_path: str = ""
    spm_vocab_size: int = 8000
    max_seq_len: int = 80

    # Encoder
    encoder_hidden_dim: int = 512
    encoder_num_layers: int = 2
    encoder_num_heads: int = 8
    encoder_dropout: float = 0.1

    # Latent space
    latent: LatentConfig = Field(default_factory=LatentConfig)

    # Skeleton decoder (role sequence from z_struct)
    skeleton: SkeletonDecoderConfig = Field(
        default_factory=SkeletonDecoderConfig,
    )

    # Phrase decoder (reused frozen PhraseDecoder)
    decoder_hidden_dim: int = 512
    decoder_num_layers: int = 4
    decoder_num_heads: int = 8
    decoder_dropout: float = 0.1
    num_memory_tokens: int = 8
    attention_head_windows: tuple[int, ...] = (3, 3, 7, 7, 15, 15, 0, 0)
    attention_global_every: int = 7
    use_rope: bool = True
    share_decoder_layers: bool = True
    pretrained_decoder_path: str | None = None
    freeze_decoder: bool = True

    # Disentanglement
    disentanglement: DisentanglementConfig = Field(
        default_factory=DisentanglementConfig,
    )

    # KL
    kl_weight: float = 0.0
    kl_free_bits: float = 0.5
    kl_warmup_steps: int = 5000

    # DIP-VAE: off-diagonal covariance penalty on z.
    dip_weight: float = 0.0

    # Z variance regularizer: pulls per-dim z variance toward target.
    # Prevents the decoder from ignoring z by ensuring it carries
    # consistent information across dimensions.
    z_var_weight: float = 0.0
    z_var_target: float = 0.05

    # Z-prediction: forces decoder hidden states to retain z information.
    # Small MLP predicts z from pooled hidden states — prevents the decoder
    # from ignoring cross-attention memory (posterior collapse).
    z_pred_weight: float = 0.0

    # Topology — z distances preserve decoded-output distances
    topo_weight: float = 0.0

    # Interpolation smoothness — midpoints between endpoints in output space
    interp_weight: float = 0.0

    # Entropy floor — prevents vocabulary collapse at tail positions
    entropy_floor: float = 0.0
    entropy_weight: float = 0.0

    # Repetition penalty — penalizes consecutive positions with similar
    # logit distributions (cosine similarity). Directly targets AR cycling.
    rep_penalty_weight: float = 0.0

    # Completeness scorer — frozen discriminator for structural coherence
    completeness_scorer_path: str = ""
    completeness_weight: float = 0.0

    # EOS sharpening — multiplier on EOS token's contribution to next-token
    # cross-entropy. EOS is rare per sample but high-stakes; upweighting
    # produces sharper termination at the correct position. Set 1.0 to disable.
    eos_class_weight: float = 1.0

    # Length-prediction head — auxiliary head that predicts the number of
    # content tokens from z. Loss = CE between predicted and actual length.
    # When enabled, ``use_predicted_length_at_decode`` makes ``_greedy_decode``
    # use the per-sample prediction as ``expected_len`` (replacing the global
    # default), so the model controls its own EOS pressure naturally.
    length_pred_weight: float = 0.0
    use_predicted_length_at_decode: bool = False

    # Decode-time defaults (also consumed as fallbacks by ``_greedy_decode``).
    eos_boost: float = 3.0
    expected_len: int = 13
    ngram_block: tuple[int, ...] = (3, 4)

    # Word dropout: randomly zero out decoder input token embeddings.
    # Forces the decoder to rely on z (cross-attention) rather than
    # just copying from autoregressive context. Annealed from
    # word_dropout → word_dropout_min over word_dropout_anneal_epochs.
    word_dropout: float = 0.0
    word_dropout_min: float = 0.05
    word_dropout_anneal_epochs: int = 3

    # Training
    batch_size: int = 128
    gradient_accumulation_steps: int = 2
    lr: float = 5e-4
    lr_min: float = 1e-4
    num_epochs: int = 3
    use_amp: bool = True
    seed: int = 42
    device: str = "cuda"

    # Data
    max_samples: int | None = None

    # Logging / checkpoints
    log_every: int = 50
    checkpoint_every_steps: int = 5000
    val_fraction: float = 0.05
    output_dir: str = "data/models/dep_tree_vae_v1"

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
    """Latent space split configuration."""

    total_dim: int = 256
    struct_dim: int = 64
    content_dim: int = 192

    def model_post_init(self, __context: object) -> None:
        if self.struct_dim + self.content_dim != self.total_dim:
            raise ValueError(
                f"struct_dim ({self.struct_dim}) + content_dim "
                f"({self.content_dim}) must equal total_dim ({self.total_dim})"
            )


class SkeletonDecoderConfig(LFMBaseConfig):
    """Config for the lightweight skeleton (role sequence) decoder."""

    hidden_dim: int = 128
    num_layers: int = 2
    num_heads: int = 4
    max_roles: int = 40
    dropout: float = 0.1


class DisentanglementConfig(LFMBaseConfig):
    """Auxiliary losses for enforcing the struct/content split."""

    struct_cls_weight: float = 1.0
    content_bow_weight: float = 0.5
    adversarial_weight: float = 0.5
    gradient_reversal_scale: float = 1.0


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

    # Training
    batch_size: int = 128
    gradient_accumulation_steps: int = 2
    lr: float = 5e-4
    lr_min: float = 1e-4
    num_epochs: int = 3
    use_amp: bool = True
    seed: int = 42
    device: str = "cuda"

    # Logging / checkpoints
    log_every: int = 50
    checkpoint_every_steps: int = 5000
    val_fraction: float = 0.05
    output_dir: str = "data/models/dep_tree_vae_v1"

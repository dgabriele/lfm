"""VAE decoder pretraining on multilingual text data.

Pre-trains a VAE encoder-decoder on typologically diverse multilingual text
so the decoder learns the joint distribution of phonotactic, morphological,
and compositional structure across human languages.  After pretraining, only
the decoder weights are saved — the encoder is discarded because during agent
training a separate input projection maps agent embeddings to the same latent
space.

Corpus loading is modular via the registry system — the default ``"leipzig"``
loader handles the Leipzig Corpora Collection format, and additional loaders
(OPUS, UD, etc.) can be registered for future use.

Usage::

    from lfm.generator.pretrain import pretrain_vae_decoder, VAEPretrainConfig

    config = VAEPretrainConfig(
        corpus_loader="leipzig",
        corpus_loader_config={"data_dir": "data/leipzig"},
    )
    metrics = pretrain_vae_decoder(config)
"""

from .config import VAEPretrainConfig, _IPA_VOWELS
from .corpus import (
    _load_corpus_labeled,
    _load_corpus_lines,
    _sanitize_samples,
    _train_sentencepiece,
)
from .forward import (
    _dip_covariance_loss,
    _free_run_decode,
    _info_nce_loss,
    _vae_forward,
)
from .model import build_model
from .checkpoint import _file_hash, save_best_checkpoint, save_resume_checkpoint, load_resume_checkpoint
from .diagnostics import sample_decode, encode_text, word_edit_distance, structural_metrics, run_epoch_diagnostics
from .trainer import VAEPretrainer, pretrain_vae_decoder

__all__ = [
    # Config
    "VAEPretrainConfig",
    "_IPA_VOWELS",
    # Corpus
    "_load_corpus_labeled",
    "_load_corpus_lines",
    "_sanitize_samples",
    "_train_sentencepiece",
    # Forward
    "_info_nce_loss",
    "_dip_covariance_loss",
    "_vae_forward",
    "_free_run_decode",
    # Model
    "build_model",
    # Checkpoint
    "_file_hash",
    "save_best_checkpoint",
    "save_resume_checkpoint",
    "load_resume_checkpoint",
    # Diagnostics
    "sample_decode",
    "encode_text",
    "word_edit_distance",
    "structural_metrics",
    "run_epoch_diagnostics",
    # Trainer
    "VAEPretrainer",
    "pretrain_vae_decoder",
]

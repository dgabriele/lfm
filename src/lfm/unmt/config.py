"""Configuration for unsupervised NMT training.

One config spans all stages of the pipeline so that a single YAML file
describes an end-to-end experiment.  Stages read only the fields they
need.
"""

from __future__ import annotations

from lfm.config.base import LFMBaseConfig


class UNMTConfig(LFMBaseConfig):
    """End-to-end configuration for unsupervised NMT.

    Tokenizer design: Neuroglot and English use **separate** BPE models
    whose vocabularies are concatenated into a single global index
    space with shared special tokens up front.  The two languages
    never share BPE units — any character-level overlap is coincidental
    since Neuroglot is semantically disjoint from English and only
    inherits structural dynamics from the multilingual VAE decoder.
    Separate embeddings also give MUSE alignment (Stage 2) two
    independent clouds to rotate.

    Attributes:
        neuroglot_corpus: Path to the monolingual Neuroglot corpus
            (plain text, one document per line).
        english_corpus: Path to the monolingual English corpus.  Accepts
            either plain text or JSONL with a ``text`` field.
        output_dir: Root directory for all artifacts produced by this
            experiment (tokenizers, embeddings, alignment, model, logs).
        neuroglot_tokenizer_prefix: Stem for the Neuroglot sentencepiece
            model.
        english_tokenizer_prefix: Stem for the English sentencepiece
            model.
        neuroglot_vocab_size: Vocabulary size for the Neuroglot
            tokenizer (BPE units only; special tokens live in the
            shared prefix).
        english_vocab_size: Vocabulary size for the English tokenizer.
        character_coverage_neuroglot: Sentencepiece character coverage
            for Neuroglot.  The Neuroglot alphabet is small and
            artificial, so 1.0 is safe.
        character_coverage_english: Sentencepiece character coverage
            for English.  0.9995 is typical for broad Unicode text.
        max_sentence_length: Longest sentence (in characters) kept when
            training either tokenizer.
        max_tokenizer_lines: Cap on lines per corpus used for tokenizer
            training.
        neuroglot_lang_tag: Language tag prefix token for Neuroglot.
            Lives in the shared-special range of the global vocabulary.
        english_lang_tag: Language tag prefix token for English.
        max_len: Maximum tokenized sequence length for training.
        word_drop_prob: Per-token drop probability for DAE noise.
        word_swap_window: Local shuffle window for DAE noise.
        word_mask_prob: Per-token mask probability for DAE noise.
        model_dim: Transformer hidden size.
        n_layers: Encoder and decoder each get this many layers.
        n_heads: Attention heads per layer.
        ff_dim: Feed-forward inner size.
        dropout: Dropout rate across the transformer stack.
        batch_size: Training batch size in sentences per GPU step.
        lr: AdamW learning rate.
        warmup_steps: Linear warmup steps before cosine decay.
        total_steps: Total optimization steps across the whole run.
        dae_weight: Weight on the denoising autoencoder loss.
        bt_weight: Weight on the backtranslation loss.
        bt_start_step: Step at which backtranslation is introduced.
        grad_accum_steps: Gradient accumulation multiplier.
        max_grad_norm: Gradient clipping norm.
        device: Compute device.
        seed: Random seed.
    """

    # Data
    neuroglot_corpus: str = "data/translator/dialogue_corpus_v7_natural.txt"
    english_corpus: str = "data/embeddings/passages.jsonl"
    output_dir: str = "data/unmt_v1"

    # Tokenizers (Stage 1) — one per language, disjoint global vocab
    neuroglot_tokenizer_prefix: str = "spm_neuroglot"
    english_tokenizer_prefix: str = "spm_english"
    neuroglot_vocab_size: int = 16000
    english_vocab_size: int = 16000
    character_coverage_neuroglot: float = 1.0
    character_coverage_english: float = 0.9995
    max_sentence_length: int = 4096
    max_tokenizer_lines: int = 2_000_000

    # Language tags and special tokens (Stages 1+)
    neuroglot_lang_tag: str = "<ng>"
    english_lang_tag: str = "<en>"
    max_len: int = 128

    # Denoising autoencoder noise (Stages 1+)
    word_drop_prob: float = 0.1
    word_swap_window: int = 3
    word_mask_prob: float = 0.1

    # Model (Stage 3)
    model_dim: int = 512
    n_layers: int = 6
    n_heads: int = 8
    ff_dim: int = 2048
    dropout: float = 0.1

    # Training (Stage 4)
    batch_size: int = 32
    lr: float = 1e-4
    warmup_steps: int = 4000
    total_steps: int = 200_000
    dae_weight: float = 1.0
    bt_weight: float = 1.0
    bt_start_step: int = 10_000
    grad_accum_steps: int = 1
    max_grad_norm: float = 1.0

    # Runtime
    device: str = "cuda"
    seed: int = 42

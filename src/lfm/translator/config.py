"""Configuration for IPA-English pair generation and translator training."""

from __future__ import annotations

from lfm.config.base import LFMBaseConfig


class PairGenerationConfig(LFMBaseConfig):
    """Configuration for generating (IPA, English) parallel pairs.

    The pair generation pipeline:
    1. Load English sentences from Leipzig corpus
    2. Encode with sentence-transformer -> embeddings
    3. Pass through trained expression game -> multi-segment IPA
    4. Save as JSONL for downstream training

    Attributes:
        leipzig_dir: Path to Leipzig corpus directory.
        languages: Language codes to load from Leipzig.
        max_sentences: Maximum sentences to process.
        min_line_length: Minimum character length for source sentences.
        decoder_path: Path to pretrained VAE decoder checkpoint.
        spm_path: Path to sentencepiece model.
        expression_checkpoint: Path to trained expression game checkpoint.
        max_segments: Max segments (must match expression game training).
        encoder_model: Sentence-transformer model name for encoding.
        encode_batch_size: Batch size for encoding and generation.
        output_path: Where to save the JSONL pairs file.
        device: Torch device string.
        seed: Random seed.
    """

    embedding_store_dir: str = "data/embeddings"
    decoder_path: str = "data/vae_decoder.pt"
    spm_path: str = "data/spm.model"
    expression_checkpoint: str = "data/expression_game/best.pt"
    max_segments: int = 16
    batch_size: int = 64
    output_path: str = "data/translator/pairs.jsonl"
    device: str = "cuda"
    seed: int = 42


class TranslatorConfig(LFMBaseConfig):
    """Configuration for IPA -> English translator training and evaluation.

    Supports any HuggingFace causal LM, with optional LoRA for larger models.

    Attributes:
        model_name: HuggingFace model identifier.
        use_lora: Enable LoRA parameter-efficient fine-tuning.
        lora_r: LoRA rank.
        lora_alpha: LoRA alpha scaling.
        lora_target_modules: Modules to apply LoRA to.
        epochs: Number of training epochs.
        lr: Learning rate.
        batch_size: Training batch size.
        max_len: Maximum sequence length for tokenization.
        gradient_accumulation_steps: Gradient accumulation steps.
        use_amp: Enable mixed precision training.
        warmup_fraction: Fraction of steps for LR warmup.
        max_grad_norm: Gradient clipping norm.
        pairs_path: Path to JSONL pairs file.
        val_fraction: Fraction held out for validation.
        eval_max_samples: Max samples for evaluation.
        eval_max_new_tokens: Max new tokens during generation.
        eval_temperature: Sampling temperature for generation.
        output_dir: Directory for model, results, and history.
        device: Torch device string.
        seed: Random seed.
    """

    # Model
    model_name: str = "Qwen/Qwen2.5-0.5B"
    use_lora: bool = False
    lora_r: int = 16
    lora_alpha: int = 32
    lora_target_modules: list[str] = ["q_proj", "v_proj"]

    # Training
    epochs: int = 3
    lr: float = 2e-5
    batch_size: int = 8
    max_len: int = 256
    gradient_accumulation_steps: int = 4
    use_amp: bool = True
    warmup_fraction: float = 0.1
    max_grad_norm: float = 1.0

    # Data
    pairs_path: str = "data/translator/pairs.jsonl"
    val_fraction: float = 0.1

    # Eval
    eval_max_samples: int = 200
    eval_max_new_tokens: int = 64
    eval_temperature: float = 0.7

    # Output
    output_dir: str = "data/translator"
    device: str = "cuda"
    seed: int = 42

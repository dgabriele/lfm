from __future__ import annotations

from pydantic import BaseModel


class SynthConfig(BaseModel):
    # Alien vocabulary
    vocab_size: int = 32_000   # BPE vocab size (controls alien_emb / alien_head dimensions)
    vocab_seed: int = 42

    # Base LM
    base_model_name: str = "Qwen/Qwen2.5-0.5B"

    # Embedding conditioning
    source_embedding_dim: int = 384
    n_prefix_tokens: int = 8

    # Phase 1: alien LM fine-tuning
    phase1_dataset_dir: str = ""
    phase1_batch_size: int = 32
    phase1_grad_accum: int = 1
    phase1_lr: float = 1e-4              # peak cipher_lr (alien_emb + alien_head)
    phase1_body_lr: float = 3e-5         # peak body lr after warmup ends
    phase1_lr_min: float = 0.0           # final lr at end of cosine decay (for both groups)
    phase1_lr_schedule: str = "constant" # "constant" or "cosine"
    phase1_body_warmup_steps: int = 0    # body frozen for first N steps (0 = always trainable)
    phase1_hidden_mse_weight: float = 0.0
    phase1_steps: int = 100_000
    phase1_max_len: int = 128
    phase1_filter_truncated: bool = False  # if True, drop sentences whose tokenisation > max_len
    phase1_log_every: int = 500
    phase1_diag_every: int = 0
    phase1_checkpoint_every: int = 10_000

    # Phase 2: embedding conditioning
    phase2_store_dir: str = ""
    phase2_batch_size: int = 64
    phase2_lr: float = 1e-3
    phase2_steps: int = 30_000
    phase2_max_len: int = 128
    phase2_length_loss_weight: float = 0.1
    phase2_log_every: int = 250
    phase2_checkpoint_every: int = 5_000

    # Generation
    length_slack: int = 10

    # Output
    output_dir: str = "data/synth"
    device: str = "cuda"
    seed: int = 42

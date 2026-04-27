from __future__ import annotations

from pydantic import BaseModel, Field


class SynthConfig(BaseModel):
    # ---- Alien vocabulary ----
    vocab_size: int = 8000
    vocab_seed: int = 42

    # ---- Base model ----
    base_model_name: str = "google/mt5-large"

    # ---- Embedding conditioning ----
    source_embedding_dim: int = 384        # e.g. MiniLM / all-MiniLM-L6-v2
    n_prefix_tokens: int = 8              # encoder-side prefix slots

    # ---- Phase 1: cipher fine-tuning ----
    phase1_dataset_dir: str = ""          # path to DepTreeVAE dataset (HDF5) or .jsonl
    phase1_batch_size: int = 32
    phase1_lr: float = 1e-4
    phase1_steps: int = 100_000
    phase1_max_source_len: int = 128
    phase1_max_target_len: int = 128
    phase1_log_every: int = 500
    phase1_diag_every: int = 0              # 0 = disabled; diagnostic metrics every N steps
    phase1_ar_every: int = 0                # 0 = disabled; AR loss every N steps
    phase1_ar_weight: float = 0.5           # weight of AR loss relative to TF loss
    phase1_ar_batch_size: int = 8           # samples per AR loss step (keep small, involves generate())
    phase1_checkpoint_every: int = 10_000

    # ---- Phase 2: embedding conditioning ----
    phase2_store_dir: str = ""            # path to embedding store (embeddings.npy + passages.jsonl)
    phase2_batch_size: int = 64
    phase2_lr: float = 1e-3
    phase2_steps: int = 30_000
    phase2_max_target_len: int = 128
    phase2_length_loss_weight: float = 0.1
    phase2_log_every: int = 250
    phase2_checkpoint_every: int = 5_000

    # ---- Generation ----
    length_slack: int = 10                # extra tokens beyond predicted length

    # ---- Output ----
    output_dir: str = "data/synth"
    device: str = "cuda"
    seed: int = 42

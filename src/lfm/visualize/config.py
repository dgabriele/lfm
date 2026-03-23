"""Configuration for visualization pipeline."""

from __future__ import annotations

from pydantic import BaseModel


class VisualizeConfig(BaseModel):
    """Configuration shared across all visualization types."""

    model_config = {"frozen": True, "extra": "allow"}

    checkpoint: str
    spm_model: str = "data/spm.model"
    corpus_cache: str = "data/preprocessed_cache.pt"
    output_dir: str = "output/viz"
    format: str = "png"
    device: str = "cuda"
    batch_size: int = 256
    max_samples: int = 50000
    dpi: int = 150
    seed: int = 42

    # Per-visualization overrides (set by CLI subcommands)
    color_by: str = "language"
    perplexity: int = 30
    method: str = "tsne"
    metric: str = "cosine"
    linkage: str = "ward"
    n_samples: int = 10000
    n_sentences: int = 20
    heads: str = ""
    pairs: str = ""
    steps: int = 20

    @classmethod
    def from_args(cls, args) -> VisualizeConfig:
        """Build config from argparse Namespace."""
        d = {}
        for field in cls.model_fields:
            arg_name = field.replace("-", "_")
            if hasattr(args, arg_name):
                d[field] = getattr(args, arg_name)
        return cls(**d)

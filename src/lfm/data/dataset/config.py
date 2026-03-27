"""Configuration and data types for dataset generation."""

from __future__ import annotations

from typing import Any, TypedDict

from lfm.config.base import LFMBaseConfig
from lfm.data.sanitize import SanitizeConfig


class ProcessedSample(TypedDict):
    """A fully processed sample ready for HDF5 storage."""

    seq: int
    language: str
    source: str
    source_file: str
    raw: str
    ipa: str
    ipa_length: int


class LLMGateConfig(LFMBaseConfig):
    """Configuration for the LLM quality gatekeeper.

    Uses a small causal LM (e.g. Qwen2.5-0.5B) as a prompted validator
    that accepts, fixes, or rejects sanitized text before IPA conversion.
    """

    enabled: bool = True
    model_name: str = "Qwen/Qwen2.5-0.5B"
    batch_size: int = 32
    device: str = "cuda"
    max_new_tokens: int = 256
    temperature: float = 0.1


class DatasetGenerateConfig(LFMBaseConfig):
    """Configuration for dataset generation.

    Controls corpus loading, sanitization, IPA conversion, balancing,
    and HDF5 output.
    """

    source: str                         # Registry name ("leipzig", "scientific", etc.)
    source_config: dict[str, Any] = {}  # Forwarded to CorpusLoaderConfig subclass
    output: str = ""                    # Output dir (default: data/datasets/<source>)
    languages: list[str] = []           # Empty = all available
    max_samples: int = 50000            # Per-language cap
    min_samples: int = 100              # Exclude languages below this
    sanitize: SanitizeConfig = SanitizeConfig()
    llm_gate: LLMGateConfig = LLMGateConfig()
    extract_constituents: bool = False
    min_constituent_length: int = 10
    num_workers: int | None = None
    seed: int = 42

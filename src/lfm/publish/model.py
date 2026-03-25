"""HuggingFace model publishing for LFM decoder checkpoints."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import torch

from lfm.publish.base import HFPublisher

logger = logging.getLogger(__name__)


class ModelRelease(HFPublisher):
    """Publish a pretrained LFM decoder to HuggingFace Hub.

    Uploads the decoder checkpoint, sentencepiece model, config snapshot,
    and training history.

    Args:
        repo_id: HuggingFace repo ID (e.g. ``"username/lfm-decoder-v1"``).
        private: Whether the repo should be private.
        token: HuggingFace API token.
    """

    def __init__(
        self,
        repo_id: str,
        private: bool = False,
        token: str | None = None,
    ) -> None:
        super().__init__(repo_id, repo_type="model", private=private, token=token)

    def _collect_files(self, **kwargs: Any) -> list[tuple[str, str]]:
        """Collect decoder checkpoint, SPM model, config, and history."""
        model_dir = Path(kwargs.get("model_dir", "data/models/v1"))
        files: list[tuple[str, str]] = []

        # Required files
        decoder_path = model_dir / "vae_decoder.pt"
        spm_path = model_dir / "spm.model"
        spm_vocab = model_dir / "spm.vocab"

        if not decoder_path.exists():
            raise FileNotFoundError(f"Decoder checkpoint not found: {decoder_path}")
        if not spm_path.exists():
            raise FileNotFoundError(f"SPM model not found: {spm_path}")

        files.append(("vae_decoder.pt", str(decoder_path)))
        files.append(("spm.model", str(spm_path)))
        if spm_vocab.exists():
            files.append(("spm.vocab", str(spm_vocab)))

        # Optional files
        config_path = model_dir / "config.yaml"
        if config_path.exists():
            files.append(("config.yaml", str(config_path)))

        history_path = model_dir / "training_history.json"
        if history_path.exists():
            files.append(("training_history.json", str(history_path)))

        return files

    def _build_card(self, **kwargs: Any) -> str:
        """Generate a HuggingFace model card."""
        model_dir = Path(kwargs.get("model_dir", "data/models/v1"))

        # Extract checkpoint metadata
        decoder_path = model_dir / "vae_decoder.pt"
        meta = {}
        if decoder_path.exists():
            ckpt = torch.load(decoder_path, map_location="cpu", weights_only=False)
            for k in ["latent_dim", "vocab_size", "decoder_hidden_dim",
                       "decoder_num_layers", "decoder_num_heads", "max_seq_len"]:
                if k in ckpt:
                    meta[k] = ckpt[k]
            if "val_loss" in ckpt:
                meta["val_loss"] = f"{ckpt['val_loss']:.4f}"
            if "z_std" in ckpt:
                meta["z_std_mean"] = f"{ckpt['z_std'].mean():.4f}"
                meta["z_active"] = int((ckpt["z_std"] > 0.01).sum())

        # Load config if available
        description = kwargs.get("description", "")

        return f"""---
license: mit
tags:
  - lfm
  - vae
  - multilingual
  - ipa
  - phonetics
  - language-faculty
library_name: pytorch
---

# LFM Decoder — Pretrained Multilingual VAE

A frozen autoregressive transformer decoder pretrained on IPA transcriptions
of 16 typologically diverse languages from the Leipzig Corpora Collection.

Part of the [Language Faculty Model](https://github.com/dgabriele/lfm) framework.

{description}

## Architecture

| Parameter | Value |
|-----------|-------|
| Latent dim | {meta.get('latent_dim', '?')} |
| Vocab size | {meta.get('vocab_size', '?')} |
| Hidden dim | {meta.get('decoder_hidden_dim', '?')} |
| Layers | {meta.get('decoder_num_layers', '?')} |
| Heads | {meta.get('decoder_num_heads', '?')} |
| Max seq len | {meta.get('max_seq_len', '?')} |
| Val loss | {meta.get('val_loss', '?')} |
| Active z dims | {meta.get('z_active', '?')}/{meta.get('latent_dim', '?')} |

## Languages

Arabic, Czech, English, Estonian, Finnish, German, Hindi, Hungarian,
Indonesian, Korean, Polish, Portuguese, Russian, Spanish, Turkish, Vietnamese.

## Usage

```python
from lfm.faculty.config import FacultyConfig
from lfm.faculty.model import LanguageFaculty
from lfm.generator.config import GeneratorConfig

config = FacultyConfig(
    dim=384,
    generator=GeneratorConfig(
        pretrained_decoder_path="vae_decoder.pt",
        spm_model_path="spm.model",
        freeze_decoder=True,
    ),
)
faculty = LanguageFaculty(config)
```

## Files

- `vae_decoder.pt` — Decoder checkpoint (latent projection + decoder + output head + z stats)
- `spm.model` — Sentencepiece tokenizer (BPE, {meta.get('vocab_size', '?')} vocab)
- `spm.vocab` — Sentencepiece vocabulary
- `config.yaml` — Training configuration snapshot
- `training_history.json` — Training session log

## License

MIT
"""

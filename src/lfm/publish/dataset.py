"""HuggingFace dataset publishing for LFM IPA corpora."""

from __future__ import annotations

import json
import logging
from collections import Counter
from pathlib import Path
from typing import Any

import torch

from lfm.publish.base import HFPublisher

logger = logging.getLogger(__name__)


class DatasetRelease(HFPublisher):
    """Publish an LFM IPA corpus to HuggingFace Hub as a dataset.

    Exports the preprocessed cache to JSONL and uploads with a dataset
    card describing languages, preprocessing, and statistics.

    Args:
        repo_id: HuggingFace repo ID (e.g. ``"username/lfm-ipa-16lang"``).
        private: Whether the repo should be private.
        token: HuggingFace API token.
    """

    def __init__(
        self,
        repo_id: str,
        private: bool = False,
        token: str | None = None,
    ) -> None:
        super().__init__(repo_id, repo_type="dataset", private=private, token=token)

    def _collect_files(self, **kwargs: Any) -> list[tuple[str, str]]:
        """Export cache to JSONL and collect for upload."""
        model_dir = Path(kwargs.get("model_dir", "data/models/v1"))
        cache_path = model_dir / "preprocessed_cache.pt"
        spm_path = model_dir / "spm.model"

        if not cache_path.exists():
            raise FileNotFoundError(f"Preprocessed cache not found: {cache_path}")

        # Load cache and export to JSONL
        cache = torch.load(cache_path, map_location="cpu", weights_only=False)
        token_ids_list = cache["token_ids_list"]
        languages = cache["languages"]

        # Decode token IDs back to IPA text
        try:
            import sentencepiece as spm
            sp = spm.SentencePieceProcessor(model_file=str(spm_path))
            has_spm = True
        except (ImportError, OSError):
            has_spm = False
            logger.warning("sentencepiece not available — exporting token IDs only")

        output_dir = model_dir / "export"
        output_dir.mkdir(parents=True, exist_ok=True)
        jsonl_path = output_dir / "ipa_corpus.jsonl"

        with open(jsonl_path, "w", encoding="utf-8") as f:
            for ids, lang in zip(token_ids_list, languages):
                record = {"language": lang, "token_ids": ids}
                if has_spm:
                    record["ipa_text"] = sp.decode(ids)
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

        logger.info("Exported %d samples to %s", len(token_ids_list), jsonl_path)

        files: list[tuple[str, str]] = [
            ("data/ipa_corpus.jsonl", str(jsonl_path)),
        ]

        if spm_path.exists():
            files.append(("spm.model", str(spm_path)))

        # Include config if available
        config_path = model_dir / "config.yaml"
        if config_path.exists():
            files.append(("config.yaml", str(config_path)))

        # Store stats for the card
        self._stats = self._compute_stats(token_ids_list, languages)

        return files

    def _build_card(self, **kwargs: Any) -> str:
        """Generate a HuggingFace dataset card."""
        stats = getattr(self, "_stats", {})
        description = kwargs.get("description", "")

        lang_table = ""
        for lang, count in sorted(
            stats.get("language_counts", {}).items(), key=lambda x: -x[1]
        ):
            pct = 100 * count / stats.get("total", 1)
            lang_table += f"| {lang} | {count:,} | {pct:.1f}% |\n"

        return f"""---
license: mit
task_categories:
  - text-generation
language:
  - ar
  - cs
  - de
  - en
  - et
  - fi
  - hi
  - hu
  - id
  - ko
  - pl
  - pt
  - ru
  - es
  - tr
  - vi
tags:
  - lfm
  - ipa
  - phonetics
  - multilingual
  - linguistic-typology
size_categories:
  - 100K<n<1M
---

# LFM IPA Corpus — 16 Languages

IPA (International Phonetic Alphabet) transcriptions of text from 16
typologically diverse languages, preprocessed for VAE decoder pretraining
in the [Language Faculty Model](https://github.com/dgabriele/lfm) framework.

{description}

## Statistics

| Metric | Value |
|--------|-------|
| Total samples | {stats.get('total', '?'):,} |
| Languages | {stats.get('num_languages', '?')} |
| Mean sequence length | {stats.get('mean_len', '?'):.1f} tokens |
| Vocabulary size | {stats.get('vocab_size', '?'):,} |

## Language Distribution

| Language | Samples | % |
|----------|---------|---|
{lang_table}

## Preprocessing

1. Source text from [Leipzig Corpora Collection](https://wortschatz.uni-leipzig.de/)
2. Sanitized: strip URLs, emails, digits; require 70%+ linguistic characters
3. IPA transcription via epitran (non-English) and CMU dict (English)
4. Hindi/Arabic post-processing: loanword filtering, schwa deletion, script cleanup
5. Sentencepiece BPE tokenization (vocab size {stats.get('vocab_size', '?'):,})

## File Format

`data/ipa_corpus.jsonl` — one JSON object per line:

```json
{{"language": "eng", "ipa_text": "ðʌ kæt sæt ɑn ðʌ mæt", "token_ids": [42, 156, ...]}}
```

## Usage

```python
from datasets import load_dataset
ds = load_dataset("{kwargs.get('repo_id', 'username/lfm-ipa-16lang')}")
```

## License

MIT
"""

    @staticmethod
    def _compute_stats(
        token_ids_list: list[list[int]],
        languages: list[str],
    ) -> dict[str, Any]:
        """Compute corpus statistics for the dataset card."""
        lang_counts = Counter(languages)
        all_lens = [len(ids) for ids in token_ids_list]
        all_tokens = set()
        for ids in token_ids_list:
            all_tokens.update(ids)

        return {
            "total": len(token_ids_list),
            "num_languages": len(lang_counts),
            "language_counts": dict(lang_counts),
            "mean_len": sum(all_lens) / max(len(all_lens), 1),
            "vocab_size": max(all_tokens) + 1 if all_tokens else 0,
        }

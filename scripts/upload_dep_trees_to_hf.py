"""Upload English IPA dependency tree dataset to HuggingFace Hub.

Usage:
    # Step 1: Convert JSONL to parquet shards (no token needed)
    poetry run python3 scripts/upload_dep_trees_to_hf.py --convert-only

    # Step 2: Upload (requires token)
    export HF_TOKEN=hf_xxxxx
    poetry run python3 scripts/upload_dep_trees_to_hf.py

    # Or both steps at once:
    poetry run python3 scripts/upload_dep_trees_to_hf.py --token hf_xxxxx

The dataset is uploaded as sharded Parquet files for efficient streaming.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

REPO_ID = "dgabri3le/english-ipa-dep-treebank"
DATASET_PATH = Path("data/datasets/english-dep-trees-v16/sentences.jsonl")
PARQUET_DIR = Path("/tmp/english-ipa-dep-treebank-parquet-v2")
SHARD_SIZE = 500_000  # rows per parquet shard


DATASET_CARD = """\
---
license: mit
task_categories:
  - token-classification
  - text-generation
language:
  - en
tags:
  - ipa
  - phonetics
  - phonology
  - universal-dependencies
  - dependency-parsing
  - syntax
  - treebank
  - nlp
size_categories:
  - 10M<n<100M
configs:
  - config_name: default
    data_files:
      - split: train
        path: "data/train-*.parquet"
---

# English IPA Universal Dependencies Treebank

A large-scale dataset of **13.9 million English sentences** paired with IPA (International Phonetic Alphabet) transcriptions and Universal Dependencies syntactic annotations.

## Dataset Description

Each example contains:
- The original English sentence (lowercased, cleaned)
- A broad IPA transcription of the sentence
- Universal Dependencies relation labels for each word
- Dependency head indices for each word
- A tagged IPA format interleaving dependency labels with IPA tokens

This dataset bridges phonological and syntactic representations, enabling research at the syntax-phonology interface.

## Fields

| Field | Type | Description |
|-------|------|-------------|
| `english` | `string` | Original English text (lowercased) |
| `ipa` | `string` | IPA transcription (space-separated words) |
| `dep_labels` | `list[string]` | Universal Dependencies relation label per word |
| `dep_heads` | `list[int]` | Head index per word (0 = root) |
| `tagged_ipa` | `string` | Interleaved format: `[dep_label] ipa_word [dep_label] ipa_word ...` |

## Example

```json
{
  "english": "the cat sat on the mat",
  "ipa": "ðʌ kæt sæt ɑn ðʌ mæt",
  "dep_labels": ["det", "nsubj", "root", "case", "det", "obl"],
  "dep_heads": [2, 3, 0, 6, 6, 3],
  "tagged_ipa": "[det] ðʌ [nsubj] kæt [root] sæt [case] ɑn [det] ðʌ [obl] mæt"
}
```

## Statistics

- **13,904,649** sentences
- **~9.4 GB** raw JSONL
- IPA transcription via CMU Pronouncing Dictionary + fallback g2p
- Dependency parses from spaCy (en_core_web_sm model)
- Universal Dependencies v2 label set

## Use Cases

- **Phonology research**: Large-scale IPA corpus for English phonotactic modeling
- **Syntax-phonology interface**: Study prosodic phrasing, stress assignment, and syntactic conditioning of phonological processes
- **Dependency parsing**: Train or evaluate parsers on IPA input
- **Token classification**: Predict dependency relations from phonological form
- **Language modeling**: Train models on IPA text with syntactic structure
- **Linguistic typology**: Compare English syntactic patterns at scale

## Loading the Dataset

```python
from datasets import load_dataset

# Stream (recommended for this size)
ds = load_dataset("dgabriele/english-ipa-ud-treebank", split="train", streaming=True)
for example in ds:
    print(example["tagged_ipa"])
    break

# Or load fully into memory (requires ~30GB RAM)
ds = load_dataset("dgabriele/english-ipa-ud-treebank", split="train")
```

## Citation

If you use this dataset, please cite:

```bibtex
@dataset{english_ipa_ud_treebank_2026,
  title={English IPA Universal Dependencies Treebank},
  author={Gabriel, Daniel},
  year={2026},
  publisher={HuggingFace},
  url={https://huggingface.co/datasets/dgabriele/english-ipa-ud-treebank}
}
```

## License

MIT
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Upload dep trees to HuggingFace")
    parser.add_argument("--token", type=str, default=None, help="HF token (or set HF_TOKEN env var)")
    parser.add_argument("--repo-id", type=str, default=REPO_ID, help="HuggingFace repo ID")
    parser.add_argument("--dataset-path", type=str, default=str(DATASET_PATH))
    parser.add_argument("--parquet-dir", type=str, default=str(PARQUET_DIR))
    parser.add_argument("--private", action="store_true", help="Make repo private")
    parser.add_argument("--shard-size", type=int, default=SHARD_SIZE, help="Rows per parquet shard")
    parser.add_argument("--convert-only", action="store_true", help="Only convert to parquet, don't upload")
    parser.add_argument("--upload-only", action="store_true", help="Only upload (parquet must already exist)")
    return parser.parse_args()


def convert_to_parquet_shards(dataset_path: Path, output_dir: Path, shard_size: int) -> list[Path]:
    """Convert JSONL to sharded Parquet files with zstd compression."""
    schema = pa.schema([
        ("raw_english", pa.string()),
        ("english", pa.string()),
        ("ipa", pa.string()),
        ("dep_labels", pa.list_(pa.string())),
        ("dep_heads", pa.list_(pa.int32())),
        ("tagged_ipa", pa.string()),
    ])

    output_dir.mkdir(parents=True, exist_ok=True)

    # Check for existing shards (resume support)
    existing = sorted(output_dir.glob("train-*.parquet"))
    if existing:
        logger.info("Found %d existing parquet shards in %s — reusing", len(existing), output_dir)
        return existing

    shard_paths: list[Path] = []
    current_rows = 0
    total_rows = 0
    shard_idx = 0
    writer = None
    current_shard_path = None

    batch_raw_english: list[str] = []
    batch_english: list[str] = []
    batch_ipa: list[str] = []
    batch_dep_labels: list[list[str]] = []
    batch_dep_heads: list[list[int]] = []
    batch_tagged_ipa: list[str] = []

    BATCH_SIZE = 50_000

    with open(dataset_path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            record = json.loads(line)
            batch_raw_english.append(record.get("raw_english", record["english"]))
            batch_english.append(record["english"])
            batch_ipa.append(record["ipa"])
            batch_dep_labels.append(record["dep_labels"])
            batch_dep_heads.append(record["dep_heads"])
            batch_tagged_ipa.append(record["tagged_ipa"])

            if len(batch_english) >= BATCH_SIZE:
                # Write batch
                if writer is None:
                    # Use placeholder name, will rename at end
                    current_shard_path = output_dir / f"train-{shard_idx:05d}.parquet.tmp"
                    writer = pq.ParquetWriter(str(current_shard_path), schema, compression="zstd")

                batch = pa.RecordBatch.from_pydict(
                    {
                        "raw_english": batch_raw_english,
                        "english": batch_english,
                        "ipa": batch_ipa,
                        "dep_labels": batch_dep_labels,
                        "dep_heads": batch_dep_heads,
                        "tagged_ipa": batch_tagged_ipa,
                    },
                    schema=schema,
                )
                writer.write_batch(batch)
                current_rows += len(batch_english)
                total_rows += len(batch_english)
                batch_raw_english = []
                batch_english = []
                batch_ipa = []
                batch_dep_labels = []
                batch_dep_heads = []
                batch_tagged_ipa = []

                if current_rows >= shard_size:
                    writer.close()
                    shard_paths.append(current_shard_path)
                    logger.info(
                        "Shard %d complete: %d rows (total: %d)",
                        shard_idx, current_rows, total_rows,
                    )
                    shard_idx += 1
                    current_rows = 0
                    writer = None

    # Final partial batch
    if batch_english:
        if writer is None:
            current_shard_path = output_dir / f"train-{shard_idx:05d}.parquet.tmp"
            writer = pq.ParquetWriter(str(current_shard_path), schema, compression="zstd")

        batch = pa.RecordBatch.from_pydict(
            {
                "raw_english": batch_raw_english,
                "english": batch_english,
                "ipa": batch_ipa,
                "dep_labels": batch_dep_labels,
                "dep_heads": batch_dep_heads,
                "tagged_ipa": batch_tagged_ipa,
            },
            schema=schema,
        )
        writer.write_batch(batch)
        current_rows += len(batch_english)
        total_rows += len(batch_english)

    # Close final shard
    if writer is not None:
        writer.close()
        shard_paths.append(current_shard_path)
        logger.info("Final shard %d: %d rows (total: %d)", shard_idx, current_rows, total_rows)

    # Rename with correct total count in filename
    total_shards = len(shard_paths)
    renamed_paths = []
    for idx, tmp_path in enumerate(shard_paths):
        final_name = f"train-{idx:05d}-of-{total_shards:05d}.parquet"
        final_path = output_dir / final_name
        tmp_path.rename(final_path)
        renamed_paths.append(final_path)

    logger.info("Conversion complete: %d shards, %d total rows in %s", total_shards, total_rows, output_dir)
    return renamed_paths


def upload_dataset(parquet_dir: Path, repo_id: str, token: str, private: bool = False) -> str:
    """Upload parquet shards and dataset card to HuggingFace using upload_folder."""
    from huggingface_hub import HfApi, login

    login(token=token)
    api = HfApi(token=token)

    # Create the repo
    repo_url = api.create_repo(
        repo_id=repo_id,
        repo_type="dataset",
        private=private,
        exist_ok=True,
    )
    logger.info("Repo: %s", repo_url)

    # Copy README.md into the parquet dir for upload_folder
    readme_path = parquet_dir / "README.md"
    external_readme = Path("/tmp/english-ipa-dep-treebank-readme.md")
    if external_readme.exists():
        import shutil
        shutil.copy2(external_readme, readme_path)
    else:
        readme_path.write_text(DATASET_CARD, encoding="utf-8")

    # Create data/ subdirectory with symlinks to parquet files
    data_dir = parquet_dir / "data"
    data_dir.mkdir(exist_ok=True)
    for p in sorted(parquet_dir.glob("train-*.parquet")):
        link = data_dir / p.name
        if not link.exists():
            link.symlink_to(p)

    # Upload the entire folder
    logger.info("Uploading folder %s to %s ...", parquet_dir, repo_id)
    api.upload_folder(
        folder_path=str(parquet_dir),
        repo_id=repo_id,
        repo_type="dataset",
        allow_patterns=["README.md", "data/*.parquet"],
    )

    url = f"https://huggingface.co/datasets/{repo_id}"
    logger.info("Upload complete: %s", url)
    return url


def main():
    args = parse_args()

    parquet_dir = Path(args.parquet_dir)

    # Convert step
    if not args.upload_only:
        dataset_path = Path(args.dataset_path)
        if not dataset_path.exists():
            raise FileNotFoundError(f"Dataset not found: {dataset_path}")
        shard_paths = convert_to_parquet_shards(dataset_path, parquet_dir, args.shard_size)
        if args.convert_only:
            logger.info("Conversion done. Run without --convert-only to upload.")
            return

    # Upload step
    token = args.token or os.environ.get("HF_TOKEN")
    if not token:
        raise ValueError(
            "No HuggingFace token found. Set HF_TOKEN env var or pass --token.\n"
            "Get a write token at https://huggingface.co/settings/tokens"
        )

    # Verify parquet shards exist
    existing = sorted(parquet_dir.glob("train-*.parquet"))
    if not existing:
        raise FileNotFoundError(
            f"No parquet shards found in {parquet_dir}. "
            "Run with --convert-only first, or without --upload-only."
        )

    upload_dataset(parquet_dir, args.repo_id, token, args.private)


if __name__ == "__main__":
    main()

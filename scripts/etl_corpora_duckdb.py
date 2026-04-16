"""
ETL: populate `data/corpora.duckdb` from `data/datasets/*`.

Design:
  - Each dataset directory ships its own `corpus.json` (display metadata)
    and a `samples.parquet` (canonical row store).  Optional `meta.json`
    carries stats produced by `scripts/corpus_to_parquet.py`.
  - This ETL reads those files and upserts into a single `corpora`
    metadata table in `data/corpora.duckdb`.  The parquet files are NOT
    copied into the duckdb file — they are queried in place at request
    time via `read_parquet(path)`.  That keeps the duckdb file small
    (just metadata), lets ML pipelines read the same parquets directly
    with polars/pandas/HF, and gets us row-group skipping for free when
    paginating by the monotonic `index` column.

Idempotent: re-running updates stats and keeps `created_at` on insert
and bumps `updated_at` always.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import duckdb


SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS corpora (
    id            VARCHAR PRIMARY KEY,
    name          VARCHAR NOT NULL,
    description   TEXT,
    tags          VARCHAR[],
    -- vae_type: which decoder/tokenization regime this corpus
    -- belongs to ('ipa' or 'token_vocab').  Authoritative — UI
    -- derives the model's VAE type from this column rather than
    -- asking the user.
    vae_type      VARCHAR,
    source_paths  VARCHAR[],
    parquet_path  VARCHAR NOT NULL,
    sample_count  BIGINT,
    vocab_size    INTEGER,
    max_seq_len   INTEGER,
    length_min    INTEGER,
    length_max    INTEGER,
    length_mean   DOUBLE,
    length_p50    INTEGER,
    length_p99    INTEGER,
    created_at    TIMESTAMP DEFAULT now(),
    updated_at    TIMESTAMP DEFAULT now()
);
"""


def discover(datasets_root: Path, db_dir: Path) -> list[dict]:
    entries: list[dict] = []
    for corpus_json in sorted(datasets_root.glob("*/corpus.json")):
        dataset_dir = corpus_json.parent
        parquet = dataset_dir / "samples.parquet"
        if not parquet.exists():
            print(f"skip {dataset_dir.name}: no samples.parquet")
            continue
        meta_path = dataset_dir / "meta.json"
        entry = json.loads(corpus_json.read_text())
        # Store as a path relative to the duckdb file's directory so
        # the same db works on host and inside the container.
        entry["parquet_path"] = str(parquet.resolve().relative_to(db_dir.resolve()))
        if meta_path.exists():
            entry["stats"] = json.loads(meta_path.read_text())
        entries.append(entry)
    return entries


def upsert(conn: duckdb.DuckDBPyConnection, entry: dict) -> None:
    stats = entry.get("stats") or {}
    length = stats.get("length") or {}
    row = {
        "id": entry["id"],
        "name": entry["name"],
        "description": entry.get("description", ""),
        "tags": entry.get("tags", []),
        "vae_type": entry.get("vae_type"),
        "source_paths": entry.get("source_paths", []),
        "parquet_path": entry["parquet_path"],
        "sample_count": stats.get("total"),
        "vocab_size": entry.get("vocab_size"),
        "max_seq_len": entry.get("max_seq_len"),
        "length_min": length.get("min"),
        "length_max": length.get("max"),
        "length_mean": length.get("mean"),
        "length_p50": length.get("p50"),
        "length_p99": length.get("p99"),
    }
    conn.execute(
        """
        INSERT INTO corpora (
            id, name, description, tags, vae_type, source_paths, parquet_path,
            sample_count, vocab_size, max_seq_len,
            length_min, length_max, length_mean, length_p50, length_p99,
            updated_at
        ) VALUES (
            $id, $name, $description, $tags, $vae_type, $source_paths, $parquet_path,
            $sample_count, $vocab_size, $max_seq_len,
            $length_min, $length_max, $length_mean, $length_p50, $length_p99,
            now()
        )
        ON CONFLICT (id) DO UPDATE SET
            name         = EXCLUDED.name,
            description  = EXCLUDED.description,
            tags         = EXCLUDED.tags,
            vae_type     = EXCLUDED.vae_type,
            source_paths = EXCLUDED.source_paths,
            parquet_path = EXCLUDED.parquet_path,
            sample_count = EXCLUDED.sample_count,
            vocab_size   = EXCLUDED.vocab_size,
            max_seq_len  = EXCLUDED.max_seq_len,
            length_min   = EXCLUDED.length_min,
            length_max   = EXCLUDED.length_max,
            length_mean  = EXCLUDED.length_mean,
            length_p50   = EXCLUDED.length_p50,
            length_p99   = EXCLUDED.length_p99,
            updated_at   = now()
        """,
        row,
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", type=Path, default=Path("data/datasets"))
    ap.add_argument("--db", type=Path, default=Path("data/corpora.duckdb"))
    args = ap.parse_args()

    args.db.parent.mkdir(parents=True, exist_ok=True)
    conn = duckdb.connect(str(args.db))
    conn.execute(SCHEMA_SQL)

    entries = discover(args.datasets, args.db.parent)
    if not entries:
        print("no corpora found")
        return

    for e in entries:
        upsert(conn, e)
        print(f"upserted {e['id']} ({e.get('stats', {}).get('total', '?')} rows)")

    rows = conn.execute(
        "SELECT id, sample_count, parquet_path FROM corpora ORDER BY id"
    ).fetchall()
    print(f"\n{len(rows)} corpora in {args.db}:")
    for r in rows:
        print(f"  {r[0]:<40} {r[1]:>12,}  {r[2]}")
    conn.close()


if __name__ == "__main__":
    main()

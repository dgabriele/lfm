"""
Convert a corpus HDF5 file to a parquet shard suitable for DuckDB.

Each dataset directory is expected to contain `samples.h5` with the groups
documented in `data/datasets/README.md` (raw, ipa, language, source, ...).
Output is written as `samples.parquet` alongside it, plus a `meta.json`
capturing counts and length percentiles for the UI.

Runs streaming with pyarrow RecordBatch chunks so memory stays flat even
on 10M+-row corpora.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq


BATCH_ROWS = 100_000


def _decode(arr: np.ndarray) -> list[str]:
    return [
        v.decode("utf-8", errors="replace") if isinstance(v, (bytes, bytearray)) else str(v)
        for v in arr
    ]


def convert(h5_path: Path, out_path: Path) -> dict:
    with h5py.File(h5_path, "r") as f:
        g = f["samples"]
        n = g["raw"].shape[0]
        have_ipa = "ipa" in g
        have_len = "ipa_length" in g
        have_lang = "language" in g
        have_src = "source" in g

        schema_fields = [
            pa.field("index", pa.int64()),
            pa.field("text", pa.string()),
        ]
        if have_ipa:
            schema_fields.append(pa.field("ipa", pa.string()))
        if have_lang:
            schema_fields.append(pa.field("language", pa.string()))
        if have_src:
            schema_fields.append(pa.field("source", pa.string()))
        if have_len:
            schema_fields.append(pa.field("length", pa.int32()))
        schema = pa.schema(schema_fields)

        writer = pq.ParquetWriter(out_path, schema, compression="zstd")
        lengths: list[int] = []
        try:
            for start in range(0, n, BATCH_ROWS):
                end = min(start + BATCH_ROWS, n)
                cols = {
                    "index": np.arange(start, end, dtype=np.int64),
                    "text": _decode(g["raw"][start:end]),
                }
                if have_ipa:
                    cols["ipa"] = _decode(g["ipa"][start:end])
                if have_lang:
                    cols["language"] = _decode(g["language"][start:end])
                if have_src:
                    cols["source"] = _decode(g["source"][start:end])
                if have_len:
                    arr = g["ipa_length"][start:end].astype(np.int32)
                    cols["length"] = arr
                    lengths.extend(arr.tolist())
                writer.write_table(pa.Table.from_pydict(cols, schema=schema))
                print(f"  {end:>10}/{n} ({end / n:5.1%})", flush=True)
        finally:
            writer.close()

    stats: dict = {"total": int(n)}
    if lengths:
        a = np.asarray(lengths, dtype=np.int32)
        stats["length"] = {
            "min": int(a.min()),
            "max": int(a.max()),
            "mean": float(a.mean()),
            "p50": int(np.percentile(a, 50)),
            "p99": int(np.percentile(a, 99)),
        }
    return stats


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset_dir", type=Path, help="e.g. data/datasets/english-constituents-v12")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    h5_path = args.dataset_dir / "samples.h5"
    out_path = args.dataset_dir / "samples.parquet"
    meta_path = args.dataset_dir / "meta.json"

    if not h5_path.exists():
        raise SystemExit(f"no samples.h5 at {h5_path}")
    if out_path.exists() and not args.force:
        raise SystemExit(f"{out_path} exists; pass --force to overwrite")

    print(f"converting {h5_path} -> {out_path}")
    stats = convert(h5_path, out_path)
    meta_path.write_text(json.dumps(stats, indent=2))
    print(f"wrote {out_path} and {meta_path}")


if __name__ == "__main__":
    main()

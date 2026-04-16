"""
Convert a plain-text corpus file (one sample per line) into a parquet
shard compatible with the FE's DuckDB layer.

Use this for corpora whose canonical form is a line-oriented text file
rather than an HDF5 file — e.g. the constituent-tagged output of the
Stanza parse pipeline before the full SPM/h5 build has run.

Output columns mirror what the FE expects: `index`, `text`, optional
`language`, `source`, `length` (character count).  A sibling
`meta.json` captures totals and length percentiles.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq


BATCH_ROWS = 100_000


def convert(
    src: Path, out: Path, *, language: str, source: str, min_length: int
) -> dict:
    schema = pa.schema(
        [
            pa.field("index", pa.int64()),
            pa.field("text", pa.string()),
            pa.field("language", pa.string()),
            pa.field("source", pa.string()),
            pa.field("length", pa.int32()),
        ]
    )
    writer = pq.ParquetWriter(out, schema, compression="zstd")
    lengths: list[int] = []
    written = 0
    try:
        batch_text: list[str] = []
        batch_len: list[int] = []
        with src.open("r", encoding="utf-8") as f:
            for line in f:
                s = line.rstrip("\n")
                if len(s) < min_length:
                    continue
                batch_text.append(s)
                batch_len.append(len(s))
                if len(batch_text) >= BATCH_ROWS:
                    end = written + len(batch_text)
                    writer.write_table(
                        pa.Table.from_pydict(
                            {
                                "index": np.arange(written, end, dtype=np.int64),
                                "text": batch_text,
                                "language": [language] * len(batch_text),
                                "source": [source] * len(batch_text),
                                "length": np.asarray(batch_len, dtype=np.int32),
                            },
                            schema=schema,
                        )
                    )
                    lengths.extend(batch_len)
                    written = end
                    batch_text.clear()
                    batch_len.clear()
                    if written % 1_000_000 == 0:
                        print(f"  {written:>10,}", flush=True)
        if batch_text:
            end = written + len(batch_text)
            writer.write_table(
                pa.Table.from_pydict(
                    {
                        "index": np.arange(written, end, dtype=np.int64),
                        "text": batch_text,
                        "language": [language] * len(batch_text),
                        "source": [source] * len(batch_text),
                        "length": np.asarray(batch_len, dtype=np.int32),
                    },
                    schema=schema,
                )
            )
            lengths.extend(batch_len)
            written = end
    finally:
        writer.close()

    stats: dict = {"total": int(written)}
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
    ap.add_argument("text_file", type=Path)
    ap.add_argument("out_dir", type=Path, help="dataset directory; samples.parquet is written here")
    ap.add_argument("--language", default="eng")
    ap.add_argument("--source", default="constituent")
    ap.add_argument("--min-length", type=int, default=1)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    if not args.text_file.exists():
        raise SystemExit(f"no file: {args.text_file}")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    out = args.out_dir / "samples.parquet"
    meta = args.out_dir / "meta.json"
    if out.exists() and not args.force:
        raise SystemExit(f"{out} exists; pass --force")

    print(f"converting {args.text_file} -> {out}")
    stats = convert(
        args.text_file,
        out,
        language=args.language,
        source=args.source,
        min_length=args.min_length,
    )
    meta.write_text(json.dumps(stats, indent=2))
    print(f"wrote {out} and {meta}")
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Thin wrapper for constituency extraction.

Prefer ``poetry run lfm dataset extract-constituents`` for the full CLI.
"""

from __future__ import annotations

import argparse
import logging
import sys

sys.stderr = open(sys.stderr.fileno(), "w", buffering=1, closefd=False)
sys.stdout = open(sys.stdout.fileno(), "w", buffering=1, closefd=False)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Extract phrase constituents from existing HDF5 dataset",
    )
    parser.add_argument("--dataset", default="data/datasets/leipzig-16lang")
    parser.add_argument("--output", default="data/datasets/leipzig-16lang-constituents")
    parser.add_argument("--min-length", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-samples", type=int, default=None)
    args = parser.parse_args()

    from lfm.data.constituency_extraction import extract_from_dataset

    extract_from_dataset(
        dataset_path=args.dataset,
        output_path=args.output,
        min_length=args.min_length,
        batch_size=args.batch_size,
        max_samples=args.max_samples,
    )

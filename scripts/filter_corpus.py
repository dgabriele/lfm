"""Post-process a multi-sentence corpus JSONL: cap word-count distribution.

Removes paragraphs that fall outside a configured ``[min, max]`` word
count range. Default ``40-300`` keeps each length bucket from 40 to 300
words at ≥1% density (well-covered for training); drops the ~2% sparse
tail (<40 and >300 words) where the model would otherwise be
undertrained on length-conditioned generation.

Usage:
  poetry run python scripts/filter_corpus.py IN.jsonl OUT.jsonl \\
      [--min-words 40] [--max-words 300]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("input")
    p.add_argument("output")
    p.add_argument("--min-words", type=int, default=40)
    p.add_argument("--max-words", type=int, default=300)
    args = p.parse_args()

    in_path = Path(args.input)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    n_in = 0
    n_out = 0
    n_short = 0
    n_long = 0
    with in_path.open() as fh_in, out_path.open("w") as fh_out:
        for line in fh_in:
            line = line.strip()
            if not line:
                continue
            n_in += 1
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            text = rec.get("text", "")
            wc = len(text.split())
            if wc < args.min_words:
                n_short += 1
                continue
            if wc > args.max_words:
                n_long += 1
                continue
            fh_out.write(line + "\n")
            n_out += 1
    print(f"input    : {n_in:,}")
    print(f"output   : {n_out:,}  ({n_out / max(n_in, 1) * 100:.2f}%)")
    print(f"dropped (too short, <{args.min_words}): {n_short:,}")
    print(f"dropped (too long,  >{args.max_words}): {n_long:,}")
    print(f"wrote    : {out_path}")


if __name__ == "__main__":
    main()

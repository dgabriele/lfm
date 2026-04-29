"""Precompute token frequency distribution over the cipher-encoded corpus.

Used by the RTD coherence-head training to sample *plausible* replacement
tokens (proportional to corpus frequency) instead of uniform random tokens.
Plausible replacements are harder to detect — they force the discriminator
to rely on contextual reasoning rather than out-of-distribution surprise.

Output: token_frequencies.npy (vocab_size,) float32, summing to 1.0,
saved into the synth output dir alongside alien_vocab.json.

Usage:
    poetry run python scripts/build_token_frequency.py configs/synth_local_qwen.yaml
"""

from __future__ import annotations

import argparse
import json
import logging
from collections import Counter
from pathlib import Path

import numpy as np
import yaml
from transformers import PreTrainedTokenizerFast

from lfm.synth.cipher import WordCipher
from lfm.synth.config import SynthConfig
from lfm.synth.vocab import AlienVocab

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger(__name__)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("config")
    p.add_argument("--max-len", type=int, default=80)
    p.add_argument("--n-samples", type=int, default=200_000,
                   help="Sample this many sentences from corpus for frequency counting "
                        "(full 1M is overkill; a sample gives stable freqs).")
    args = p.parse_args()

    cfg = SynthConfig(**yaml.safe_load(Path(args.config).read_text()))
    out_dir = Path(cfg.output_dir)
    vocab = AlienVocab.load(out_dir)
    cipher = WordCipher.from_dirs(vocab, out_dir)
    alien_tok = PreTrainedTokenizerFast.from_pretrained(str(out_dir / "alien_tokenizer"))

    dataset_path = Path(cfg.phase1_dataset_dir)
    if dataset_path.suffix == ".jsonl":
        all_lines = [json.loads(l)["text"] for l in dataset_path.read_text().splitlines() if l.strip()]
    else:
        all_lines = [l.strip() for l in dataset_path.read_text().splitlines() if l.strip()]
    sample = all_lines[: args.n_samples]
    logger.info("counting tokens over %d sentences", len(sample))

    # Batch encode for speed
    counts: Counter = Counter()
    chunk = 1024
    for i in range(0, len(sample), chunk):
        cipher_texts = cipher.encode_batch(sample[i : i + chunk])
        encs = alien_tok(cipher_texts, padding=False, truncation=True,
                        max_length=args.max_len, add_special_tokens=True)["input_ids"]
        for ids in encs:
            counts.update(ids)
        if (i // chunk) % 20 == 0:
            logger.info("processed %d / %d", min(i + chunk, len(sample)), len(sample))

    vocab_size = len(alien_tok)
    freqs = np.zeros(vocab_size, dtype=np.float64)
    for tid, c in counts.items():
        freqs[tid] = c

    # Zero out special tokens — replacement should never be a special token
    for tid in (alien_tok.pad_token_id, alien_tok.eos_token_id,
                alien_tok.bos_token_id, alien_tok.sep_token_id,
                alien_tok.unk_token_id, alien_tok.mask_token_id):
        if tid is not None:
            freqs[tid] = 0.0

    total = freqs.sum()
    if total == 0:
        raise RuntimeError("no token counts collected")
    freqs = (freqs / total).astype(np.float32)

    out_path = out_dir / "token_frequencies.npy"
    np.save(out_path, freqs)
    nonzero = int((freqs > 0).sum())
    logger.info("wrote token frequency distribution → %s  (vocab=%d, nonzero=%d, top-token p=%.4f)",
                out_path, vocab_size, nonzero, float(freqs.max()))


if __name__ == "__main__":
    main()

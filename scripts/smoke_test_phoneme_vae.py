#!/usr/bin/env python
"""Smoke-test the PhonemeVAEGenerator: instantiate, forward, decode.

Verifies the new sibling subclass wires up correctly with the phoneme
alphabet before we invest in the full corpus-transcoding + pretraining
pipeline.  Does NOT train — just checks shapes and the tokenizer round-trip.
"""

from __future__ import annotations

import json
from pathlib import Path

import torch

from lfm.generator.config import GeneratorConfig
from lfm.generator.phoneme_tokenizer import PhonemeTokenizer
from lfm.generator.phoneme_vae import PhonemeVAEGenerator

ALPHABET_PATH = Path("data/phoneme_alphabet_multi.json")


def main() -> None:
    print("Loading alphabet:", ALPHABET_PATH)
    tokenizer = PhonemeTokenizer(ALPHABET_PATH)
    print(f"  vocab_size = {tokenizer.vocab_size}")
    print(f"  first 10 phonemes: {tokenizer.phonemes[:10]}")

    # Round-trip check
    test_ids = [0, 1, 2, 3, 4, 5, 6, 7, 8]
    decoded = tokenizer.batch_decode(torch.tensor([test_ids]), word_size=3)
    print(f"  round-trip decode of {test_ids}: {decoded}")

    # Instantiate the VAE generator with phoneme alphabet
    print("\nInstantiating PhonemeVAEGenerator...")
    config = GeneratorConfig(
        latent_dim=256,
        vocab_size=tokenizer.vocab_size,
        max_output_len=32,
        decoder_hidden_dim=128,   # small for smoke test
        decoder_num_heads=8,      # match default head_windows length
        decoder_num_layers=2,
        decoder_dropout=0.0,
        num_memory_tokens=4,
        spm_model_path=str(ALPHABET_PATH),
        use_rope=True,
        share_decoder_layers=True,
    )
    model = PhonemeVAEGenerator(config)
    model.eval()

    n_params = sum(p.numel() for p in model.parameters())
    print(f"  instantiated model: {n_params:,} parameters")
    print(f"  vocab + BOS/EOS: {config.vocab_size + 2}")

    # Forward pass: feed synthetic embeddings in, verify VAE round-trip.
    print("\nRunning forward pass with synthetic embeddings...")
    batch, seq_len, emb_dim = 2, 5, 64
    embeddings = torch.randn(batch, seq_len, emb_dim)
    mask = torch.ones(batch, seq_len, dtype=torch.bool)
    with torch.no_grad():
        out = model(embeddings, mask)
    print(f"  forward output keys: {list(out.keys())}")
    tokens = out["tokens"]
    print(f"  tokens shape: {tokens.shape}")
    print(f"  sample output (batch 0): {tokens[0].tolist()}")

    # Decode to text
    text = model.decode_to_text(tokens)
    print(f"  decoded text:")
    for i, t in enumerate(text):
        print(f"    [{i}] {t!r}")

    print("\nSmoke test passed ✓")


if __name__ == "__main__":
    main()

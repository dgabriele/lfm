"""Corpus generator: source embedding → alien sentence at scale."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import torch
from transformers import PreTrainedTokenizerFast

from lfm.synth.config import SynthConfig
from lfm.synth.model import SynthLM

logger = logging.getLogger(__name__)


class CorpusGenerator:
    """Generate alien-surface sentences from an embedding store.

    Args:
        model: Fully trained SynthLM (phase 1 + phase 2 weights loaded).
        tokenizer: Alien tokenizer (for decoding output token IDs).
        config: SynthConfig.
    """

    def __init__(
        self,
        model: SynthLM,
        tokenizer: PreTrainedTokenizerFast,
        config: SynthConfig,
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
        self.device = torch.device(config.device)
        self._alien_stop_id = tokenizer.eos_token_id
        self._alien_pad_id = tokenizer.pad_token_id

    def generate_corpus(
        self,
        store_dir: str,
        output_path: str,
        batch_size: int = 64,
    ) -> int:
        """Generate one alien sentence per embedding and write to output_path.

        Returns:
            Number of sentences written.
        """
        embeddings = np.load(Path(store_dir) / "embeddings.npy", mmap_mode="r")
        N = len(embeddings)
        out_path = Path(output_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        self.model.eval().to(self.device)
        written = 0

        with out_path.open("w", encoding="utf-8") as fh:
            for start in range(0, N, batch_size):
                end = min(start + batch_size, N)
                emb = torch.tensor(
                    embeddings[start:end].astype(np.float32),
                    dtype=torch.float32,
                    device=self.device,
                )
                token_ids = self.model.generate(
                    emb,
                    eos_id=self._alien_stop_id,
                    pad_id=self._alien_pad_id,
                )
                sentences = self.tokenizer.batch_decode(token_ids, skip_special_tokens=True)
                for s in sentences:
                    fh.write(s.strip() + "\n")
                written += len(sentences)

                if written % 10_000 == 0 or written == N:
                    logger.info("generated %d / %d sentences", written, N)

        logger.info("corpus written to %s (%d sentences)", output_path, written)
        return written

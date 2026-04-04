"""Generate (IPA, English) parallel pairs via the trained expression game.

Uses the same embedding store the expression game was trained on to ensure
the input distribution matches.  Saves pairs as JSONL for downstream
translation training.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import sentencepiece as spm_lib
import torch
import yaml

from lfm.translator.config import PairGenerationConfig

logger = logging.getLogger(__name__)


class PairGenerator:
    """Generate (IPA, English) pairs from a trained expression game.

    Args:
        config: Pair generation configuration.
    """

    def __init__(self, config: PairGenerationConfig) -> None:
        self.config = config

    def generate(self) -> list[tuple[str, str]]:
        """Run the full pair generation pipeline.

        Returns:
            List of ``(ipa_string, english_string)`` tuples.
        """
        cfg = self.config
        torch.manual_seed(cfg.seed)
        device = torch.device(cfg.device)

        # 1. Load embeddings and passage text from the same store
        #    the expression game was trained on
        embeddings, texts = self._load_store()

        # 2. Generate IPA via expression game
        pairs = self._generate_expressions(texts, embeddings, device)

        # 3. Save
        self._save(pairs)

        return pairs

    def _load_store(self) -> tuple[np.ndarray, list[str]]:
        """Load embeddings and passage text from the embedding store."""
        from lfm.embeddings.store import EmbeddingStore

        cfg = self.config
        store = EmbeddingStore(cfg.embedding_store_dir)
        store.load()

        embeddings = np.array(store._embeddings)  # copy out of mmap

        # Load passage text
        passages_path = store.store_dir / "passages.jsonl"
        if not passages_path.is_file():
            raise FileNotFoundError(
                f"No passages.jsonl in {store.store_dir}. "
                f"Regenerate embeddings with the latest precompute_embeddings.py "
                f"to include passage text."
            )

        texts: list[str] = []
        with passages_path.open("r", encoding="utf-8") as fh:
            for line in fh:
                texts.append(json.loads(line).get("text", ""))

        logger.info(
            "Loaded %d embeddings + %d passages from %s",
            len(embeddings), len(texts), cfg.embedding_store_dir,
        )
        return embeddings, texts

    def _generate_expressions(
        self,
        texts: list[str],
        embeddings: np.ndarray,
        device: torch.device,
    ) -> list[tuple[str, str]]:
        """Load expression game, decode multi-phrase expressions."""
        from lfm.agents.games.expression import ExpressionGame, ExpressionGameConfig
        from lfm.faculty.model import LanguageFaculty

        cfg = self.config

        # Build expression game from checkpoint config
        ckpt = torch.load(cfg.expression_checkpoint, map_location=device, weights_only=False)
        game_cfg = ExpressionGameConfig(
            decoder_path=cfg.decoder_path,
            spm_path=cfg.spm_path,
            z_generator=ckpt.get("z_generator", "gru"),
            max_phrases=ckpt.get("max_phrases", ckpt.get("max_segments", cfg.max_phrases)),
            embedding_dim=ckpt.get("embedding_dim", 384),
            z_hidden_dim=ckpt.get("z_hidden_dim", 512),
            num_memory_tokens=ckpt.get("num_memory_tokens", 8),
            max_tokens_per_phrase=ckpt.get("max_tokens_per_phrase", ckpt.get("max_tokens_per_segment", 48)),
            diffusion_steps=ckpt.get("diffusion_steps", 4),
            diffusion_layers=ckpt.get("diffusion_layers", 4),
            device=cfg.device,
        )
        faculty = LanguageFaculty(game_cfg.build_faculty_config()).to(device)
        game = ExpressionGame(game_cfg, faculty).to(device)
        game.load_checkpoint_state(ckpt)
        game.eval()

        # Load SPM for detokenization
        sp = spm_lib.SentencePieceProcessor(model_file=cfg.spm_path)
        vocab_size = sp.vocab_size()
        eos_id = vocab_size + 1

        logger.info(
            "Expression game loaded (step=%d, acc=%.1f%%)",
            ckpt.get("step", -1),
            ckpt.get("accuracy", 0) * 100,
        )

        # Generate in batches
        pairs: list[tuple[str, str]] = []
        batch_size = cfg.batch_size
        n = len(texts)

        with torch.no_grad():
            for start in range(0, n, batch_size):
                end = min(start + batch_size, n)
                emb = torch.tensor(
                    embeddings[start:end], dtype=torch.float32, device=device,
                )

                # z-sequence + multi-phrase decode
                z_out = game.z_gen(emb)
                if len(z_out) == 4:
                    z_seq, _, z_weights, _ = z_out
                else:
                    z_seq, z_weights, _ = z_out
                tokens, gen_mask, _ = game._multiphrase_decode(z_seq, z_weights)

                # Detokenize each expression
                for i in range(end - start):
                    token_ids = tokens[i].tolist()
                    mask = gen_mask[i].tolist()
                    valid_ids = [
                        t for t, m in zip(token_ids, mask)
                        if m and t != eos_id and t < vocab_size
                    ]
                    ipa = sp.decode(valid_ids).strip()
                    if ipa:
                        pairs.append((ipa, texts[start + i]))

                if (start // batch_size) % 10 == 0:
                    logger.info("Generated %d / %d pairs", len(pairs), n)

        # Free VRAM
        del game, faculty
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        logger.info("Generated %d valid IPA-English pairs", len(pairs))
        return pairs

    def _save(self, pairs: list[tuple[str, str]]) -> None:
        """Save pairs as JSONL and config snapshot as YAML."""
        cfg = self.config
        output_path = Path(cfg.output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Save pairs
        with open(output_path, "w", encoding="utf-8") as f:
            for ipa, english in pairs:
                f.write(json.dumps({"ipa": ipa, "english": english}, ensure_ascii=False) + "\n")
        logger.info("Saved %d pairs to %s", len(pairs), output_path)

        # Save config snapshot
        config_path = output_path.parent / "pairs_config.yaml"
        with open(config_path, "w") as f:
            yaml.dump(cfg.model_dump(), f, default_flow_style=False)
        logger.info("Saved config snapshot to %s", config_path)

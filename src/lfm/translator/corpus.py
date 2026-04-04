"""Generate romanized IPA corpus for self-supervised LLM pretraining.

Generates a large text corpus from the trained expression game by
running all embeddings through the diffusion z-generator + frozen
decoder, romanizing the output, and writing plain text.  Multiple
passes with different random seeds scale the corpus.

The LLM learns the alien language via next-token prediction on this
corpus, then translates via few-shot cross-lingual transfer.
"""

from __future__ import annotations

import logging
from pathlib import Path

import sentencepiece as spm_lib
import torch

from lfm.translator.config import CorpusConfig
from lfm.translator.romanize import romanize, syllable_hyphenate

# Corpus output modes
MODE_HYPHENATED_IPA = "hyphenated_ipa"  # default: syllable-hyphenated IPA
MODE_ROMANIZED = "romanized"  # legacy: romanized ASCII (lossy)
MODE_HYPHENATED_ROMANIZED = "hyphenated_romanized"  # romanized + syllable hyphens

logger = logging.getLogger(__name__)


def _format_expression(ipa: str, mode: str) -> str:
    """Format a single IPA expression according to the output mode."""
    if mode == MODE_HYPHENATED_IPA:
        return syllable_hyphenate(ipa)
    if mode == MODE_HYPHENATED_ROMANIZED:
        return romanize(syllable_hyphenate(ipa))
    if mode == MODE_ROMANIZED:
        return romanize(ipa)
    return syllable_hyphenate(ipa)  # default


class CorpusGenerator:
    """Generate romanized IPA corpus from a trained expression game.

    Args:
        config: Corpus generation configuration.
    """

    def __init__(self, config: CorpusConfig) -> None:
        self.config = config

    def generate(self) -> dict[str, float]:
        """Generate the full corpus.

        Returns:
            Dict with ``num_lines``, ``num_tokens``, ``unique_lines``.
        """
        cfg = self.config
        device = torch.device(cfg.device)

        # Load expression game
        from lfm.agents.games.expression import ExpressionGame, ExpressionGameConfig
        from lfm.faculty.model import LanguageFaculty

        ckpt = torch.load(cfg.expression_checkpoint, map_location=device, weights_only=False)
        game_cfg = ExpressionGameConfig(
            decoder_path=cfg.decoder_path,
            spm_path=cfg.spm_path,
            z_generator=ckpt.get("z_generator", "gru"),
            max_phrases=ckpt.get("max_phrases", ckpt.get("max_segments", 4)),
            embedding_dim=ckpt.get("embedding_dim", 384),
            z_hidden_dim=ckpt.get("z_hidden_dim", 512),
            num_memory_tokens=ckpt.get("num_memory_tokens", 8),
            diffusion_steps=ckpt.get("diffusion_steps", 4),
            diffusion_layers=ckpt.get("diffusion_layers", 4),
            use_halt=ckpt.get("use_halt", True),
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

        # Load embeddings
        from lfm.embeddings.store import EmbeddingStore
        import numpy as np

        store = EmbeddingStore(cfg.embedding_store_dir)
        store.load()
        embeddings = np.array(store._embeddings)
        n_emb = len(embeddings)

        logger.info(
            "Generating corpus: %d embeddings × %d passes = %d expressions",
            n_emb, cfg.num_passes, n_emb * cfg.num_passes,
        )

        # Generate
        output_path = Path(cfg.output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        total_lines = 0
        total_tokens = 0
        seen_hashes: set[int] = set()

        with open(output_path, "w", encoding="utf-8") as f:
            for pass_idx in range(cfg.num_passes):
                torch.manual_seed(cfg.seed + pass_idx)
                logger.info("Pass %d/%d ...", pass_idx + 1, cfg.num_passes)

                with torch.no_grad():
                    for start in range(0, n_emb, cfg.batch_size):
                        end = min(start + cfg.batch_size, n_emb)
                        emb = torch.tensor(
                            embeddings[start:end], dtype=torch.float32, device=device,
                        )

                        z_out = game.z_gen(emb)
                        z_seq, z_weights = z_out[0], z_out[-2]
                        tokens, mask, _ = game._multiphrase_decode(z_seq, z_weights)

                        for i in range(end - start):
                            ids = [
                                t.item() for t, m in zip(tokens[i], mask[i])
                                if m and t.item() != eos_id and t.item() < vocab_size
                            ]
                            ipa = sp.decode(ids).strip()
                            if not ipa:
                                continue

                            line = _format_expression(ipa, cfg.output_mode)
                            if not line:
                                continue

                            f.write(line + "\n")
                            total_lines += 1
                            total_tokens += len(line.split())
                            seen_hashes.add(hash(line))

                        if (start // cfg.batch_size) % 100 == 0:
                            pct = (pass_idx * n_emb + end) / (cfg.num_passes * n_emb) * 100
                            logger.info(
                                "  %d/%d (%.0f%%) — %d lines, %d tokens, %d unique",
                                pass_idx * n_emb + end,
                                cfg.num_passes * n_emb,
                                pct,
                                total_lines,
                                total_tokens,
                                len(seen_hashes),
                            )

        # Free VRAM
        del game, faculty
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        unique = len(seen_hashes)
        logger.info(
            "Corpus complete: %d lines, %d tokens, %d unique (%.1f%%) → %s",
            total_lines, total_tokens, unique,
            unique / max(total_lines, 1) * 100,
            output_path,
        )

        return {
            "num_lines": total_lines,
            "num_tokens": total_tokens,
            "unique_lines": unique,
        }

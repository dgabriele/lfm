"""Corpus generation for self-supervised LLM pretraining.

Shared base class for single-expression and multi-turn dialogue corpus
generators.  Both produce syllable-hyphenated IPA text (default) from
trained games by running embeddings through the generation pipeline.
"""

from __future__ import annotations

import abc
import logging
from pathlib import Path

import numpy as np
import sentencepiece as spm_lib
import torch
from torch import Tensor

from lfm.translator.romanize import romanize, syllable_hyphenate

# Corpus output modes
MODE_HYPHENATED_IPA = "hyphenated_ipa"
MODE_ROMANIZED = "romanized"
MODE_HYPHENATED_ROMANIZED = "hyphenated_romanized"
MODE_ROMANIZED_ISO = "romanized_iso"

logger = logging.getLogger(__name__)


def format_ipa(ipa: str, mode: str) -> str:
    """Format an IPA string according to the output mode.

    Args:
        ipa: Raw IPA text from sentencepiece decode.
        mode: One of ``"hyphenated_ipa"``, ``"romanized"``,
            ``"hyphenated_romanized"``.

    Returns:
        Formatted string (may be empty if input is blank).
    """
    if mode == MODE_HYPHENATED_IPA:
        return syllable_hyphenate(ipa)
    if mode == MODE_HYPHENATED_ROMANIZED:
        return romanize(syllable_hyphenate(ipa))
    if mode == MODE_ROMANIZED:
        return romanize(ipa)
    if mode == MODE_ROMANIZED_ISO:
        from lfm.translator.romanize import romanize_iso
        return romanize_iso(ipa)
    return syllable_hyphenate(ipa)


class BaseCorpusGenerator(abc.ABC):
    """Shared infrastructure for generating IPA corpora from trained games.

    Subclasses implement :meth:`generate` with game-specific logic.
    This base provides embedding loading, sentencepiece decoding,
    token-to-IPA conversion, and GPU cleanup.
    """

    def __init__(self, config) -> None:
        self.config = config

    def _load_embeddings(self) -> tuple[np.ndarray, int]:
        """Load embedding store.

        Returns:
            (embeddings_array, num_embeddings).
        """
        from lfm.embeddings.store import EmbeddingStore

        store = EmbeddingStore(self.config.embedding_store_dir)
        store.load()
        embeddings = np.array(store._embeddings)
        return embeddings, len(embeddings)

    def _load_spm(self) -> tuple[spm_lib.SentencePieceProcessor, int, int]:
        """Load sentencepiece model.

        Returns:
            (sp_processor, vocab_size, eos_id).
        """
        sp = spm_lib.SentencePieceProcessor(model_file=self.config.spm_path)
        vocab_size = sp.vocab_size()
        eos_id = vocab_size + 1
        return sp, vocab_size, eos_id

    def decode_tokens_to_ipa(
        self,
        tokens: Tensor,
        mask: Tensor,
        sp: spm_lib.SentencePieceProcessor,
        vocab_size: int,
        eos_id: int,
    ) -> list[str]:
        """Legacy SPM-direct decode path.  Kept for backward compat.

        For new code (and both v7/v8 VAEs uniformly), prefer
        :meth:`decode_tokens_to_surface`, which routes through the VAE's
        ``render_surface`` method and handles alphabet-specific
        formatting in the VAE class where it belongs.

        Args:
            tokens: ``(B, S)`` token IDs.
            mask: ``(B, S)`` boolean mask (True = valid).
            sp: Sentencepiece processor.
            vocab_size: SPM vocabulary size.
            eos_id: EOS token ID.

        Returns:
            List of ``B`` formatted IPA strings (empty string for blanks).
        """
        mode = self.config.output_mode
        results: list[str] = []
        for i in range(tokens.size(0)):
            ids = [
                t.item() for t, m in zip(tokens[i], mask[i])
                if m and t.item() != eos_id and t.item() < vocab_size
            ]
            ipa = sp.decode(ids).strip()
            results.append(format_ipa(ipa, mode) if ipa else "")
        return results

    def decode_tokens_to_surface(
        self,
        tokens: Tensor,
        mask: Tensor,
        game,
    ) -> list[str]:
        """VAE-agnostic decode: route through the game's VAE.

        Works identically for v7 (IPA/SPM) and v8 (phoneme Latin)
        decoders.  Alphabet-specific formatting happens inside
        ``game.gen.render_surface``; this method just passes through
        the corpus-level ``output_mode`` config.

        Args:
            tokens: ``(B, S)`` token IDs.
            mask: ``(B, S)`` boolean mask (True = valid).
            game: The agent game containing ``game.gen`` (a
                ``BaseVAEGenerator`` subclass).

        Returns:
            List of ``B`` surface strings, one per batch row.
        """
        mode = getattr(self.config, "output_mode", None)
        return game.gen.render_surface(
            tokens, mask=mask, output_mode=mode,
        )

    def _cleanup_gpu(self) -> None:
        """Free GPU memory after generation."""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    @abc.abstractmethod
    def generate(self) -> dict[str, float]:
        """Generate the full corpus. Returns stats dict."""


class ExpressionCorpusGenerator(BaseCorpusGenerator):
    """Generate single-expression IPA corpus from a trained expression game.

    Each line is one expression (one embedding → one multi-phrase decode).

    Args:
        config: ``CorpusConfig`` instance.
    """

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

        ckpt = torch.load(
            cfg.expression_checkpoint, map_location=device, weights_only=False,
        )
        # Prefer checkpoint; fall back to store metadata; 384 only as last resort.
        from lfm.embeddings.store import EmbeddingStore
        fallback_dim = 384
        try:
            meta = EmbeddingStore.read_metadata(cfg.embedding_store_dir)
            if "embedding_dim" in meta:
                fallback_dim = int(meta["embedding_dim"])
        except FileNotFoundError:
            pass

        game_cfg = ExpressionGameConfig(
            decoder_path=cfg.decoder_path,
            spm_path=cfg.spm_path,
            z_generator=ckpt.get("z_generator", "gru"),
            max_phrases=ckpt.get("max_phrases", ckpt.get("max_segments", 4)),
            embedding_dim=ckpt.get("embedding_dim", fallback_dim),
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

        sp, vocab_size, eos_id = self._load_spm()
        embeddings, n_emb = self._load_embeddings()

        logger.info(
            "Generating expression corpus: %d embeddings × %d passes",
            n_emb, cfg.num_passes,
        )

        output_path = Path(cfg.output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        total_lines = 0
        total_tokens = 0
        seen: set[int] = set()

        with open(output_path, "w", encoding="utf-8") as f:
            for pass_idx in range(cfg.num_passes):
                torch.manual_seed(cfg.seed + pass_idx)
                logger.info("Pass %d/%d ...", pass_idx + 1, cfg.num_passes)

                with torch.no_grad():
                    for start in range(0, n_emb, cfg.batch_size):
                        end = min(start + cfg.batch_size, n_emb)
                        emb = torch.tensor(
                            embeddings[start:end],
                            dtype=torch.float32, device=device,
                        )

                        z_out = game.z_gen(emb)
                        z_seq, z_weights = z_out[0], z_out[-2]
                        tokens, mask, _ = game._multiphrase_decode(
                            z_seq, z_weights,
                        )

                        # VAE-agnostic surface render (works for v7 IPA
                        # and v8 phoneme uniformly).
                        lines = self.decode_tokens_to_surface(
                            tokens, mask, game,
                        )
                        for line in lines:
                            if not line:
                                continue
                            f.write(line + "\n")
                            total_lines += 1
                            total_tokens += len(line.split())
                            seen.add(hash(line))

                        if (start // cfg.batch_size) % 100 == 0:
                            pct = (pass_idx * n_emb + end) / (
                                cfg.num_passes * n_emb
                            ) * 100
                            logger.info(
                                "  %d/%d (%.0f%%) — %d lines, %d unique",
                                pass_idx * n_emb + end,
                                cfg.num_passes * n_emb,
                                pct, total_lines, len(seen),
                            )

        del game, faculty
        self._cleanup_gpu()

        unique = len(seen)
        logger.info(
            "Corpus complete: %d lines, %d tokens, %d unique (%.1f%%) → %s",
            total_lines, total_tokens, unique,
            unique / max(total_lines, 1) * 100, output_path,
        )
        return {
            "num_lines": total_lines,
            "num_tokens": total_tokens,
            "unique_lines": unique,
        }


# Backward-compatible alias
CorpusGenerator = ExpressionCorpusGenerator

"""Generate multi-turn dialogue corpus for self-supervised LLM pretraining.

Runs embeddings through a trained DialogueGame's generation pipeline
(z-gen → Phase 1 decode → no-grad decoder rerun for context summaries)
and writes multi-turn documents with turn markers.

Each document is a monologue about one input embedding::

    [T0] sə-kav-kos sɛj-sjes-tin-kaaɛ
    [T1] pa-ria-vse jɪɹ-zŋa-koem-ʕla
    [T2] ma-la-tu-ni-thun-ta-i-thu-lun
    [T3] bu sur-tal ha-pha-net

Documents are separated by blank lines.  The LLM learns turn structure,
progressive elaboration, and discourse patterns from the turn ordering.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import sentencepiece as spm_lib
import torch
from torch import Tensor

from lfm.agents.decode import rerun_decoder_multiphrase_no_grad
from lfm.translator.config import DialogueCorpusConfig
from lfm.translator.corpus import BaseCorpusGenerator

logger = logging.getLogger(__name__)


class DialogueCorpusGenerator(BaseCorpusGenerator):
    """Generate multi-turn dialogue corpus from a trained dialogue game.

    Args:
        config: ``DialogueCorpusConfig`` instance.
    """

    def __init__(self, config: DialogueCorpusConfig) -> None:
        super().__init__(config)

    def _load_dialogue_game(self) -> tuple:
        """Load trained dialogue game from checkpoint.

        Returns:
            (game, device) tuple.
        """
        from lfm.agents.games.dialogue import DialogueGame, DialogueGameConfig
        from lfm.faculty.model import LanguageFaculty

        cfg = self.config
        device = torch.device(cfg.device)

        ckpt = torch.load(
            cfg.dialogue_checkpoint, map_location=device, weights_only=False,
        )

        # Auto-detect embedding_dim from the store so the game config
        # matches whatever latent space the store was built in.
        from lfm.embeddings.store import EmbeddingStore
        dim_kwargs: dict = {}
        try:
            meta = EmbeddingStore.read_metadata(cfg.embedding_store_dir)
            if "embedding_dim" in meta:
                dim_kwargs["embedding_dim"] = int(meta["embedding_dim"])
                logger.info(
                    "Auto-detected embedding_dim=%d from store at %s",
                    dim_kwargs["embedding_dim"], cfg.embedding_store_dir,
                )
        except FileNotFoundError:
            pass

        game_cfg = DialogueGameConfig(
            decoder_path=cfg.decoder_path,
            spm_path=cfg.spm_path,
            embedding_store_dir=cfg.embedding_store_dir,
            max_phrases=ckpt.get("max_phrases", ckpt.get("max_segments", 4)),
            num_turns=ckpt.get("num_turns", 4),
            device=cfg.device,
            contrastive_scoring=bool(ckpt.get("contrastive_scoring", False)),
            **dim_kwargs,
        )
        faculty = LanguageFaculty(game_cfg.build_faculty_config()).to(device)
        game = DialogueGame(game_cfg, faculty).to(device)
        game.load_checkpoint_state(ckpt)
        game.eval()

        logger.info(
            "Loaded dialogue game: %d turns, %d max phrases",
            game_cfg.num_turns, game_cfg.max_phrases,
        )
        return game, device

    def _generate_documents(
        self,
        game,
        embedding_batch: Tensor,
        sp: spm_lib.SentencePieceProcessor,
        vocab_size: int,
        eos_id: int,
    ) -> list[list[str]]:
        """Generate multi-turn documents for a batch of embeddings.

        Runs the dialogue game's generation pipeline (z-gen → Phase 1
        decode → no-grad decoder rerun → context summary) for each turn,
        accumulating context across turns.

        Args:
            game: Trained ``DialogueGame`` in eval mode.
            embedding_batch: ``(B, dim)`` input embeddings.
            sp: Sentencepiece processor.
            vocab_size: SPM vocabulary size.
            eos_id: EOS token ID.

        Returns:
            List of ``B`` documents, each a list of formatted turn
            strings with ``[TN]`` markers.
        """
        batch = embedding_batch.size(0)
        num_turns = game.config.num_turns

        # Targets for context transformer: (B, 1, dim)
        targets = embedding_batch.unsqueeze(1)

        context_summaries: list[Tensor] = []
        documents: list[list[str]] = [[] for _ in range(batch)]

        for turn_idx in range(num_turns):
            turn_emb = game.turn_embeddings[turn_idx]
            context = (
                torch.stack(context_summaries, dim=1)
                if context_summaries else None
            )

            # Conditioning
            conditioning = game.context_transformer(
                targets, turn_emb, context, target_mask=None,
            )

            # Z generation
            z_seq, z_weights, _ = game.z_gen(conditioning)

            # Phase 1: decode tokens (no grad)
            tokens, gen_mask, phrase_bounds = game.phrase_decoder.decode(
                z_seq, z_weights,
            )

            # No-grad decoder rerun for context summary
            hidden = rerun_decoder_multiphrase_no_grad(
                game.gen, z_seq, z_weights, tokens, gen_mask, phrase_bounds,
            )
            trimmed_mask = gen_mask[:, :hidden.size(1)]
            summary = game._summarize_turn(hidden, trimmed_mask)
            context_summaries.append(summary)

            # Decode tokens to IPA
            ipa_strings = self.decode_tokens_to_ipa(
                tokens, gen_mask, sp, vocab_size, eos_id,
            )

            for i, ipa in enumerate(ipa_strings):
                if ipa:
                    documents[i].append(f"[T{turn_idx}] {ipa}")

            # Free intermediates
            del hidden, tokens, gen_mask, phrase_bounds, z_seq, z_weights

        return documents

    def generate(self) -> dict[str, float]:
        """Generate the full dialogue corpus.

        Returns:
            Dict with ``num_documents``, ``num_turns``, ``total_tokens``,
            ``unique_documents``.
        """
        cfg = self.config

        game, device = self._load_dialogue_game()
        sp, vocab_size, eos_id = self._load_spm()
        embeddings, n_emb = self._load_embeddings()

        logger.info(
            "Generating dialogue corpus: %d embeddings × %d passes",
            n_emb, cfg.num_passes,
        )

        output_path = Path(cfg.output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        num_documents = 0
        num_turns = 0
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

                        documents = self._generate_documents(
                            game, emb, sp, vocab_size, eos_id,
                        )

                        for doc_turns in documents:
                            if not doc_turns:
                                continue

                            if cfg.paragraph_format:
                                # Natural paragraph: strip turn markers,
                                # capitalize, add periods, single line
                                sentences = []
                                for t in doc_turns:
                                    # Remove [TN] prefix if present
                                    text = t.split("] ", 1)[-1] if "] " in t else t
                                    text = text.strip()
                                    if text:
                                        text = text[0].upper() + text[1:]
                                        if not text.endswith("."):
                                            text += "."
                                        sentences.append(text)
                                doc_text = " ".join(sentences)
                            else:
                                tag = f"[{cfg.language_tag}]\n" if cfg.language_tag else ""
                                doc_text = tag + "\n".join(doc_turns)

                            f.write(doc_text + "\n"
                                    + ("" if cfg.paragraph_format else "\n"))

                            num_documents += 1
                            num_turns += len(doc_turns)
                            total_tokens += sum(
                                len(t.split()) for t in doc_turns
                            )
                            seen.add(hash(doc_text))

                        if (start // cfg.batch_size) % 100 == 0:
                            pct = (pass_idx * n_emb + end) / (
                                cfg.num_passes * n_emb
                            ) * 100
                            logger.info(
                                "  %d/%d (%.0f%%) — %d docs, %d turns, "
                                "%d unique",
                                pass_idx * n_emb + end,
                                cfg.num_passes * n_emb,
                                pct, num_documents, num_turns, len(seen),
                            )

        del game
        self._cleanup_gpu()

        unique = len(seen)
        logger.info(
            "Generated %d documents, %d turns, %d tokens, "
            "%d unique (%.1f%%) → %s",
            num_documents, num_turns, total_tokens, unique,
            unique / max(num_documents, 1) * 100, output_path,
        )

        # Tokenize to H5 if model specified
        if cfg.tokenize_model:
            from transformers import AutoTokenizer

            from lfm.translator.tokenized_dataset import TokenizedH5Dataset

            logger.info(
                "Tokenizing corpus to H5 (model=%s, max_len=%d)...",
                cfg.tokenize_model, cfg.tokenize_max_len,
            )
            tokenizer = AutoTokenizer.from_pretrained(cfg.tokenize_model)
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            TokenizedH5Dataset.from_corpus(
                output_path, tokenizer, cfg.tokenize_max_len,
            )

        return {
            "num_documents": num_documents,
            "num_turns": num_turns,
            "total_tokens": total_tokens,
            "unique_documents": unique,
        }

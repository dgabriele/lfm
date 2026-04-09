"""Generate bilingual (Xenoglot + English) corpus for cross-lingual bridging.

Each document is a single line containing three components:

1. **Cluster anchor** ``[C{id}]`` — a shared token that appears in both
   xenoglot and English contexts, forcing the LLM to build a bridging
   representation at this token position.

2. **Xenoglot dialogue turns** ``[T0] ... [T1] ...`` — multi-turn IPA
   expressions from the trained dialogue game, preserving progressive
   elaboration structure.

3. **English passage** ``[EN] ...`` — the source passage whose embedding
   produced the xenoglot turns.  Under the causal mask, the model must
   attend to the cluster anchor and xenoglot tokens when predicting
   English — forcing cross-lingual representation alignment.

Single-line format ensures both languages appear in the same context
window when tokenized by :class:`TokenizedH5Dataset`::

    [C1847] [T0] sə-kav-kos sɛj-sjes-tin-kaaɛ [T1] pa-ria-vse ... [EN] The village market...

This corpus is designed for Phase 3 of the progressive curriculum:

- **Phase 1**: Pure xenoglot (learn phonotactic/discourse structure)
- **Phase 2**: Batch-interleaved xenoglot + English (prevent forgetting)
- **Phase 3**: Bilingual documents with anchors (force bridging)
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import torch

from lfm.translator.config import BilingualCorpusConfig
from lfm.translator.dialogue_corpus import DialogueCorpusGenerator

logger = logging.getLogger(__name__)


class BilingualCorpusGenerator(DialogueCorpusGenerator):
    """Generate cluster-anchored bilingual documents from a trained dialogue game.

    Extends :class:`DialogueCorpusGenerator` with:

    - **Cluster anchors**: ``[C{id}]`` tokens from the embedding store's
      2048-cluster partition, shared across xenoglot and English contexts.
    - **English passages**: Source text from ``passages.jsonl``, paired
      with the xenoglot expression by embedding index.
    - **Single-line format**: All turns + English on one line so the
      tokenizer places them in the same context window.

    Args:
        config: :class:`BilingualCorpusConfig` instance.
    """

    def __init__(self, config: BilingualCorpusConfig) -> None:
        super().__init__(config)

    def _load_passages_and_clusters(
        self,
    ) -> tuple[np.ndarray, list[str]]:
        """Load passage texts and cluster labels from the embedding store.

        Returns:
            (cluster_labels, passage_texts) where cluster_labels is
            shape ``(N,)`` int32 and passage_texts is a list of N strings.
        """
        from lfm.embeddings.store import EmbeddingStore

        store = EmbeddingStore(self.config.embedding_store_dir)
        store.load()

        cluster_labels = np.array(store._cluster_labels)

        passages_path = store.store_dir / "passages.jsonl"
        texts: list[str] = []
        if passages_path.is_file():
            with passages_path.open("r", encoding="utf-8") as fh:
                for line in fh:
                    texts.append(json.loads(line).get("text", ""))
            logger.info(
                "Loaded %d passages + %d cluster labels from %s",
                len(texts), len(cluster_labels), self.config.embedding_store_dir,
            )
        else:
            logger.warning(
                "No passages.jsonl found in %s — English portions will be omitted",
                self.config.embedding_store_dir,
            )

        return cluster_labels, texts

    def _format_bilingual_line(
        self,
        cluster_id: int,
        turns: list[str],
        english: str,
    ) -> str:
        """Format a single bilingual document line.

        Args:
            cluster_id: Semantic cluster ID for the anchor token.
            turns: List of ``"[TN] ipa-text"`` strings.
            english: Source English passage (may be truncated).

        Returns:
            Single-line document string.
        """
        parts = [f"[C{cluster_id}]"]
        parts.extend(turns)
        if english:
            max_chars = self.config.max_english_chars
            truncated = english[:max_chars].strip()
            parts.append(f"[EN] {truncated}")
        return " ".join(parts)

    def generate(self) -> dict[str, float]:
        """Generate the full bilingual corpus.

        Returns:
            Dict with ``num_documents``, ``num_turns``, ``total_tokens``,
            ``unique_documents``, ``num_with_english``.
        """
        cfg = self.config

        game, device = self._load_dialogue_game()
        sp, vocab_size, eos_id = self._load_spm()
        embeddings, n_emb = self._load_embeddings()
        cluster_labels, passages = self._load_passages_and_clusters()

        has_passages = len(passages) >= n_emb

        logger.info(
            "Generating bilingual corpus: %d embeddings × %d passes, "
            "passages=%s, clusters=%d",
            n_emb, cfg.num_passes,
            "yes" if has_passages else "no",
            len(np.unique(cluster_labels)) if len(cluster_labels) > 0 else 0,
        )

        output_path = Path(cfg.output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        num_documents = 0
        num_turns = 0
        total_tokens = 0
        num_with_english = 0
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

                        # Generate xenoglot turns via dialogue game
                        documents = self._generate_documents(
                            game, emb, sp, vocab_size, eos_id,
                        )

                        for i, doc_turns in enumerate(documents):
                            if not doc_turns:
                                continue

                            emb_idx = start + i
                            cluster_id = int(cluster_labels[emb_idx]) if emb_idx < len(cluster_labels) else 0
                            english = passages[emb_idx] if has_passages and emb_idx < len(passages) else ""

                            line = self._format_bilingual_line(
                                cluster_id, doc_turns, english,
                            )
                            f.write(line + "\n")

                            num_documents += 1
                            num_turns += len(doc_turns)
                            total_tokens += len(line.split())
                            if english:
                                num_with_english += 1
                            seen.add(hash(line))

                        if (start // cfg.batch_size) % 100 == 0:
                            pct = (pass_idx * n_emb + end) / (
                                cfg.num_passes * n_emb
                            ) * 100
                            logger.info(
                                "  %d/%d (%.0f%%) — %d docs, %d unique, "
                                "%d with English",
                                pass_idx * n_emb + end,
                                cfg.num_passes * n_emb,
                                pct, num_documents, len(seen),
                                num_with_english,
                            )

        del game
        self._cleanup_gpu()

        unique = len(seen)
        logger.info(
            "Generated %d documents (%d unique, %.1f%%), "
            "%d turns, %d tokens, %d with English → %s",
            num_documents, unique,
            unique / max(num_documents, 1) * 100,
            num_turns, total_tokens, num_with_english,
            output_path,
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
            "num_with_english": num_with_english,
        }

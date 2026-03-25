"""Generate (IPA, English) parallel pairs via the frozen faculty pipeline.

Sequential VRAM pipeline: load sentence-transformer -> encode -> free ->
load faculty -> generate IPA -> free.  Saves pairs as JSONL for downstream
training experiments.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import torch
import yaml

from lfm.translator.config import PairGenerationConfig

logger = logging.getLogger(__name__)


class PairGenerator:
    """Generate (IPA, English) pairs from the faculty pipeline.

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

        # 1. Load English sentences
        texts = self._load_texts()
        if not texts:
            raise RuntimeError("No texts loaded from corpus")

        # 2. Encode texts -> embeddings (loads sentence-transformer)
        embeddings = self._encode_texts(texts)

        # 3. Generate IPA via faculty (loads decoder)
        pairs = self._generate_ipa(texts, embeddings, device)

        # 4. Save
        self._save(pairs)

        return pairs

    def _load_texts(self) -> list[str]:
        """Load English sentences from Leipzig corpus."""
        from lfm.data.loaders.leipzig import LeipzigCorpusConfig, LeipzigCorpusLoader

        cfg = self.config
        logger.info("Loading sentences from %s...", cfg.leipzig_dir)

        loader = LeipzigCorpusLoader(
            LeipzigCorpusConfig(
                data_dir=cfg.leipzig_dir,
                languages=cfg.languages,
                max_samples_per_language=cfg.max_sentences,
                min_line_length=cfg.min_line_length,
            )
        )
        samples = loader.load()
        texts = [text for _, text in samples]
        logger.info("Loaded %d sentences", len(texts))
        return texts

    def _encode_texts(self, texts: list[str]) -> np.ndarray:
        """Encode texts with sentence-transformer, then free VRAM."""
        from sentence_transformers import SentenceTransformer

        cfg = self.config
        logger.info("Loading encoder: %s", cfg.encoder_model)
        encoder = SentenceTransformer(cfg.encoder_model, device=cfg.device)

        logger.info("Encoding %d texts...", len(texts))
        embeddings = encoder.encode(
            texts,
            batch_size=cfg.encode_batch_size,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        embeddings = embeddings.astype(np.float32)

        # Free encoder VRAM
        del encoder
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        logger.info("Encoder freed from VRAM")

        return embeddings

    def _generate_ipa(
        self,
        texts: list[str],
        embeddings: np.ndarray,
        device: torch.device,
    ) -> list[tuple[str, str]]:
        """Load faculty, generate IPA messages, then free VRAM."""
        from lfm.faculty.config import FacultyConfig
        from lfm.faculty.model import LanguageFaculty
        from lfm.generator.config import GeneratorConfig

        cfg = self.config
        logger.info("Loading faculty (decoder: %s)", cfg.decoder_path)

        faculty_config = FacultyConfig(
            dim=384,
            generator=GeneratorConfig(
                pretrained_decoder_path=cfg.decoder_path,
                spm_model_path=cfg.spm_path,
                freeze_decoder=True,
                max_output_len=cfg.max_output_len,
            ),
        )
        faculty = LanguageFaculty(faculty_config).to(device)
        faculty.generator.eval()

        # Trigger lazy init
        with torch.no_grad():
            dummy = torch.randn(1, 384, device=device)
            faculty(dummy)

        # Generate pairs in batches
        pairs: list[tuple[str, str]] = []
        batch_size = cfg.encode_batch_size
        n = len(texts)

        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            batch = torch.tensor(
                embeddings[start:end], dtype=torch.float32, device=device,
            )

            with torch.no_grad():
                out = faculty(batch)

            tokens = out["generator.tokens"]
            ipa_texts = faculty.generator.decode_to_text(tokens)

            for i in range(end - start):
                ipa = ipa_texts[i].strip()
                if ipa:
                    pairs.append((ipa, texts[start + i]))

            if (start // batch_size) % 10 == 0:
                logger.info("Generated %d / %d pairs", len(pairs), n)

        # Free faculty VRAM
        del faculty
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        logger.info("Faculty freed from VRAM")

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

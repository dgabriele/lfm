"""Evaluate trained IPA -> English translator with BLEU and semantic similarity."""

from __future__ import annotations

import json
import logging
import math
from collections import Counter
from pathlib import Path

import numpy as np
import torch

logger = logging.getLogger(__name__)


def compute_bleu(references: list[str], hypotheses: list[str]) -> dict[str, float]:
    """Compute corpus-level BLEU scores (1-4 gram).

    Returns:
        Dict with ``bleu_1`` through ``bleu_4`` (individual, with BP)
        and ``bleu_4_geometric`` (standard BLEU-4).
    """

    def _ngrams(tokens: list[str], n: int) -> Counter:
        return Counter(tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1))

    total_matches = [0] * 4
    total_counts = [0] * 4
    total_ref_len = 0
    total_hyp_len = 0

    for ref, hyp in zip(references, hypotheses):
        ref_tokens = ref.lower().split()
        hyp_tokens = hyp.lower().split()
        total_ref_len += len(ref_tokens)
        total_hyp_len += len(hyp_tokens)

        for n in range(1, 5):
            ref_ngrams = _ngrams(ref_tokens, n)
            hyp_ngrams = _ngrams(hyp_tokens, n)
            for ng, count in hyp_ngrams.items():
                total_matches[n - 1] += min(count, ref_ngrams.get(ng, 0))
            total_counts[n - 1] += max(len(hyp_tokens) - n + 1, 0)

    if total_hyp_len == 0:
        return {
            "bleu_1": 0.0, "bleu_2": 0.0,
            "bleu_3": 0.0, "bleu_4": 0.0,
            "bleu_4_geometric": 0.0,
        }

    bp = min(1.0, math.exp(1 - total_ref_len / total_hyp_len))

    results = {}
    log_avg = 0.0
    for n in range(4):
        if total_counts[n] == 0 or total_matches[n] == 0:
            precision = 0.0
        else:
            precision = total_matches[n] / total_counts[n]
        results[f"bleu_{n + 1}"] = bp * precision
        if precision > 0:
            log_avg += math.log(precision) / 4

    if all(total_matches[i] > 0 for i in range(4)):
        results["bleu_4_geometric"] = bp * math.exp(log_avg)
    else:
        results["bleu_4_geometric"] = 0.0

    return results


class TranslatorEvaluator:
    """Evaluate a trained IPA -> English translator.

    Loads a saved model, generates translations, then computes:
    - BLEU-1/2/3/4 (corpus-level)
    - Semantic cosine similarity via sentence-transformer re-embedding

    Args:
        model_dir: Directory containing the trained model.
        device: Torch device string.
    """

    def __init__(self, model_dir: str, device: str = "cuda") -> None:
        self.model_dir = Path(model_dir)
        self.device = device

    def evaluate(
        self,
        pairs_path: str | None = None,
        max_samples: int = 200,
        max_new_tokens: int = 64,
        temperature: float = 0.7,
    ) -> dict[str, float]:
        """Run evaluation on held-out pairs.

        Args:
            pairs_path: Override path to JSONL pairs (else reads from config).
            max_samples: Maximum evaluation samples.
            max_new_tokens: Max new tokens per generation.
            temperature: Sampling temperature.

        Returns:
            Dict of evaluation metrics.
        """
        from transformers import AutoModelForCausalLM, AutoTokenizer

        device = torch.device(self.device)

        # Resolve pairs path
        if pairs_path is None:
            config_path = self.model_dir / "config.yaml"
            if config_path.exists():
                import yaml
                with open(config_path) as f:
                    cfg = yaml.safe_load(f)
                pairs_path = cfg.get("pairs_path", "data/models/v1/translator/pairs.jsonl")
            else:
                pairs_path = "data/models/v1/translator/pairs.jsonl"

        # Load pairs and take val split
        pairs = self._load_pairs(pairs_path)
        rng = np.random.default_rng(42)
        n_val = max(1, int(len(pairs) * 0.1))
        perm = rng.permutation(len(pairs))
        val_pairs = [pairs[i] for i in perm[:n_val]][:max_samples]

        # Load model
        model_path = self.model_dir / "model"
        logger.info("Loading model from %s", model_path)
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            trust_remote_code=True,
        ).to(device)
        model.eval()

        # Generate translations
        logger.info("Generating translations for %d samples...", len(val_pairs))
        references = []
        hypotheses = []

        for ipa, eng in val_pairs:
            prompt = f"<ipa> {ipa} </ipa> <eng>"
            inputs = tokenizer(prompt, return_tensors="pt").to(device)

            with torch.no_grad():
                generated = model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    do_sample=True,
                    top_p=0.9,
                    pad_token_id=tokenizer.pad_token_id,
                )

            output_text = tokenizer.decode(
                generated[0][inputs["input_ids"].shape[1]:],
                skip_special_tokens=False,
            )

            if "</eng>" in output_text:
                output_text = output_text.split("</eng>")[0]
            output_text = output_text.strip()

            references.append(eng)
            hypotheses.append(output_text)

        # Free model VRAM
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # BLEU scores
        results = compute_bleu(references, hypotheses)
        logger.info("BLEU: %s", {k: f"{v:.4f}" for k, v in results.items()})

        # Semantic similarity
        cosine_sims = self._compute_semantic_similarity(references, hypotheses)
        if cosine_sims is not None:
            results["semantic_cosine_mean"] = float(cosine_sims.mean())
            results["semantic_cosine_std"] = float(cosine_sims.std())
            results["semantic_cosine_median"] = float(np.median(cosine_sims))
            logger.info(
                "Semantic: mean=%.4f median=%.4f",
                cosine_sims.mean(), np.median(cosine_sims),
            )

        # Show examples
        logger.info("\n=== Translation Examples ===")
        for i in range(min(10, len(references))):
            logger.info("  IPA:  %s", val_pairs[i][0][:80])
            logger.info("  REF:  %s", references[i][:80])
            logger.info("  HYP:  %s", hypotheses[i][:80])
            logger.info("")

        # Save results + translations
        with open(self.model_dir / "results.json", "w") as f:
            json.dump(results, f, indent=2)

        with open(self.model_dir / "translations.jsonl", "w") as f:
            for i in range(len(references)):
                record = {
                    "ipa": val_pairs[i][0],
                    "reference": references[i],
                    "hypothesis": hypotheses[i],
                }
                if cosine_sims is not None:
                    record["cosine_similarity"] = float(cosine_sims[i])
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

        logger.info("Saved results to %s", self.model_dir)
        return results

    def _compute_semantic_similarity(
        self, references: list[str], hypotheses: list[str],
    ) -> np.ndarray | None:
        """Compute per-example cosine similarity via sentence-transformer."""
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            logger.warning("sentence-transformers not available for semantic eval")
            return None

        encoder = SentenceTransformer("all-MiniLM-L6-v2", device=self.device)
        ref_emb = encoder.encode(
            references, convert_to_numpy=True, normalize_embeddings=True,
        )
        hyp_emb = encoder.encode(
            hypotheses, convert_to_numpy=True, normalize_embeddings=True,
        )

        cosine_sims = np.array([
            float(np.dot(ref_emb[i], hyp_emb[i]))
            for i in range(len(references))
        ])

        del encoder
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return cosine_sims

    @staticmethod
    def _load_pairs(path: str) -> list[tuple[str, str]]:
        """Load pairs from JSONL."""
        pairs = []
        with open(path, encoding="utf-8") as f:
            for line in f:
                record = json.loads(line)
                pairs.append((record["ipa"], record["english"]))
        return pairs

"""LLM quality gatekeeper for dataset generation.

Uses a small causal LM (e.g. Qwen2.5-0.5B) as a prompted validator that
operates on sanitized raw text (before IPA conversion).  Each sample is
classified as accept / fix / reject.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass

from lfm.data.dataset.config import LLMGateConfig

logger = logging.getLogger(__name__)

_PROMPT_TEMPLATE = """\
You are a multilingual text quality validator for a linguistic dataset.
Evaluate this sentence for quality: is it a well-formed, complete,
natural sentence suitable for phonetic transcription?

Language: {lang}
Sentence: {text}

Respond with JSON only:
{{"verdict": "accept"|"fix"|"reject", \
"text": "<corrected if fix, else original>", \
"reason": "<brief explanation if fix or reject>"}}"""


@dataclass
class GateResult:
    """Result of LLM gatekeeper evaluation."""

    verdict: str    # "accept", "fix", "reject"
    text: str       # Original or corrected text
    reason: str     # Explanation (empty for accept)


class LLMGatekeeper:
    """Quality gate using a small causal LM.

    Evaluates sanitized text samples in batches and returns accept/fix/reject
    verdicts. Fixed samples use the LLM's corrected text for downstream
    IPA conversion.

    The LLM is loaded lazily on first call to ``evaluate()`` and freed
    via ``unload()``.
    """

    def __init__(self, config: LLMGateConfig) -> None:
        self.config = config
        self._model = None
        self._tokenizer = None

    def _load_model(self) -> None:
        """Lazily load the model and tokenizer."""
        if self._model is not None:
            return

        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        logger.info("Loading LLM gatekeeper: %s", self.config.model_name)
        self._tokenizer = AutoTokenizer.from_pretrained(
            self.config.model_name, trust_remote_code=True,
        )
        if self._tokenizer.pad_token is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token

        self._model = AutoModelForCausalLM.from_pretrained(
            self.config.model_name,
            torch_dtype=torch.float16,
            trust_remote_code=True,
        ).to(self.config.device)
        self._model.eval()
        logger.info("LLM gatekeeper loaded on %s", self.config.device)

    def unload(self) -> None:
        """Free model VRAM."""
        if self._model is not None:
            import gc

            import torch

            del self._model
            del self._tokenizer
            self._model = None
            self._tokenizer = None
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            logger.info("LLM gatekeeper unloaded")

    def evaluate(
        self, samples: list[tuple[str, str]],
    ) -> list[GateResult]:
        """Evaluate a batch of (language, sanitized_text) samples.

        Args:
            samples: List of ``(lang, text)`` tuples (sanitized raw text).

        Returns:
            List of ``GateResult`` — one per input sample.
        """
        import torch

        self._load_model()
        assert self._model is not None
        assert self._tokenizer is not None

        results: list[GateResult] = []
        cfg = self.config

        for batch_start in range(0, len(samples), cfg.batch_size):
            batch = samples[batch_start : batch_start + cfg.batch_size]
            prompts = [
                _PROMPT_TEMPLATE.format(lang=lang, text=text)
                for lang, text in batch
            ]

            inputs = self._tokenizer(
                prompts, return_tensors="pt", padding=True, truncation=True,
                max_length=512,
            ).to(cfg.device)

            with torch.no_grad():
                outputs = self._model.generate(
                    **inputs,
                    max_new_tokens=cfg.max_new_tokens,
                    temperature=cfg.temperature,
                    do_sample=cfg.temperature > 0,
                    pad_token_id=self._tokenizer.pad_token_id,
                )

            # Decode only the generated portion
            for i, (lang, text) in enumerate(batch):
                input_len = inputs["input_ids"].shape[1]
                generated = outputs[i][input_len:]
                response = self._tokenizer.decode(generated, skip_special_tokens=True).strip()

                result = self._parse_response(response, text)
                results.append(result)

            if batch_start > 0 and batch_start % (cfg.batch_size * 10) == 0:
                logger.info(
                    "LLM gate progress: %d / %d samples",
                    batch_start, len(samples),
                )

        return results

    @staticmethod
    def _parse_response(response: str, original_text: str) -> GateResult:
        """Parse LLM JSON response into a GateResult."""
        try:
            # Try to extract JSON from the response
            # The LLM might wrap it in markdown code blocks
            text = response.strip()
            if text.startswith("```"):
                text = text.split("```")[1]
                if text.startswith("json"):
                    text = text[4:]
            data = json.loads(text)
            verdict = data.get("verdict", "reject").lower()
            if verdict not in ("accept", "fix", "reject"):
                verdict = "reject"
            return GateResult(
                verdict=verdict,
                text=data.get("text", original_text) if verdict == "fix" else original_text,
                reason=data.get("reason", ""),
            )
        except (json.JSONDecodeError, KeyError, AttributeError):
            # If we can't parse, conservatively accept (don't lose data)
            return GateResult(verdict="accept", text=original_text, reason="")

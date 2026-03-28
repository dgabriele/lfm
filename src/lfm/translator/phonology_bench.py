"""Phonology benchmark for evaluating LLM understanding of the emergent language.

Adapted from PhonologyBench (Suvarna et al. 2024) for LFM's emergent IPA.
Tests whether the translator LLM has acquired genuine phonological
competence in the alien language — not just memorized (IPA, English) pairs.

Three tasks:

1. **Syllable counting**: Given alien IPA, count syllables.
   Tests: phonological parsing, vowel/consonant distinction.

2. **Rhyme detection**: Do two alien utterances rhyme?
   Tests: phonotactic understanding, final-syllable matching.

3. **Minimal pair discrimination**: Which utterance encodes a different
   meaning from the other two?
   Tests: phonological-semantic mapping, sensitivity to small IPA changes.

Usage::

    from lfm.translator.phonology_bench import PhonologyBench

    bench = PhonologyBench(
        translator_model_dir="data/models/v4-phase1/translator",
        faculty_decoder_path="data/models/v4-phase1/vae_decoder.pt",
        spm_path="data/models/v4-phase1/spm.model",
    )
    results = bench.run_all()
    bench.print_report(results)

CLI::

    lfm translate eval-phonology \
        --model-dir data/models/v4-phase1/translator \
        --decoder-path data/models/v4-phase1/vae_decoder.pt
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class TaskResult:
    """Result of a single phonology bench task."""

    task_name: str
    accuracy: float
    total: int
    correct: int
    examples: list[dict] = field(default_factory=list)


@dataclass
class BenchResults:
    """Aggregate results from all phonology bench tasks."""

    syllable_counting: TaskResult
    rhyme_detection: TaskResult
    minimal_pairs: TaskResult

    @property
    def mean_accuracy(self) -> float:
        tasks = [self.syllable_counting, self.rhyme_detection, self.minimal_pairs]
        return sum(t.accuracy for t in tasks) / len(tasks)

    def to_dict(self) -> dict:
        return {
            "mean_accuracy": self.mean_accuracy,
            "syllable_counting": {
                "accuracy": self.syllable_counting.accuracy,
                "total": self.syllable_counting.total,
                "correct": self.syllable_counting.correct,
            },
            "rhyme_detection": {
                "accuracy": self.rhyme_detection.accuracy,
                "total": self.rhyme_detection.total,
                "correct": self.rhyme_detection.correct,
            },
            "minimal_pairs": {
                "accuracy": self.minimal_pairs.accuracy,
                "total": self.minimal_pairs.total,
                "correct": self.minimal_pairs.correct,
            },
        }


class PhonologyBench:
    """Evaluate translator LLM's phonological competence in the emergent language.

    Generates test data from the frozen decoder (z perturbations,
    interpolations, random samples) and prompts the translator LLM
    to answer phonological questions about the IPA output.

    Args:
        translator_model_dir: Path to the trained translator model.
        faculty_decoder_path: Path to the frozen decoder checkpoint.
        spm_path: Path to the sentencepiece model.
        num_samples: Number of test samples per task.
        device: Torch device for inference.
    """

    def __init__(
        self,
        translator_model_dir: str | Path,
        faculty_decoder_path: str | Path,
        spm_path: str | Path,
        num_samples: int = 200,
        device: str = "cuda",
    ) -> None:
        self.translator_dir = Path(translator_model_dir)
        self.decoder_path = Path(faculty_decoder_path)
        self.spm_path = Path(spm_path)
        self.num_samples = num_samples
        self.device = device
        self._translator = None
        self._tokenizer = None

    def _load_translator(self) -> None:
        """Lazily load the translator LLM."""
        if self._translator is not None:
            return

        from transformers import AutoModelForCausalLM, AutoTokenizer

        model_path = self.translator_dir / "model"
        self._tokenizer = AutoTokenizer.from_pretrained(model_path)
        self._translator = AutoModelForCausalLM.from_pretrained(
            model_path, torch_dtype="auto",
        ).to(self.device)
        self._translator.eval()
        logger.info("Loaded translator from %s", model_path)

    def _generate_ipa_samples(self, n: int) -> list[str]:
        """Generate diverse IPA samples from the frozen decoder.

        Uses random z sampling from the encoder's tracked distribution.
        """
        import torch

        from lfm.faculty.config import FacultyConfig
        from lfm.faculty.model import LanguageFaculty
        from lfm.generator.config import GeneratorConfig

        gen_config = GeneratorConfig(
            pretrained_decoder_path=str(self.decoder_path),
            spm_model_path=str(self.spm_path),
            freeze_decoder=True,
        )
        faculty = LanguageFaculty(FacultyConfig(generator=gen_config))
        faculty.to(self.device)
        faculty.eval()

        # Trigger lazy init
        with torch.no_grad():
            faculty(torch.randn(1, gen_config.latent_dim).to(self.device))

        gen = faculty.generator
        z_mean = gen._z_mean.to(self.device)
        z_std = gen._z_std.to(self.device)

        # Sample random z from the encoder distribution
        z = torch.randn(n, gen_config.latent_dim, device=self.device)
        z = z * z_std + z_mean

        with torch.no_grad():
            token_ids, _, _, lengths, _ = gen._decode(z)

        import sentencepiece as spm

        sp = spm.SentencePieceProcessor()
        sp.load(str(self.spm_path))

        samples = []
        for i in range(n):
            ids = token_ids[i, : lengths[i]].cpu().tolist()
            text = sp.decode(ids)
            if text.strip():
                samples.append(text.strip())

        # Cleanup
        del faculty, gen
        torch.cuda.empty_cache()

        return samples

    def _query_translator(self, prompt: str) -> str:
        """Query the translator LLM with a prompt."""
        self._load_translator()

        inputs = self._tokenizer(prompt, return_tensors="pt").to(self.device)
        import torch

        with torch.no_grad():
            outputs = self._translator.generate(
                **inputs,
                max_new_tokens=64,
                temperature=0.1,
                do_sample=False,
            )
        response = self._tokenizer.decode(
            outputs[0][inputs.input_ids.shape[1] :],
            skip_special_tokens=True,
        )
        return response.strip()

    def eval_syllable_counting(self) -> TaskResult:
        """Test: given alien IPA, count syllables."""
        from lfm.data.syllabify import syllabify_ipa

        samples = self._generate_ipa_samples(self.num_samples)
        correct = 0
        total = 0
        examples = []

        for ipa in samples:
            syllables = [s for s in syllabify_ipa(ipa) if s.strip()]
            ground_truth = len(syllables)
            if ground_truth == 0:
                continue

            prompt = (
                f"Count the number of syllables in this phonetic transcription. "
                f"Reply with only the number.\n\n"
                f"Transcription: {ipa}\n\n"
                f"Number of syllables:"
            )

            response = self._query_translator(prompt)
            try:
                predicted = int("".join(c for c in response if c.isdigit())[:3])
            except (ValueError, IndexError):
                predicted = -1

            is_correct = predicted == ground_truth
            if is_correct:
                correct += 1
            total += 1

            if len(examples) < 20:
                examples.append({
                    "ipa": ipa,
                    "ground_truth": ground_truth,
                    "predicted": predicted,
                    "correct": is_correct,
                })

        return TaskResult(
            task_name="syllable_counting",
            accuracy=correct / max(total, 1),
            total=total,
            correct=correct,
            examples=examples,
        )

    def eval_rhyme_detection(self) -> TaskResult:
        """Test: do two alien utterances rhyme (share final syllable)?"""
        from lfm.data.syllabify import syllabify_ipa

        samples = self._generate_ipa_samples(self.num_samples * 2)
        correct = 0
        total = 0
        examples = []

        import random

        rng = random.Random(42)

        for i in range(0, len(samples) - 1, 2):
            ipa_a = samples[i]
            ipa_b = samples[i + 1]

            syls_a = [s for s in syllabify_ipa(ipa_a) if s.strip()]
            syls_b = [s for s in syllabify_ipa(ipa_b) if s.strip()]
            if not syls_a or not syls_b:
                continue

            # Ground truth: do they share the same final syllable?
            ground_truth = syls_a[-1] == syls_b[-1]

            prompt = (
                f"Do these two phonetic transcriptions rhyme "
                f"(share the same ending sound)? "
                f"Reply with only 'yes' or 'no'.\n\n"
                f"A: {ipa_a}\n"
                f"B: {ipa_b}\n\n"
                f"Rhyme:"
            )

            response = self._query_translator(prompt).lower()
            predicted = "yes" in response

            is_correct = predicted == ground_truth
            if is_correct:
                correct += 1
            total += 1

            if len(examples) < 20:
                examples.append({
                    "ipa_a": ipa_a,
                    "ipa_b": ipa_b,
                    "ground_truth": ground_truth,
                    "predicted": predicted,
                    "correct": is_correct,
                })

        return TaskResult(
            task_name="rhyme_detection",
            accuracy=correct / max(total, 1),
            total=total,
            correct=correct,
            examples=examples,
        )

    def eval_minimal_pairs(self) -> TaskResult:
        """Test: which of three utterances encodes a different meaning?

        Generates triplets where two utterances come from nearby z vectors
        (similar meaning) and one from a distant z (different meaning).
        The translator must identify the odd one out.
        """
        import torch

        from lfm.faculty.config import FacultyConfig
        from lfm.faculty.model import LanguageFaculty
        from lfm.generator.config import GeneratorConfig

        gen_config = GeneratorConfig(
            pretrained_decoder_path=str(self.decoder_path),
            spm_model_path=str(self.spm_path),
            freeze_decoder=True,
        )
        faculty = LanguageFaculty(FacultyConfig(generator=gen_config))
        faculty.to(self.device)
        faculty.eval()

        with torch.no_grad():
            faculty(torch.randn(1, gen_config.latent_dim).to(self.device))

        gen = faculty.generator
        z_mean = gen._z_mean.to(self.device)
        z_std = gen._z_std.to(self.device)

        import sentencepiece as spm

        sp = spm.SentencePieceProcessor()
        sp.load(str(self.spm_path))

        import random

        rng = random.Random(42)

        correct = 0
        total = 0
        examples = []
        n = self.num_samples

        for _ in range(n):
            # Generate anchor z
            z_anchor = torch.randn(1, gen_config.latent_dim, device=self.device)
            z_anchor = z_anchor * z_std + z_mean

            # Similar z (small perturbation)
            z_similar = z_anchor + 0.1 * z_std * torch.randn_like(z_anchor)

            # Distant z (independent sample)
            z_distant = torch.randn(1, gen_config.latent_dim, device=self.device)
            z_distant = z_distant * z_std + z_mean

            z_batch = torch.cat([z_anchor, z_similar, z_distant], dim=0)
            with torch.no_grad():
                token_ids, _, _, lengths, _ = gen._decode(z_batch)

            ipas = []
            for i in range(3):
                ids = token_ids[i, : lengths[i]].cpu().tolist()
                ipas.append(sp.decode(ids).strip())

            if not all(ipas):
                continue

            # Shuffle and track which is the odd one out
            labels = ["A", "B", "C"]
            items = list(zip(labels, ipas, [0, 0, 1]))  # 0=similar, 1=distant
            rng.shuffle(items)
            odd_label = next(label for label, _, flag in items if flag == 1)

            prompt = (
                f"Three phonetic transcriptions are given. Two encode similar "
                f"meanings and one is different. Which one is the odd one out? "
                f"Reply with only the letter (A, B, or C).\n\n"
            )
            for label, ipa, _ in items:
                prompt += f"{label}: {ipa}\n"
            prompt += "\nOdd one out:"

            response = self._query_translator(prompt).strip().upper()
            predicted = response[0] if response and response[0] in "ABC" else "?"

            is_correct = predicted == odd_label
            if is_correct:
                correct += 1
            total += 1

            if len(examples) < 20:
                examples.append({
                    "items": [(l, ipa) for l, ipa, _ in items],
                    "ground_truth": odd_label,
                    "predicted": predicted,
                    "correct": is_correct,
                })

        # Cleanup
        del faculty, gen
        torch.cuda.empty_cache()

        return TaskResult(
            task_name="minimal_pairs",
            accuracy=correct / max(total, 1),
            total=total,
            correct=correct,
            examples=examples,
        )

    def run_all(self) -> BenchResults:
        """Run all three phonology benchmark tasks."""
        logger.info("Running phonology benchmark (%d samples per task)", self.num_samples)

        syl = self.eval_syllable_counting()
        logger.info("Syllable counting: %.1f%% (%d/%d)", syl.accuracy * 100, syl.correct, syl.total)

        rhyme = self.eval_rhyme_detection()
        logger.info("Rhyme detection: %.1f%% (%d/%d)", rhyme.accuracy * 100, rhyme.correct, rhyme.total)

        minimal = self.eval_minimal_pairs()
        logger.info("Minimal pairs: %.1f%% (%d/%d)", minimal.accuracy * 100, minimal.correct, minimal.total)

        return BenchResults(
            syllable_counting=syl,
            rhyme_detection=rhyme,
            minimal_pairs=minimal,
        )

    @staticmethod
    def print_report(results: BenchResults) -> None:
        """Print a formatted report."""
        print("\n" + "=" * 50)
        print("PHONOLOGY BENCHMARK RESULTS")
        print("=" * 50)
        print(f"Mean accuracy: {results.mean_accuracy:.1%}")
        print()
        for task in [results.syllable_counting, results.rhyme_detection, results.minimal_pairs]:
            print(f"{task.task_name}:")
            print(f"  Accuracy: {task.accuracy:.1%} ({task.correct}/{task.total})")
            if task.examples:
                print(f"  Examples:")
                for ex in task.examples[:3]:
                    print(f"    {ex}")
            print()

    def save_results(self, results: BenchResults, output_path: str | Path) -> None:
        """Save results to JSON."""
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(results.to_dict(), f, indent=2)
        logger.info("Saved phonology bench results to %s", path)

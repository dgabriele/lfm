"""LLM translation pilot: IPA → English.

Trains a small language model to translate the emergent IPA output back
into English.  This closes the vision loop: agent → LFM → IPA → LLM →
English.  Even partial success (topic/sentiment recovery) validates that
the IPA output carries recoverable semantic content.

Pipeline:
1. Load English sentences from Leipzig corpus
2. Encode with sentence-transformer → embeddings
3. Pass through frozen faculty → IPA messages
4. Fine-tune a small causal LM on "IPA → English" pairs
5. Evaluate with BLEU and cosine similarity of re-embedded translations

Usage::

    python scripts/train_translator.py
    python scripts/train_translator.py --model_name Qwen/Qwen2.5-0.5B
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader, Dataset

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(message)s",
)
logger = logging.getLogger(__name__)


# ── Dataset ─────────────────────────────────────────────────────────


class IPATranslationDataset(Dataset):
    """Dataset of (IPA_message, English_text) pairs."""

    def __init__(
        self,
        ipa_messages: list[str],
        english_texts: list[str],
        tokenizer,
        max_len: int = 256,
    ) -> None:
        self.pairs = list(zip(ipa_messages, english_texts))
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> dict[str, Tensor]:
        ipa, eng = self.pairs[idx]

        # Format: "<ipa> {ipa_text} </ipa> <eng> {eng_text} </eng>"
        prompt = f"<ipa> {ipa} </ipa> <eng>"
        target = f" {eng} </eng>"
        full = prompt + target

        encoding = self.tokenizer(
            full,
            max_length=self.max_len,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )

        input_ids = encoding["input_ids"].squeeze(0)
        attention_mask = encoding["attention_mask"].squeeze(0)

        # Labels: mask out the prompt portion (only train on the English output)
        prompt_encoding = self.tokenizer(
            prompt,
            max_length=self.max_len,
            truncation=True,
            return_tensors="pt",
        )
        prompt_len = prompt_encoding["input_ids"].shape[1]

        labels = input_ids.clone()
        labels[:prompt_len] = -100  # ignore prompt in loss
        labels[attention_mask == 0] = -100  # ignore padding

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }


# ── Pair generation ─────────────────────────────────────────────────


def generate_pairs(
    texts: list[str],
    faculty,
    encoder,
    batch_size: int = 64,
    device: str = "cuda",
) -> list[tuple[str, str]]:
    """Generate (IPA_message, English_text) pairs.

    Args:
        texts: English source texts.
        faculty: LanguageFaculty instance (eval mode, on device).
        encoder: SentenceTransformer for encoding texts → embeddings.
        batch_size: Batch size for encoding and generation.
        device: Compute device.

    Returns:
        List of (ipa_string, english_string) tuples.
    """
    torch_device = torch.device(device)

    # Encode texts → embeddings
    logger.info("Encoding %d texts...", len(texts))
    embeddings = encoder.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    embeddings = embeddings.astype(np.float32)

    # Generate IPA messages
    pairs: list[tuple[str, str]] = []
    n = len(texts)

    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        batch = torch.tensor(
            embeddings[start:end], dtype=torch.float32, device=torch_device,
        )

        with torch.no_grad():
            out = faculty(batch)

        tokens = out["generator.tokens"]
        ipa_texts = faculty.generator.decode_to_text(tokens)

        for i in range(end - start):
            ipa = ipa_texts[i].strip()
            if ipa:  # skip empty messages
                pairs.append((ipa, texts[start + i]))

        if (start // batch_size) % 10 == 0:
            logger.info("Generated %d / %d pairs", len(pairs), n)

    logger.info("Generated %d valid IPA-English pairs", len(pairs))
    return pairs


# ── BLEU computation ────────────────────────────────────────────────


def compute_bleu(references: list[str], hypotheses: list[str]) -> dict[str, float]:
    """Compute corpus-level BLEU scores (1-4 gram).

    Simple implementation without external dependencies.
    """
    from collections import Counter

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

    # Brevity penalty
    if total_hyp_len == 0:
        return {"bleu_1": 0.0, "bleu_2": 0.0, "bleu_3": 0.0, "bleu_4": 0.0}

    import math

    bp = min(1.0, math.exp(1 - total_ref_len / total_hyp_len))

    results = {}
    log_avg = 0.0
    for n in range(4):
        if total_counts[n] == 0 or total_matches[n] == 0:
            precision = 0.0
        else:
            precision = total_matches[n] / total_counts[n]
        results[f"bleu_{n + 1}"] = bp * precision  # individual BLEU-n
        if precision > 0:
            log_avg += math.log(precision) / 4

    # BLEU-4 (geometric mean)
    if all(total_matches[i] > 0 for i in range(4)):
        results["bleu_4_geometric"] = bp * math.exp(log_avg)
    else:
        results["bleu_4_geometric"] = 0.0

    return results


# ── Training ────────────────────────────────────────────────────────


def train_translator(
    pairs: list[tuple[str, str]],
    model_name: str = "Qwen/Qwen2.5-0.5B",
    epochs: int = 3,
    lr: float = 2e-5,
    batch_size: int = 8,
    max_len: int = 256,
    val_fraction: float = 0.1,
    device: str = "cuda",
    output_dir: str = "data/translator",
) -> dict[str, float]:
    """Fine-tune a causal LM on IPA → English translation.

    Args:
        pairs: List of (IPA, English) pairs.
        model_name: HuggingFace model to fine-tune.
        epochs: Number of training epochs.
        lr: Learning rate.
        batch_size: Training batch size.
        max_len: Maximum sequence length.
        val_fraction: Fraction held out for validation.
        device: Compute device.
        output_dir: Where to save the fine-tuned model.

    Returns:
        Dictionary with training and evaluation metrics.
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer

    torch_device = torch.device(device)

    # Load model and tokenizer
    logger.info("Loading model: %s", model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        trust_remote_code=True,
    ).to(torch_device)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = model.config.eos_token_id

    # Add special tokens for the translation format
    special_tokens = {"additional_special_tokens": ["<ipa>", "</ipa>", "<eng>", "</eng>"]}
    num_added = tokenizer.add_special_tokens(special_tokens)
    if num_added > 0:
        model.resize_token_embeddings(len(tokenizer))
        logger.info("Added %d special tokens", num_added)

    # Split data
    n_val = max(1, int(len(pairs) * val_fraction))
    rng = np.random.default_rng(42)
    perm = rng.permutation(len(pairs))
    val_pairs = [pairs[i] for i in perm[:n_val]]
    train_pairs = [pairs[i] for i in perm[n_val:]]

    logger.info("Train: %d pairs, Val: %d pairs", len(train_pairs), len(val_pairs))

    # Datasets
    train_ds = IPATranslationDataset(
        [p[0] for p in train_pairs],
        [p[1] for p in train_pairs],
        tokenizer, max_len,
    )
    val_ds = IPATranslationDataset(
        [p[0] for p in val_pairs],
        [p[1] for p in val_pairs],
        tokenizer, max_len,
    )

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size)

    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    grad_accum_steps = max(1, 32 // batch_size)

    results: dict[str, float] = {}

    # Training loop
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        n_steps = 0

        for batch_idx, batch in enumerate(train_loader):
            input_ids = batch["input_ids"].to(torch_device)
            attention_mask = batch["attention_mask"].to(torch_device)
            labels = batch["labels"].to(torch_device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )
            loss = outputs.loss / grad_accum_steps
            loss.backward()

            if (batch_idx + 1) % grad_accum_steps == 0:
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad()

            total_loss += outputs.loss.item()
            n_steps += 1

            if batch_idx % 50 == 0:
                logger.info(
                    "  epoch=%d step=%d loss=%.4f",
                    epoch, batch_idx, outputs.loss.item(),
                )

        avg_train_loss = total_loss / max(n_steps, 1)
        logger.info("Epoch %d: avg_train_loss=%.4f", epoch, avg_train_loss)

        # Validation
        model.eval()
        total_val_loss = 0.0
        n_val_steps = 0

        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch["input_ids"].to(torch_device)
                attention_mask = batch["attention_mask"].to(torch_device)
                labels = batch["labels"].to(torch_device)

                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels,
                )
                total_val_loss += outputs.loss.item()
                n_val_steps += 1

        avg_val_loss = total_val_loss / max(n_val_steps, 1)
        logger.info("Epoch %d: avg_val_loss=%.4f", epoch, avg_val_loss)

        results[f"epoch_{epoch}_train_loss"] = avg_train_loss
        results[f"epoch_{epoch}_val_loss"] = avg_val_loss

    # ── Generate translations on val set ─────────────────────────
    logger.info("Generating translations on val set...")
    model.eval()

    references = []
    hypotheses = []

    for ipa, eng in val_pairs[:200]:  # cap at 200 for speed
        prompt = f"<ipa> {ipa} </ipa> <eng>"
        inputs = tokenizer(prompt, return_tensors="pt").to(torch_device)

        with torch.no_grad():
            generated = model.generate(
                **inputs,
                max_new_tokens=64,
                temperature=0.7,
                do_sample=True,
                top_p=0.9,
                pad_token_id=tokenizer.pad_token_id,
            )

        output_text = tokenizer.decode(
            generated[0][inputs["input_ids"].shape[1]:],
            skip_special_tokens=False,
        )

        # Extract text before </eng> if present
        if "</eng>" in output_text:
            output_text = output_text.split("</eng>")[0]
        output_text = output_text.strip()

        references.append(eng)
        hypotheses.append(output_text)

    # ── Evaluate ─────────────────────────────────────────────────
    # BLEU scores
    bleu = compute_bleu(references, hypotheses)
    results.update(bleu)
    logger.info("BLEU scores: %s", {k: f"{v:.4f}" for k, v in bleu.items()})

    # Semantic similarity via re-embedding
    try:
        from sentence_transformers import SentenceTransformer

        eval_encoder = SentenceTransformer("all-MiniLM-L6-v2", device=device)
        ref_emb = eval_encoder.encode(references, convert_to_numpy=True, normalize_embeddings=True)
        hyp_emb = eval_encoder.encode(hypotheses, convert_to_numpy=True, normalize_embeddings=True)

        cosine_sims = np.array([
            float(np.dot(ref_emb[i], hyp_emb[i]))
            for i in range(len(references))
        ])
        results["semantic_cosine_mean"] = float(cosine_sims.mean())
        results["semantic_cosine_std"] = float(cosine_sims.std())
        results["semantic_cosine_median"] = float(np.median(cosine_sims))
        logger.info(
            "Semantic similarity: mean=%.4f, median=%.4f",
            cosine_sims.mean(), np.median(cosine_sims),
        )
        del eval_encoder
    except ImportError:
        logger.warning("sentence-transformers not available for semantic eval")

    # ── Show examples ────────────────────────────────────────────
    logger.info("\n=== Translation Examples ===")
    for i in range(min(10, len(references))):
        ipa = val_pairs[i][0]
        logger.info("  IPA:  %s", ipa[:80])
        logger.info("  REF:  %s", references[i][:80])
        logger.info("  HYP:  %s", hypotheses[i][:80])
        logger.info("")

    # ── Save ─────────────────────────────────────────────────────
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    model.save_pretrained(output_path / "model")
    tokenizer.save_pretrained(output_path / "model")
    logger.info("Saved model to %s", output_path / "model")

    # Save results
    with (output_path / "results.json").open("w") as f:
        json.dump(results, f, indent=2)

    # Save example translations
    examples = [
        {"ipa": val_pairs[i][0], "reference": references[i], "hypothesis": hypotheses[i]}
        for i in range(len(references))
    ]
    with (output_path / "translations.jsonl").open("w") as f:
        for ex in examples:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")

    logger.info("\n=== Final Results ===")
    for k, v in sorted(results.items()):
        logger.info("  %s: %.6f", k, v)

    return results


# ── Main ────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(description="Train IPA → English translator")
    parser.add_argument("--leipzig_dir", default="data/leipzig")
    parser.add_argument("--decoder_path", default="data/vae_decoder.pt")
    parser.add_argument("--spm_path", default="data/spm.model")
    parser.add_argument("--model_name", default="Qwen/Qwen2.5-0.5B")
    parser.add_argument("--max_sentences", type=int, default=5000)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output_dir", default="data/translator")
    args = parser.parse_args()

    from sentence_transformers import SentenceTransformer

    from lfm.data.loaders.leipzig import LeipzigCorpusConfig, LeipzigCorpusLoader
    from lfm.faculty.config import FacultyConfig
    from lfm.faculty.model import LanguageFaculty
    from lfm.generator.config import GeneratorConfig

    torch_device = torch.device(args.device)

    # 1. Load English sentences
    logger.info("Loading Leipzig English sentences...")
    loader = LeipzigCorpusLoader(
        LeipzigCorpusConfig(
            data_dir=args.leipzig_dir,
            languages=["eng"],
            max_samples_per_language=args.max_sentences,
            min_line_length=30,
        )
    )
    samples = loader.load()
    texts = [text for _, text in samples]
    logger.info("Loaded %d sentences", len(texts))

    # 2. Build faculty
    faculty_config = FacultyConfig(
        dim=384,
        generator=GeneratorConfig(
            pretrained_decoder_path=args.decoder_path,
            spm_model_path=args.spm_path,
            freeze_decoder=True,
            max_output_len=32,
        ),
    )
    faculty = LanguageFaculty(faculty_config).to(torch_device)
    faculty.generator.eval()

    # 3. Load sentence encoder
    encoder = SentenceTransformer("all-MiniLM-L6-v2", device=args.device)

    # Trigger lazy init
    with torch.no_grad():
        dummy = torch.randn(1, 384, device=torch_device)
        faculty(dummy)

    # 4. Generate IPA-English pairs
    pairs = generate_pairs(
        texts, faculty, encoder,
        batch_size=64, device=args.device,
    )

    # Free encoder VRAM
    del encoder
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Free faculty VRAM
    del faculty
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # 5. Train translator
    train_translator(
        pairs,
        model_name=args.model_name,
        epochs=args.epochs,
        lr=args.lr,
        batch_size=args.batch_size,
        device=args.device,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()

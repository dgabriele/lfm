"""Self-distill English instruct data from Qwen2.5-0.5B-Instruct.

Loads a pool of instruction prompts from open datasets, batches them
through the base Instruct model, and saves the resulting (prompt,
response) pairs as JSONL for use as the English interleave in bilingual
LoRA training.

The goal is to *preserve* the base model's English generation capacity
during Neuroglot LoRA training — so we specifically want this model's
own output distribution, not an external teacher's.

Prompt pool (default):
    - databricks/databricks-dolly-15k (~15K, Apache 2.0)
    - OpenAssistant/oasst1 root prompts, English only (~10-15K, Apache 2.0)
    - HuggingFaceH4/no_robots (~10K, CC-BY-NC, enable with --no-robots)

Generation produces two responses per prompt at different temperatures,
so the data captures the spread of the base model's assistant-role
distribution rather than a single mode.

The script is resumable: it appends to the output JSONL and skips
any prompt/temperature pairs already written.

Usage::

    poetry run python scripts/self_distill_english.py \\
        --output data/translator/english_distill.jsonl \\
        --batch-size 16
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import torch

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", stream=sys.stderr)
logger = logging.getLogger(__name__)

DEFAULT_SYSTEM_PROMPT = "You are a helpful assistant."


def load_dolly_prompts() -> list[str]:
    from datasets import load_dataset

    logger.info("Loading databricks/databricks-dolly-15k...")
    ds = load_dataset("databricks/databricks-dolly-15k", split="train")
    prompts = []
    for row in ds:
        instruction = (row.get("instruction") or "").strip()
        context = (row.get("context") or "").strip()
        if not instruction:
            continue
        if context:
            user = f"{instruction}\n\n{context}"
        else:
            user = instruction
        prompts.append(user)
    logger.info("  loaded %d dolly prompts", len(prompts))
    return prompts


def load_oasst1_root_prompts() -> list[str]:
    from datasets import load_dataset

    logger.info("Loading OpenAssistant/oasst1 root prompts (English)...")
    ds = load_dataset("OpenAssistant/oasst1", split="train")
    prompts = []
    for row in ds:
        if row.get("parent_id") is not None:
            continue
        if row.get("role") != "prompter":
            continue
        if row.get("lang") != "en":
            continue
        text = (row.get("text") or "").strip()
        if text:
            prompts.append(text)
    logger.info("  loaded %d oasst1 root prompts", len(prompts))
    return prompts


def load_no_robots_prompts() -> list[str]:
    from datasets import load_dataset

    logger.info("Loading HuggingFaceH4/no_robots...")
    ds = load_dataset("HuggingFaceH4/no_robots", split="train")
    prompts = []
    for row in ds:
        msgs = row.get("messages") or []
        if msgs and msgs[0].get("role") == "user":
            text = (msgs[0].get("content") or "").strip()
            if text:
                prompts.append(text)
    logger.info("  loaded %d no_robots prompts", len(prompts))
    return prompts


def build_prompt_pool(
    use_dolly: bool,
    use_oasst: bool,
    use_no_robots: bool,
    max_prompt_chars: int,
) -> list[str]:
    pool: list[str] = []
    if use_dolly:
        pool.extend(load_dolly_prompts())
    if use_oasst:
        pool.extend(load_oasst1_root_prompts())
    if use_no_robots:
        pool.extend(load_no_robots_prompts())

    # Dedupe preserving order.
    seen: set[str] = set()
    unique: list[str] = []
    for p in pool:
        if p in seen:
            continue
        if len(p) > max_prompt_chars:
            continue
        seen.add(p)
        unique.append(p)

    logger.info(
        "Prompt pool: %d total → %d unique ≤ %d chars",
        len(pool), len(unique), max_prompt_chars,
    )
    return unique


def load_existing(output_path: Path) -> set[tuple[str, float]]:
    """Return the set of (prompt, temperature) pairs already written."""
    if not output_path.exists():
        return set()
    done: set[tuple[str, float]] = set()
    with open(output_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                done.add((rec["user"], float(rec["temperature"])))
            except (KeyError, ValueError, json.JSONDecodeError):
                continue
    logger.info("Resume: %d existing records in %s", len(done), output_path)
    return done


def generate_batch(
    model,
    tokenizer,
    messages_batch: list[list[dict]],
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    device: torch.device,
) -> list[str]:
    """Left-pad a batch of chat prompts and generate continuations."""
    prompts = [
        tokenizer.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True,
        )
        for msgs in messages_batch
    ]
    tokenizer.padding_side = "left"
    enc = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=1024,
    ).to(device)
    with torch.no_grad():
        out = model.generate(
            **enc,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    input_len = enc["input_ids"].shape[1]
    responses = []
    for i in range(out.shape[0]):
        generated = out[i, input_len:]
        text = tokenizer.decode(generated, skip_special_tokens=True).strip()
        responses.append(text)
    return responses


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument(
        "--output", type=Path,
        default=Path("data/translator/english_distill.jsonl"),
    )
    parser.add_argument("--system-prompt", default=DEFAULT_SYSTEM_PROMPT)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-new-tokens", type=int, default=384)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument(
        "--temperatures", type=float, nargs="+", default=[0.7, 1.0],
        help="Generate one response per prompt at each temperature.",
    )
    parser.add_argument("--max-prompt-chars", type=int, default=2000)
    parser.add_argument("--no-dolly", action="store_true")
    parser.add_argument("--no-oasst", action="store_true")
    parser.add_argument(
        "--no-robots", action="store_true",
        help="Include HuggingFaceH4/no_robots (CC-BY-NC).",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--limit", type=int, default=None,
        help="Cap the unique prompt pool to the first N prompts (for testing).",
    )
    args = parser.parse_args()

    device = torch.device(args.device)
    args.output.parent.mkdir(parents=True, exist_ok=True)

    pool = build_prompt_pool(
        use_dolly=not args.no_dolly,
        use_oasst=not args.no_oasst,
        use_no_robots=args.no_robots,
        max_prompt_chars=args.max_prompt_chars,
    )
    if args.limit is not None:
        pool = pool[:args.limit]
        logger.info("Limited pool to %d prompts", len(pool))

    if not pool:
        logger.error("Prompt pool is empty — nothing to do.")
        return

    total_target = len(pool) * len(args.temperatures)
    already_done = load_existing(args.output)

    from transformers import AutoModelForCausalLM, AutoTokenizer

    logger.info("Loading model: %s", args.model)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16,
    ).to(device)
    model.eval()
    logger.info("Model loaded on %s", device)

    # Work queue: (prompt, temperature) pairs not yet written.
    work: list[tuple[str, float]] = []
    for temp in args.temperatures:
        for prompt in pool:
            if (prompt, temp) not in already_done:
                work.append((prompt, temp))
    logger.info(
        "Work: %d pending of %d total (%.1f%% already done)",
        len(work), total_target,
        100.0 * (total_target - len(work)) / max(total_target, 1),
    )

    # Group work by temperature so each generate() call uses a single temperature.
    with open(args.output, "a", encoding="utf-8") as f:
        written = 0
        start_time = time.time()
        for temp in args.temperatures:
            temp_work = [p for p, t in work if t == temp]
            if not temp_work:
                continue
            logger.info(
                "Pass temperature=%.2f: %d prompts to generate",
                temp, len(temp_work),
            )
            for batch_start in range(0, len(temp_work), args.batch_size):
                batch_prompts = temp_work[batch_start:batch_start + args.batch_size]
                messages_batch = [
                    [
                        {"role": "system", "content": args.system_prompt},
                        {"role": "user", "content": p},
                    ]
                    for p in batch_prompts
                ]
                try:
                    responses = generate_batch(
                        model, tokenizer, messages_batch,
                        max_new_tokens=args.max_new_tokens,
                        temperature=temp,
                        top_p=args.top_p,
                        device=device,
                    )
                except torch.cuda.OutOfMemoryError:
                    torch.cuda.empty_cache()
                    logger.error(
                        "OOM at batch_start=%d, batch_size=%d — "
                        "rerun with smaller --batch-size",
                        batch_start, args.batch_size,
                    )
                    raise

                for prompt, response in zip(batch_prompts, responses):
                    if not response:
                        continue
                    rec = {
                        "system": args.system_prompt,
                        "user": prompt,
                        "assistant": response,
                        "temperature": temp,
                    }
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    written += 1

                f.flush()

                if written % 200 == 0 or batch_start + args.batch_size >= len(temp_work):
                    elapsed = time.time() - start_time
                    rate = written / max(elapsed, 1e-6)
                    remaining = (len(work) - written) / max(rate, 1e-6)
                    logger.info(
                        "  progress: %d/%d (%.0f%%) — %.1f rec/s — ETA %.0f min",
                        written, len(work),
                        100.0 * written / max(len(work), 1),
                        rate, remaining / 60.0,
                    )

        elapsed = time.time() - start_time
        logger.info(
            "Done: wrote %d new records in %.0fs (%.1f rec/s), total in file = %d",
            written, elapsed, written / max(elapsed, 1e-6),
            len(already_done) + written,
        )


if __name__ == "__main__":
    main()

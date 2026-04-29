"""Analyze statistical properties of generated alien text vs ground-truth cipher.

Loads a Phase 1 checkpoint, samples generated alien sequences with temperature
sampling, and compares their statistical fingerprint (Zipf slope, n-gram entropies,
KL divergence, length distribution) to a held-out reference cipher batch.

Runs on CPU by default to avoid conflicting with a concurrent training run.

Usage:
  poetry run python scripts/analyze_phase1_statistics.py \\
      configs/synth_local_qwen.yaml \\
      --checkpoint data/synth_qwen_local/phase1_checkpoint.pt \\
      --n-samples 200 --max-len 64 --device cpu
"""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import yaml
from transformers import PreTrainedTokenizerFast

from lfm.synth.backend import CausalDecoderBackend
from lfm.synth.cipher import WordCipher
from lfm.synth.config import SynthConfig
from lfm.synth.model import SynthLM
from lfm.synth.vocab import AlienVocab


# ── statistics ────────────────────────────────────────────────────────────────

def unigram_stats(sequences: list[list[int]]) -> dict:
    flat = [t for seq in sequences for t in seq]
    counts = Counter(flat)
    n_total = sum(counts.values())
    n_unique = len(counts)
    probs = np.array(sorted(counts.values(), reverse=True), dtype=np.float64) / n_total
    entropy = float(-(probs * np.log(probs + 1e-12)).sum())
    ranks = np.arange(1, len(probs) + 1)
    log_rank = np.log(ranks)
    log_freq = np.log(probs)
    slope = float(np.polyfit(log_rank, log_freq, 1)[0])
    return {
        "tokens": n_total,
        "types": n_unique,
        "ttr": n_unique / n_total,
        "entropy_nats": entropy,
        "zipf_slope": slope,
    }


def bigram_stats(sequences: list[list[int]]) -> tuple[Counter, dict]:
    bg = Counter()
    for seq in sequences:
        for a, b in zip(seq[:-1], seq[1:]):
            bg[(a, b)] += 1
    n_total = sum(bg.values())
    probs = np.array(sorted(bg.values(), reverse=True), dtype=np.float64) / n_total
    entropy = float(-(probs * np.log(probs + 1e-12)).sum())
    return bg, {
        "bigram_tokens": n_total,
        "bigram_types": len(bg),
        "bigram_entropy_nats": entropy,
    }


def bigram_kl(p_bg: Counter, q_bg: Counter, smooth: float = 1e-6) -> float:
    """KL(p || q) over union of bigram support, with floor smoothing."""
    p_total = sum(p_bg.values())
    q_total = sum(q_bg.values())
    keys = set(p_bg) | set(q_bg)
    kl = 0.0
    for k in keys:
        p = p_bg.get(k, 0) / p_total
        q = q_bg.get(k, 0) / q_total
        if p > 0:
            kl += p * np.log((p + smooth) / (q + smooth))
    return float(kl)


def ngram_diversity(sequences: list[list[int]], n: int) -> dict:
    grams = Counter()
    total = 0
    for seq in sequences:
        for i in range(len(seq) - n + 1):
            grams[tuple(seq[i : i + n])] += 1
            total += 1
    return {
        f"unique_{n}grams": len(grams),
        f"diversity_{n}gram": (len(grams) / total) if total else 0.0,
    }


def repetition_rates(sequences: list[list[int]], n: int) -> float:
    """Fraction of n-grams that exactly repeat the immediately preceding n-gram."""
    rep = total = 0
    for seq in sequences:
        for i in range(2 * n, len(seq) + 1):
            cur = tuple(seq[i - n : i])
            prev = tuple(seq[i - 2 * n : i - n])
            if cur == prev:
                rep += 1
            total += 1
    return (rep / total) if total else 0.0


def js_divergence(p_counts: Counter, q_counts: Counter, smooth: float = 1e-6) -> float:
    """Symmetric, bounded Jensen-Shannon divergence in nats."""
    p_total = sum(p_counts.values())
    q_total = sum(q_counts.values())
    keys = set(p_counts) | set(q_counts)
    js = 0.0
    for k in keys:
        p = p_counts.get(k, 0) / p_total
        q = q_counts.get(k, 0) / q_total
        m = 0.5 * (p + q)
        if p > 0:
            js += 0.5 * p * np.log((p + smooth) / (m + smooth))
        if q > 0:
            js += 0.5 * q * np.log((q + smooth) / (m + smooth))
    return float(js)


def top_k_overlap(p_counts: Counter, q_counts: Counter, ks: list[int]) -> dict:
    """For each K: |top-K(p) ∩ top-K(q)| / K and KL on the top-K support."""
    p_sorted = [k for k, _ in p_counts.most_common()]
    q_sorted = [k for k, _ in q_counts.most_common()]
    out = {}
    for k in ks:
        p_top = set(p_sorted[:k])
        q_top = set(q_sorted[:k])
        out[f"top{k}_overlap"] = len(p_top & q_top) / k
    return out


def conditional_distribution(sequences: list[list[int]], anchor: int, top_n: int = 10) -> Counter:
    """Distribution of the token immediately following `anchor`."""
    succ = Counter()
    for seq in sequences:
        for a, b in zip(seq[:-1], seq[1:]):
            if a == anchor:
                succ[b] += 1
    return succ


def length_stats(sequences: list[list[int]]) -> dict:
    lens = np.array([len(s) for s in sequences])
    return {
        "len_mean": float(lens.mean()),
        "len_median": float(np.median(lens)),
        "len_std": float(lens.std()),
        "len_min": int(lens.min()),
        "len_max": int(lens.max()),
    }


# ── generation ────────────────────────────────────────────────────────────────

@torch.no_grad()
def generate_samples(
    model: SynthLM,
    seed_ids: torch.Tensor,
    max_len: int,
    eos_id: int,
    temperature: float,
    device: torch.device,
) -> list[list[int]]:
    """Temperature-sampled autoregressive generation. seed_ids: (B, S) int64."""
    B = seed_ids.size(0)
    context = model.backend.embed_alien(seed_ids.to(device))
    out: list[list[int]] = [seed_ids[i].tolist() for i in range(B)]
    done = torch.zeros(B, dtype=torch.bool, device=device)
    for _ in range(max_len):
        hidden = model.backend.forward_hidden(context)
        logits = model.backend.alien_logits(hidden[:, -1]) / temperature
        probs = torch.softmax(logits.float(), dim=-1)
        next_id = torch.multinomial(probs, num_samples=1).squeeze(-1)
        for i in range(B):
            if not done[i]:
                out[i].append(int(next_id[i]))
                if int(next_id[i]) == eos_id:
                    done[i] = True
        if done.all():
            break
        context = torch.cat([context, model.backend.embed_alien(next_id.unsqueeze(1))], dim=1)
    # Trim at first EOS if present
    trimmed = []
    for seq in out:
        if eos_id in seq:
            seq = seq[: seq.index(eos_id) + 1]
        trimmed.append(seq)
    return trimmed


def cipher_reference(
    sentences: list[str], cipher: WordCipher, alien_tok: PreTrainedTokenizerFast, max_len: int
) -> list[list[int]]:
    """Tokenize cipher-encoded sentences, truncating to max_len + 1."""
    encoded = cipher.encode_batch(sentences)
    ids = alien_tok(encoded, padding=False, truncation=True, max_length=max_len + 1)["input_ids"]
    return ids


# ── main ─────────────────────────────────────────────────────────────────────

def report(label: str, stats: dict) -> None:
    print(f"\n=== {label} ===")
    for k, v in stats.items():
        if isinstance(v, float):
            print(f"  {k:<22s} {v:.4f}")
        else:
            print(f"  {k:<22s} {v}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("config", help="YAML config")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--n-samples", type=int, default=200)
    parser.add_argument("--max-len", type=int, default=64)
    parser.add_argument("--seed-len", type=int, default=2,
                        help="Number of cipher tokens to seed each generation with.")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    torch.manual_seed(args.seed)

    cfg = SynthConfig(**yaml.safe_load(Path(args.config).read_text()))
    out_dir = Path(cfg.output_dir)
    vocab = AlienVocab.load(out_dir)
    alien_tok = PreTrainedTokenizerFast.from_pretrained(str(out_dir / "alien_tokenizer"))
    cipher = WordCipher(vocab)

    print(f"Loading backend: {cfg.base_model_name}")
    backend = CausalDecoderBackend(
        cfg.base_model_name,
        alien_vocab_size=len(alien_tok),
        with_reference_body=False,  # not needed for analysis
    )
    model = SynthLM(backend, cfg)
    print(f"Loading checkpoint: {args.checkpoint}")
    raw = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    state = raw["model"] if isinstance(raw, dict) and "model" in raw else raw
    model.load_phase1_state(state)
    device = torch.device(args.device)
    model.to(device).eval()

    # Load corpus and pick held-out sentences (offset to avoid training overlap).
    dataset_path = Path(cfg.phase1_dataset_dir)
    if dataset_path.suffix == ".jsonl":
        all_lines = [json.loads(l)["text"] for l in dataset_path.read_text().splitlines() if l.strip()]
    else:
        all_lines = [l.strip() for l in dataset_path.read_text().splitlines() if l.strip()]
    rng.shuffle(all_lines)
    picked = all_lines[: args.n_samples]
    print(f"Sampling {args.n_samples} held-out sentences (corpus size: {len(all_lines)})")

    # Reference cipher token sequences.
    ref_seqs = cipher_reference(picked, cipher, alien_tok, args.max_len)

    # Generated sequences seeded with the first `seed_len` tokens of each cipher seq.
    eos_id = alien_tok.eos_token_id
    pad_id = alien_tok.pad_token_id or 0
    seeds = [seq[: args.seed_len] for seq in ref_seqs]
    max_seed = max(len(s) for s in seeds)
    seed_tensor = torch.full((len(seeds), max_seed), pad_id, dtype=torch.long)
    for i, s in enumerate(seeds):
        seed_tensor[i, : len(s)] = torch.tensor(s, dtype=torch.long)

    print(f"Generating with temperature={args.temperature} on {device}...")
    # Process in mini-batches to bound memory.
    bs = 16 if device.type == "cuda" else 8
    gen_seqs: list[list[int]] = []
    for i in range(0, len(seed_tensor), bs):
        chunk = seed_tensor[i : i + bs]
        out = generate_samples(model, chunk, args.max_len, eos_id, args.temperature, device)
        gen_seqs.extend(out)
        print(f"  generated {min(i + bs, len(seed_tensor))}/{len(seed_tensor)}")

    # ── analyze ──
    print("\n" + "=" * 64)
    print("STATISTICAL FINGERPRINT COMPARISON")
    print("=" * 64)

    ref_uni, gen_uni = unigram_stats(ref_seqs), unigram_stats(gen_seqs)
    report("UNIGRAM — reference cipher", ref_uni)
    report("UNIGRAM — generated", gen_uni)

    ref_bg, ref_bg_stats = bigram_stats(ref_seqs)
    gen_bg, gen_bg_stats = bigram_stats(gen_seqs)
    report("BIGRAM — reference cipher", ref_bg_stats)
    report("BIGRAM — generated", gen_bg_stats)
    print(f"\n  KL(gen || ref) bigram:   {bigram_kl(gen_bg, ref_bg):.4f} nats")
    print(f"  KL(ref || gen) bigram:   {bigram_kl(ref_bg, gen_bg):.4f} nats")
    print(f"  JS divergence  bigram:   {js_divergence(gen_bg, ref_bg):.4f} nats  (max ≈ 0.693)")

    # Unigram JS — independent of bigram sample density.
    ref_unigrams = Counter(t for s in ref_seqs for t in s)
    gen_unigrams = Counter(t for s in gen_seqs for t in s)
    print(f"  JS divergence  unigram:  {js_divergence(gen_unigrams, ref_unigrams):.4f} nats")

    print("\n=== TOP-K BIGRAM OVERLAP (does the model produce the same load-bearing patterns?) ===")
    overlap = top_k_overlap(ref_bg, gen_bg, [10, 25, 50, 100, 250])
    for k, v in overlap.items():
        print(f"  {k:<22s} {v:.3f}")

    print("\n=== TOP-K UNIGRAM OVERLAP ===")
    overlap_u = top_k_overlap(ref_unigrams, gen_unigrams, [10, 25, 50, 100, 250])
    for k, v in overlap_u.items():
        print(f"  {k:<22s} {v:.3f}")

    # Conditional distributions for the top-5 most frequent unigrams in reference.
    print("\n=== CONDITIONAL NEXT-TOKEN DISTRIBUTIONS (top-5 anchor tokens) ===")
    print("  For each high-frequency 'function-like' token, compare what comes after.")
    print("  Reports JS divergence between ref's and gen's successor distributions.")
    top_anchors = [tok for tok, _ in ref_unigrams.most_common(5)]
    for anchor in top_anchors:
        ref_succ = conditional_distribution(ref_seqs, anchor)
        gen_succ = conditional_distribution(gen_seqs, anchor)
        js = js_divergence(gen_succ, ref_succ)
        anchor_str = alien_tok.decode([anchor], skip_special_tokens=False)
        ref_total = sum(ref_succ.values())
        gen_total = sum(gen_succ.values())
        ref_top3 = ", ".join(f"{alien_tok.decode([t], skip_special_tokens=False)}({c})"
                             for t, c in ref_succ.most_common(3))
        gen_top3 = ", ".join(f"{alien_tok.decode([t], skip_special_tokens=False)}({c})"
                             for t, c in gen_succ.most_common(3))
        print(f"\n  anchor={anchor_str!r:<14s}  ref_n={ref_total:<5d} gen_n={gen_total:<5d}  JS={js:.3f}")
        print(f"    ref top-3: {ref_top3}")
        print(f"    gen top-3: {gen_top3}")

    print("\n=== N-GRAM DIVERSITY ===")
    for n in (2, 3, 4):
        r = ngram_diversity(ref_seqs, n)
        g = ngram_diversity(gen_seqs, n)
        print(f"  n={n}  ref unique={r[f'unique_{n}grams']:>6d}  div={r[f'diversity_{n}gram']:.4f}  "
              f"|  gen unique={g[f'unique_{n}grams']:>6d}  div={g[f'diversity_{n}gram']:.4f}")

    print("\n=== REPETITION RATES (consecutive n-gram repeats) ===")
    for n in (1, 2, 3):
        r = repetition_rates(ref_seqs, n)
        g = repetition_rates(gen_seqs, n)
        print(f"  n={n}  ref={r:.4f}  gen={g:.4f}")

    report("LENGTH — reference cipher", length_stats(ref_seqs))
    report("LENGTH — generated", length_stats(gen_seqs))

    # Side-by-side text samples
    print("\n=== SAMPLE PAIRS ===")
    for i in range(min(5, len(picked))):
        print(f"\n  EN:  {picked[i][:120]}")
        print(f"  REF: {alien_tok.decode(ref_seqs[i], skip_special_tokens=True)[:160]}")
        print(f"  GEN: {alien_tok.decode(gen_seqs[i], skip_special_tokens=True)[:160]}")


if __name__ == "__main__":
    main()

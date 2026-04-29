"""Probes for relational/compositional structure preservation.

Layer 1 — Embedding probe (no Phase 1 needed):
  Measure cosine similarity between Qwen mean-pool embeddings of minimal
  pair sentences that differ only in relational structure (anaphora,
  argument role swap, tense, synonymy). Tells us whether the source
  embeddings even carry relational distinctions before our system sees
  them.

Layer 2 — Anaphoric scaffolding probe:
  Identify the alien BPE tokens that correspond to English pronouns
  ("he", "she", "it", ...) via the cipher. Measure where these
  pronoun-like tokens appear in (a) reference cipher text and
  (b) Phase 1 generated text. Compare positional statistics: do the
  pronoun-like tokens appear after content-noun-like tokens at similar
  rates? This tests whether the LM has learned the positional grammar
  of pronominal reference even though it cannot resolve binding.

Usage:
  poetry run python scripts/probe_relational_structure.py \\
      configs/synth_local_qwen.yaml \\
      --checkpoint data/synth_qwen_local/phase1_checkpoint.pt \\
      --n-samples 200 --device cpu
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
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedTokenizerFast,
)

from lfm.synth.backend import CausalDecoderBackend
from lfm.synth.cipher import WordCipher
from lfm.synth.config import SynthConfig
from lfm.synth.model import SynthLM
from lfm.synth.vocab import AlienVocab


ENGLISH_PRONOUNS = [
    "he", "she", "it", "they", "him", "her", "them",
    "his", "hers", "their", "theirs",
]


# ── Layer 1: minimal-pair embedding probe ────────────────────────────────────

# Each pair-set tests one type of relational distinction.
# A LOW similarity within a pair-set means the embedding distinguishes
# that relational change. A HIGH similarity means the change is collapsed.
PAIR_SETS: dict[str, list[tuple[str, str]]] = {
    "anaphora_vs_repeated_name": [
        ("Daniel said hello and then he went to the store.",
         "Daniel said hello and then Daniel went to the store."),
        ("Mary picked up the book and she read it carefully.",
         "Mary picked up the book and Mary read the book carefully."),
        ("The dog wagged its tail when it saw the postman.",
         "The dog wagged the dog's tail when the dog saw the postman."),
        ("After Bob arrived, he greeted everyone warmly.",
         "After Bob arrived, Bob greeted everyone warmly."),
    ],
    "anaphora_vs_different_referent": [
        ("Daniel said hello and then he went to the store.",
         "Daniel said hello and then Bob went to the store."),
        ("Mary picked up the book and she read it carefully.",
         "Mary picked up the book and Lisa read the book carefully."),
        ("After Bob arrived, he greeted everyone warmly.",
         "After Bob arrived, Tom greeted everyone warmly."),
    ],
    "argument_role_swap": [
        ("The cat chased the dog through the yard.",
         "The dog chased the cat through the yard."),
        ("Bob hit Daniel after the meeting.",
         "Daniel hit Bob after the meeting."),
        ("Mary gave Tom the book yesterday.",
         "Tom gave Mary the book yesterday."),
        ("The teacher praised the student loudly.",
         "The student praised the teacher loudly."),
    ],
    "tense_change": [
        ("Daniel met Bob yesterday at the park.",
         "Daniel will meet Bob tomorrow at the park."),
        ("The cat chased the dog through the yard.",
         "The cat will chase the dog through the yard."),
        ("Mary read the book carefully.",
         "Mary had read the book carefully."),
    ],
    "synonymy": [
        ("Bob hit Daniel after the meeting.",
         "Bob struck Daniel after the meeting."),
        ("The cat chased the dog through the yard.",
         "The cat pursued the dog through the yard."),
        ("Mary read the book carefully.",
         "Mary perused the book carefully."),
    ],
    "unrelated_baseline": [
        ("Daniel said hello and then he went to the store.",
         "The committee approved the budget unanimously last week."),
        ("Mary picked up the book and she read it carefully.",
         "Heavy rains caused flooding in three coastal towns."),
        ("The cat chased the dog through the yard.",
         "Quantum entanglement violates classical local realism."),
    ],
}


@torch.no_grad()
def encode_hidden(model, tokenizer, sentences: list[str], device: torch.device):
    """Forward through Qwen and return (hidden_states, attention_mask)."""
    enc = tokenizer(sentences, padding=True, truncation=True, return_tensors="pt").to(device)
    out = model(**enc, output_hidden_states=True)
    return out.hidden_states[-1], enc["attention_mask"]   # (N, T, D), (N, T)


def _pool_mean(hidden, mask):
    m = mask.unsqueeze(-1).float()
    return (hidden * m).sum(dim=1) / m.sum(dim=1).clamp(min=1.0)


def _pool_last(hidden, mask):
    """Take the hidden state at the last non-pad position (true last token)."""
    last_idx = (mask.sum(dim=1) - 1).clamp(min=0).long()      # (N,)
    return hidden[torch.arange(hidden.size(0), device=hidden.device), last_idx]


def _pool_last_k_mean(hidden, mask, k: int):
    """Mean over the last k non-pad tokens of each sequence."""
    N, T, D = hidden.shape
    seq_lens = mask.sum(dim=1).long()                          # (N,)
    out = torch.zeros(N, D, device=hidden.device, dtype=hidden.dtype)
    for i in range(N):
        L = int(seq_lens[i])
        kk = min(k, L)
        out[i] = hidden[i, L - kk : L].mean(dim=0)
    return out


def _pool_max(hidden, mask):
    m = mask.unsqueeze(-1).float()
    h = hidden.masked_fill(m == 0, float("-inf"))
    return h.max(dim=1).values


def _pool_first_mean_last(hidden, mask):
    first = hidden[:, 0]
    mean = _pool_mean(hidden, mask)
    last = _pool_last(hidden, mask)
    return torch.cat([first, mean, last], dim=-1)


def _pool_last_k_concat(hidden, mask, k: int):
    """Concatenate the last k non-pad hidden states into a (B, k*D) vector.

    For sequences shorter than k, the missing leading positions are padded
    with zeros — the cosine on the flattened vector then naturally weights
    the present positions correctly.
    """
    N, T, D = hidden.shape
    seq_lens = mask.sum(dim=1).long()
    out = torch.zeros(N, k * D, device=hidden.device, dtype=hidden.dtype)
    for i in range(N):
        L = int(seq_lens[i])
        kk = min(k, L)
        # Place the last kk hidden states at the END of the concat slot,
        # so position alignment ('most recent') is consistent across samples.
        slice_h = hidden[i, L - kk : L].reshape(-1)            # (kk*D,)
        out[i, (k - kk) * D :] = slice_h
    return out


def _pool_xattn_last_k(hidden, mask, k: int):
    """Non-parametric cross-attention: the last k hidden states act as queries
    that softmax-attend over the full sequence (keys=values=hidden).
    Returns (B, k*D) flattened. This is a conservative lower bound on what
    a *learned* K-query cross-attention pooler (Option B) could achieve, since
    learned queries are strictly more expressive than identity-weight queries.
    """
    N, T, D = hidden.shape
    seq_lens = mask.sum(dim=1).long()
    scale = D ** -0.5
    out = torch.zeros(N, k * D, device=hidden.device, dtype=hidden.dtype)
    for i in range(N):
        L = int(seq_lens[i])
        kk = min(k, L)
        h = hidden[i, :L]            # (L, D)
        q = h[L - kk : L]            # (kk, D), the most-recent positions
        scores = (q @ h.T) * scale   # (kk, L)
        attn = torch.softmax(scores.float(), dim=-1).to(h.dtype)
        result = attn @ h            # (kk, D)
        out[i, (k - kk) * D :] = result.reshape(-1)
    return out


POOLERS = {
    "mean":              _pool_mean,
    "last":              _pool_last,
    "last4_mean":        lambda h, m: _pool_last_k_mean(h, m, 4),
    "last8_mean":        lambda h, m: _pool_last_k_mean(h, m, 8),
    "max":               _pool_max,
    "first_mean_last":   _pool_first_mean_last,
    "last4_concat":      lambda h, m: _pool_last_k_concat(h, m, 4),
    "last8_concat":      lambda h, m: _pool_last_k_concat(h, m, 8),
    "last16_concat":     lambda h, m: _pool_last_k_concat(h, m, 16),
    "xattn_last4":       lambda h, m: _pool_xattn_last_k(h, m, 4),
    "xattn_last8":       lambda h, m: _pool_xattn_last_k(h, m, 8),
}


@torch.no_grad()
def mean_pool_embed(
    model, tokenizer, sentences: list[str], device: torch.device,
    strategy: str = "mean",
) -> torch.Tensor:
    """L2-normalised pooled embedding under the chosen strategy."""
    hidden, mask = encode_hidden(model, tokenizer, sentences, device)
    pooled = POOLERS[strategy](hidden, mask)
    return torch.nn.functional.normalize(pooled.float(), dim=-1)


def layer1_probe(model_name: str, device: torch.device) -> None:
    print("\n" + "=" * 96)
    print("LAYER 1 — Pooling-strategy comparison on relational minimal pairs")
    print("=" * 96)
    print(f"\nLoading {model_name} for embedding...")
    tok = AutoTokenizer.from_pretrained(model_name)
    mdl = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.bfloat16
    ).to(device).eval()

    # Compute mean cosine sim per (strategy, category)
    strategies = list(POOLERS.keys())
    table: dict[str, dict[str, float]] = {s: {} for s in strategies}
    for strategy in strategies:
        for label, pairs in PAIR_SETS.items():
            sims = []
            for a, b in pairs:
                emb = mean_pool_embed(mdl, tok, [a, b], device, strategy=strategy)
                sims.append(float((emb[0] * emb[1]).sum()))
            table[strategy][label] = float(np.mean(sims))

    # Print: rows = categories, cols = strategies
    cats = list(PAIR_SETS.keys())
    print()
    header = f"  {'category':<34s}" + "".join(f"{s:>14s}" for s in strategies)
    print(header)
    print("  " + "-" * (len(header) - 2))
    for label in cats:
        row = f"  {label:<34s}" + "".join(f"{table[s][label]:>14.4f}" for s in strategies)
        print(row)

    print("\n=== DERIVED METRICS (per pooling strategy) ===")
    print("  spread          = unrelated_baseline distance from 1.0 (how far apart unrelated sentences are)")
    print("  syn_recall      = synonymy similarity  (higher = recognises semantic equivalence)")
    print("  role_split      = 1 - argument_role_swap (higher = distinguishes 'A hit B' from 'B hit A')")
    print("  ref_split       = 1 - anaphora_vs_different_referent (higher = distinguishes 'he=Daniel' from 'he=Bob')")
    print("  rel_score       = (role_split + ref_split) / 2 (composite relational sensitivity)")
    print("  signal_to_noise = rel_score / (1 - syn_recall)  (relational signal scaled by semantic noise)")
    print()
    print(f"  {'strategy':<18s} {'spread':>9s} {'syn_recall':>11s} "
          f"{'role_split':>11s} {'ref_split':>11s} {'rel_score':>11s} {'snr':>9s}")
    print("  " + "-" * 86)
    for s in strategies:
        spread = 1.0 - table[s]["unrelated_baseline"]
        syn_recall = table[s]["synonymy"]
        role_split = 1.0 - table[s]["argument_role_swap"]
        ref_split = 1.0 - table[s]["anaphora_vs_different_referent"]
        rel = (role_split + ref_split) / 2
        snr = rel / max(1.0 - syn_recall, 1e-6)
        print(f"  {s:<18s} {spread:>9.4f} {syn_recall:>11.4f} "
              f"{role_split:>11.4f} {ref_split:>11.4f} {rel:>11.4f} {snr:>9.3f}")


# ── Layer 2: anaphoric scaffolding probe ─────────────────────────────────────

@torch.no_grad()
def generate_samples(
    model: SynthLM, seed_ids: torch.Tensor, max_len: int, eos_id: int,
    temperature: float, device: torch.device,
) -> list[list[int]]:
    B = seed_ids.size(0)
    context = model.backend.embed_alien(seed_ids.to(device))
    out: list[list[int]] = [seed_ids[i].tolist() for i in range(B)]
    done = torch.zeros(B, dtype=torch.bool, device=device)
    for _ in range(max_len):
        hidden = model.backend.forward_hidden(context)
        logits = model.backend.alien_logits(hidden[:, -1]) / temperature
        probs = torch.softmax(logits.float(), dim=-1)
        nxt = torch.multinomial(probs, num_samples=1).squeeze(-1)
        for i in range(B):
            if not done[i]:
                out[i].append(int(nxt[i]))
                if int(nxt[i]) == eos_id:
                    done[i] = True
        if done.all():
            break
        context = torch.cat([context, model.backend.embed_alien(nxt.unsqueeze(1))], dim=1)
    trimmed = []
    for seq in out:
        if eos_id in seq:
            seq = seq[: seq.index(eos_id) + 1]
        trimmed.append(seq)
    return trimmed


def identify_pronoun_tokens(
    cipher: WordCipher, alien_tok: PreTrainedTokenizerFast
) -> tuple[set[int], dict[str, list[int]]]:
    """Map English pronouns through cipher → BPE token IDs."""
    pronoun_ids: set[int] = set()
    pronoun_to_ids: dict[str, list[int]] = {}
    for p in ENGLISH_PRONOUNS:
        ciphered = cipher.encode_sentence(p).strip()
        ids = alien_tok(ciphered, add_special_tokens=False)["input_ids"]
        pronoun_to_ids[p] = ids
        pronoun_ids.update(ids)
    return pronoun_ids, pronoun_to_ids


def analyze_anaphoric_positions(
    sequences: list[list[int]],
    pronoun_ids: set[int],
    alien_tok: PreTrainedTokenizerFast,
    content_min_chars: int = 4,
) -> dict:
    """For every pronoun-like token occurrence, record its absolute and
    relative position, and whether a 'content-like' (long) token appeared
    earlier in the sequence."""
    abs_positions: list[int] = []
    rel_positions: list[float] = []
    distances_back: list[int] = []
    no_prior_content = 0
    pronoun_token_freq: Counter = Counter()
    n_seqs_with_pronoun = 0

    for seq in sequences:
        had = False
        for i, tok in enumerate(seq):
            if tok in pronoun_ids:
                had = True
                pronoun_token_freq[tok] += 1
                abs_positions.append(i)
                rel_positions.append(i / max(len(seq) - 1, 1))
                # Look backward for a content-like (long-decoded) token
                found = None
                for back in range(1, i + 1):
                    prev = seq[i - back]
                    s = alien_tok.decode([prev], skip_special_tokens=False)
                    if len(s) >= content_min_chars:
                        found = back
                        break
                if found is None:
                    no_prior_content += 1
                else:
                    distances_back.append(found)
        if had:
            n_seqs_with_pronoun += 1

    n_occ = len(abs_positions)
    return {
        "n_sequences": len(sequences),
        "n_with_pronoun": n_seqs_with_pronoun,
        "n_pronoun_occurrences": n_occ,
        "occurrences_per_seq": n_occ / max(len(sequences), 1),
        "mean_abs_pos": float(np.mean(abs_positions)) if abs_positions else 0.0,
        "mean_rel_pos": float(np.mean(rel_positions)) if rel_positions else 0.0,
        "with_prior_content_pct": (
            (n_occ - no_prior_content) / n_occ * 100 if n_occ else 0.0
        ),
        "mean_dist_back": (
            float(np.mean(distances_back)) if distances_back else 0.0
        ),
        "median_dist_back": (
            float(np.median(distances_back)) if distances_back else 0.0
        ),
    }


def layer2_probe(
    cfg: SynthConfig, checkpoint_path: str, n_samples: int, max_len: int,
    device: torch.device, seed: int,
) -> None:
    print("\n\n" + "=" * 76)
    print("LAYER 2 — Anaphoric scaffolding in alien generation")
    print("=" * 76)

    out_dir = Path(cfg.output_dir)
    vocab = AlienVocab.load(out_dir)
    alien_tok = PreTrainedTokenizerFast.from_pretrained(str(out_dir / "alien_tokenizer"))
    cipher = WordCipher(vocab)

    pronoun_ids, pronoun_to_ids = identify_pronoun_tokens(cipher, alien_tok)
    print(f"\nMapping English pronouns to alien BPE tokens:")
    for p, ids in pronoun_to_ids.items():
        decoded = " ".join(alien_tok.decode([i], skip_special_tokens=False) for i in ids)
        print(f"  {p:<8s} → {ids}  =  {decoded!r}")
    print(f"\nTotal distinct pronoun-like token IDs: {len(pronoun_ids)}")

    print(f"\nLoading Phase 1 backend + checkpoint...")
    backend = CausalDecoderBackend(
        cfg.base_model_name, alien_vocab_size=len(alien_tok), with_reference_body=False,
    )
    model = SynthLM(backend, cfg)
    raw = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    state = raw["model"] if isinstance(raw, dict) and "model" in raw else raw
    model.load_phase1_state(state)
    model.to(device).eval()

    # Filter corpus for pronoun-containing sentences
    print("\nSampling pronoun-containing sentences from corpus...")
    dataset_path = Path(cfg.phase1_dataset_dir)
    if dataset_path.suffix == ".jsonl":
        all_lines = [json.loads(l)["text"]
                     for l in dataset_path.read_text().splitlines() if l.strip()]
    else:
        all_lines = [l.strip() for l in dataset_path.read_text().splitlines() if l.strip()]
    rng = random.Random(seed)
    rng.shuffle(all_lines)
    pronoun_set = set(ENGLISH_PRONOUNS)
    picked: list[str] = []
    for line in all_lines:
        words = [w.strip(".,!?;:\"'()[]{}").lower() for w in line.split()]
        if any(w in pronoun_set for w in words):
            picked.append(line)
            if len(picked) >= n_samples:
                break
    print(f"  Picked {len(picked)} pronoun-containing sentences")

    # Reference cipher tokenizations
    ref_seqs: list[list[int]] = []
    for sent in picked:
        ids = alien_tok(
            cipher.encode_sentence(sent).strip(),
            add_special_tokens=True, max_length=max_len + 1, truncation=True,
        )["input_ids"]
        ref_seqs.append(ids)

    # Generate from same seeds (first 2 cipher tokens)
    print(f"\nGenerating from {len(picked)} sentences (seed_len=2, temp=1.0)...")
    pad_id = alien_tok.pad_token_id or 0
    eos_id = alien_tok.eos_token_id
    seeds = [seq[:2] for seq in ref_seqs]
    max_seed = max(len(s) for s in seeds)
    seed_tensor = torch.full((len(seeds), max_seed), pad_id, dtype=torch.long)
    for i, s in enumerate(seeds):
        seed_tensor[i, : len(s)] = torch.tensor(s, dtype=torch.long)

    bs = 16 if device.type == "cuda" else 8
    gen_seqs: list[list[int]] = []
    for i in range(0, len(seed_tensor), bs):
        chunk = seed_tensor[i : i + bs]
        out = generate_samples(model, chunk, max_len, eos_id, 1.0, device)
        gen_seqs.extend(out)
        print(f"  generated {min(i + bs, len(seed_tensor))}/{len(seed_tensor)}")

    ref_stats = analyze_anaphoric_positions(ref_seqs, pronoun_ids, alien_tok)
    gen_stats = analyze_anaphoric_positions(gen_seqs, pronoun_ids, alien_tok)

    print("\n" + "-" * 60)
    print(f"{'metric':<32s} {'reference':>14s} {'generated':>14s}")
    print("-" * 60)
    for k in (
        "n_sequences", "n_with_pronoun", "n_pronoun_occurrences", "occurrences_per_seq",
        "mean_abs_pos", "mean_rel_pos", "with_prior_content_pct",
        "mean_dist_back", "median_dist_back",
    ):
        rv, gv = ref_stats[k], gen_stats[k]
        if isinstance(rv, float):
            print(f"{k:<32s} {rv:>14.4f} {gv:>14.4f}")
        else:
            print(f"{k:<32s} {rv:>14d} {gv:>14d}")

    print("\nInterpretation:")
    print("  - 'occurrences_per_seq' similar → LM produces pronoun-like tokens at similar rate.")
    print("  - 'mean_rel_pos' similar      → pronouns sit at similar relative positions.")
    print("  - 'with_prior_content_pct'    → fraction of pronouns with a preceding content-like")
    print("                                   (long-decoded) token. High in both = the LM has")
    print("                                   learned the antecedent-then-pronoun positional grammar.")
    print("  - 'mean_dist_back' similar    → typical distance back to the antecedent matches.")


# ── main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("config", help="YAML config")
    parser.add_argument("--checkpoint", required=True, help="Phase 1 checkpoint .pt")
    parser.add_argument("--n-samples", type=int, default=200)
    parser.add_argument("--max-len", type=int, default=64)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--skip-layer1", action="store_true")
    parser.add_argument("--skip-layer2", action="store_true")
    args = parser.parse_args()

    cfg = SynthConfig(**yaml.safe_load(Path(args.config).read_text()))
    device = torch.device(args.device)
    torch.manual_seed(args.seed)

    if not args.skip_layer1:
        layer1_probe(cfg.base_model_name, device)
    if not args.skip_layer2:
        layer2_probe(cfg, args.checkpoint, args.n_samples, args.max_len, device, args.seed)


if __name__ == "__main__":
    main()

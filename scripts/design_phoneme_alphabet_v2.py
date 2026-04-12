#!/usr/bin/env python
"""Design a phoneme alphabet by mining Qwen's full vocabulary for rare,
stable, semantically-neutral tokens — across all scripts, not just ASCII.

v2 difference from v1: scans all ~151K vocab tokens instead of enumerating
a-z pairs, so rare tokens in Cyrillic / Greek / CJK / misc scripts can
surface naturally alongside rare ASCII combos.

Rarity is triangulated across three signals, since no single signal
covers all of Qwen's training distribution:

  1. **prose_rate**  — frequency in english_corpus.txt (catches prose morphemes)
  2. **code_rate**   — frequency in stack-smol-xl.jsonl  (catches code tokens)
  3. **out_norm**    — L2 norm of the LM-head output weight for that token
                       — proxy for how often Qwen was trained to emit it across
                       all pretraining languages.  Rarely-used tokens have
                       small norms from limited gradient flow.

A candidate must be below percentile cutoffs on all three signals to
qualify as "rare everywhere we can measure".

We still require concat stability (BPE boundary preservation when
phonemes are concatenated) and filter against known English words.
"""

from __future__ import annotations

import itertools
import json
from collections import Counter
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_NAME = "Qwen/Qwen2.5-0.5B"
OUTPUT = Path("data/phoneme_alphabet_v2.json")
TARGET_SIZE = 60
SEED = 42

# Phonemes must be between these lengths (chars), stripped of space prefix.
MIN_LEN = 2
MAX_LEN = 5

# Stability threshold — min fraction of test partners that preserve boundaries.
STABILITY_THRESHOLD = 0.85

# Frequency sources.
FREQ_PROSE_PATH = Path("data/translator/english_corpus.txt")
FREQ_CODE_PATH = Path("data/qwen_targets_cache/stack-smol-xl.jsonl")
FREQ_PROSE_LINES = 100_000
FREQ_CODE_LINES = 20_000

# Candidate must score below this percentile on each of the three rarity
# signals (prose_rate, code_rate, lm_head_norm).
RARITY_PERCENTILE = 0.40

# Diversity: cap per-first-char to avoid alphabet concentration.
PER_FIRST_CHAR_CAP = 8

# Partners to test stability against.
STABILITY_PARTNERS = 80


def enumerate_vocab_candidates(tok) -> list[tuple[int, str]]:
    """Return (token_id, bare_string) for vocab tokens usable as phonemes.

    Requirements:
      - The token is a space-prefixed BPE unit (starts with 'Ġ' in Qwen).
      - The bare (non-space-prefixed) form also exists as a single token,
        so the phoneme works both word-initially and internally.
      - Bare string length in [MIN_LEN, MAX_LEN].
      - No whitespace or control chars in the bare string.
    """
    vocab = tok.get_vocab()  # {str: int}
    inv = {v: k for k, v in vocab.items()}

    candidates: list[tuple[int, str]] = []
    for tok_str, tid in vocab.items():
        if not tok_str.startswith("Ġ"):
            continue
        bare = tok_str[1:]
        if not (MIN_LEN <= len(bare) <= MAX_LEN):
            continue
        if any(ch.isspace() or not ch.isprintable() for ch in bare):
            continue
        # The bare form must also exist in vocab as a standalone token
        if bare not in vocab:
            continue
        candidates.append((tid, bare))
    return candidates


def count_corpus_tokens(
    tok, path: Path, max_lines: int, is_jsonl: bool,
) -> Counter[int]:
    c: Counter[int] = Counter()
    with path.open() as f:
        for i, line in enumerate(f):
            if i >= max_lines:
                break
            text = line.strip()
            if is_jsonl:
                try:
                    text = json.loads(text).get("text", "")
                except json.JSONDecodeError:
                    continue
            if not text:
                continue
            if len(text) > 4000:
                text = text[:4000]
            ids = tok.encode(text, add_special_tokens=False)
            c.update(ids)
    return c


def measure_stability(
    tok, candidates: list[tuple[int, str]], partners: list[str],
) -> list[tuple[int, str, float, float]]:
    """For each candidate (tid, bare), test left/right stability.

    left_stable: ' A+B' tokenizes with leading token == space-prefixed A.
    right_stable: ' B+A' tokenizes as exactly two tokens: [Ġ+B, A].
    """
    # Pre-compute partner ids.
    partner_ids = [
        tok.encode(" " + b, add_special_tokens=False) for b in partners
    ]
    # Filter partners that themselves tokenize to a single token.
    partners_clean = [
        (b, ids[0]) for b, ids in zip(partners, partner_ids) if len(ids) == 1
    ]

    out: list[tuple[int, str, float, float]] = []
    n = len(partners_clean)
    for tid, bare in candidates:
        ls_ok = 0
        rs_ok = 0
        for b, b_id in partners_clean:
            # Left: " A+B" should have first token = " A" (id=tid)
            ids = tok.encode(" " + bare + b, add_special_tokens=False)
            if len(ids) >= 2 and ids[0] == tid:
                ls_ok += 1
            # Right: " B+A" should be exactly [Ġ+B, A]
            ids2 = tok.encode(" " + b + bare, add_special_tokens=False)
            if len(ids2) == 2 and ids2[0] == b_id:
                toks = tok.convert_ids_to_tokens(ids2)
                if toks[1] == bare:
                    rs_ok += 1
        out.append((tid, bare, ls_ok / n, rs_ok / n))
    return out


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading {MODEL_NAME} on {device}...")
    tok = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, torch_dtype=torch.float32,
    ).to(device).eval()

    print("\n[1] Enumerating vocab phoneme candidates...")
    candidates = enumerate_vocab_candidates(tok)
    print(f"  {len(candidates)} candidates (space-prefixed, len {MIN_LEN}-{MAX_LEN},")
    print(f"  both space-prefixed and bare forms in vocab)")
    # Sample distribution by length
    by_len = Counter(len(b) for _, b in candidates)
    print(f"  by length: {dict(sorted(by_len.items()))}")
    # Sample first chars to get a sense
    first_chars = Counter(b[0] for _, b in candidates)
    print(f"  top-10 first chars: {first_chars.most_common(10)}")

    print("\n[2] Computing rarity signals...")

    # (a) prose + code frequencies
    print(f"  counting tokens in {FREQ_PROSE_PATH} (≤{FREQ_PROSE_LINES} lines)...")
    prose_counts = count_corpus_tokens(tok, FREQ_PROSE_PATH, FREQ_PROSE_LINES, False)
    prose_total = max(sum(prose_counts.values()), 1)
    print(f"    prose total: {prose_total:,} tokens")

    print(f"  counting tokens in {FREQ_CODE_PATH} (≤{FREQ_CODE_LINES} lines)...")
    code_counts = count_corpus_tokens(tok, FREQ_CODE_PATH, FREQ_CODE_LINES, True)
    code_total = max(sum(code_counts.values()), 1)
    print(f"    code total: {code_total:,} tokens")

    # (b) LM-head output weight norms — proxy for pretraining-distribution
    # frequency across all languages
    print("  computing LM-head output-weight L2 norms for all vocab tokens...")
    with torch.no_grad():
        out_weight = model.get_output_embeddings().weight  # (vocab, hidden)
        all_out_norms = out_weight.norm(dim=-1).cpu().numpy()
    print(f"    out_norm: min={all_out_norms.min():.3f} "
          f"median={float(torch.tensor(all_out_norms).median()):.3f} "
          f"max={all_out_norms.max():.3f}")

    # Assemble per-candidate scores
    print("\n[3] Scoring and filtering candidates...")
    scores: list[dict] = []
    for tid, bare in candidates:
        pr = prose_counts.get(tid, 0) / prose_total
        cr = code_counts.get(tid, 0) / code_total
        on = float(all_out_norms[tid])
        scores.append({
            "tid": tid, "bare": bare,
            "prose_rate": pr, "code_rate": cr, "out_norm": on,
        })

    # Compute percentile cutoffs
    def percentile_cutoff(values: list[float], pct: float) -> float:
        vs = sorted(values)
        idx = min(int(len(vs) * pct), len(vs) - 1)
        return vs[idx]

    cut_prose = percentile_cutoff([s["prose_rate"] for s in scores], RARITY_PERCENTILE)
    cut_code = percentile_cutoff([s["code_rate"] for s in scores], RARITY_PERCENTILE)
    cut_norm = percentile_cutoff([s["out_norm"] for s in scores], RARITY_PERCENTILE)
    print(f"  cutoffs @ pct={RARITY_PERCENTILE}:")
    print(f"    prose_rate ≤ {cut_prose*1e6:.2f}/M")
    print(f"    code_rate  ≤ {cut_code*1e6:.2f}/M")
    print(f"    out_norm   ≤ {cut_norm:.3f}")

    rare = [
        s for s in scores
        if s["prose_rate"] <= cut_prose
        and s["code_rate"] <= cut_code
        and s["out_norm"] <= cut_norm
    ]
    print(f"  {len(rare)} candidates pass ALL three rarity cuts")

    # Show example rare candidates across scripts
    print("  sample rare candidates:")
    for s in rare[:20]:
        print(f"    {s['bare']!r:>8}  "
              f"prose={s['prose_rate']*1e6:>7.2f}/M  "
              f"code={s['code_rate']*1e6:>7.2f}/M  "
              f"norm={s['out_norm']:.2f}")

    # --------------------------------------------------------------
    # [4] Concat stability — same test as before, applied to rare pool.
    # --------------------------------------------------------------
    print(f"\n[4] Testing concat stability (threshold={STABILITY_THRESHOLD})...")
    # Use a sample of rare candidates themselves as partners — this
    # tests "does phoneme A preserve its boundary when concatenated with
    # another rare phoneme B?", which is the actual usage pattern.
    partners = [s["bare"] for s in rare[:STABILITY_PARTNERS]]
    pair_candidates = [(s["tid"], s["bare"]) for s in rare]
    stab = measure_stability(tok, pair_candidates, partners)

    stable = [
        (tid, bare, ls, rs) for tid, bare, ls, rs in stab
        if ls >= STABILITY_THRESHOLD and rs >= STABILITY_THRESHOLD
    ]
    stable.sort(key=lambda x: -(x[2] * x[3]))
    print(f"  {len(stable)} candidates pass stability (from {len(rare)} rare)")
    print("  top-15 by stability product:")
    for tid, bare, ls, rs in stable[:15]:
        print(f"    {bare!r:>8}: left={ls:.2f} right={rs:.2f}  prod={ls*rs:.2f}")

    # --------------------------------------------------------------
    # [5] Diversity selection — prefer varied first characters.
    # --------------------------------------------------------------
    print(f"\n[5] Selecting diverse inventory (target={TARGET_SIZE}, "
          f"cap={PER_FIRST_CHAR_CAP} per first-char)...")
    selected: list[tuple[int, str]] = []
    first_counts: Counter[str] = Counter()
    for tid, bare, ls, rs in stable:
        first = bare[0]
        if first_counts[first] >= PER_FIRST_CHAR_CAP:
            continue
        selected.append((tid, bare))
        first_counts[first] += 1
        if len(selected) >= TARGET_SIZE:
            break

    selected_bare = [b for _, b in selected]
    print(f"  Selected {len(selected)} phonemes.")
    print(f"  Length distribution: {dict(sorted(Counter(len(b) for b in selected_bare).items()))}")
    print(f"  First-char distribution: {dict(sorted(first_counts.items()))}")
    # Script mix
    def script_of(s: str) -> str:
        c = s[0]
        if c.isascii() and c.isalpha():
            return "ASCII"
        if "\u0400" <= c <= "\u04FF":
            return "Cyrillic"
        if "\u0370" <= c <= "\u03FF":
            return "Greek"
        if "\u4E00" <= c <= "\u9FFF":
            return "CJK"
        if "\u3040" <= c <= "\u30FF":
            return "Kana"
        if "\u0530" <= c <= "\u058F":
            return "Armenian"
        if "\u10A0" <= c <= "\u10FF":
            return "Georgian"
        if "\u0590" <= c <= "\u05FF":
            return "Hebrew"
        if "\u0600" <= c <= "\u06FF":
            return "Arabic"
        if "\u0900" <= c <= "\u097F":
            return "Devanagari"
        return "Other"
    script_counts = Counter(script_of(b) for b in selected_bare)
    print(f"  Script distribution: {dict(script_counts)}")
    print(f"  Phonemes:")
    for b in selected_bare:
        print(f"    {b!r}")

    # --------------------------------------------------------------
    # [6] Validation — 3-phoneme word tokenization stability
    # --------------------------------------------------------------
    import random
    print("\n[6] Validating 3-phoneme word tokenization...")
    random.seed(SEED)
    sample_words = [
        "".join(random.choice(selected_bare) for _ in range(3))
        for _ in range(5000)
    ]
    word_to_ids: dict[str, tuple[int, ...]] = {}
    token_lengths: list[int] = []
    for w in sample_words:
        if w in word_to_ids:
            continue
        ids = tuple(tok.encode(" " + w, add_special_tokens=False))
        word_to_ids[w] = ids
        token_lengths.append(len(ids))
    print(f"  Unique words: {len(word_to_ids)}")
    lc = Counter(token_lengths)
    for k in sorted(lc):
        print(f"    {k} tokens: {lc[k]} ({100*lc[k]/len(token_lengths):.1f}%)")

    # Context stability
    test_words = list(word_to_ids.keys())[:200]
    pref1 = " the "
    pref2 = " a very "
    stable_ctx = 0
    for w in test_words:
        ids1 = tok.encode(pref1 + w + " ends here.", add_special_tokens=False)
        ids2 = tok.encode(pref2 + w + " appears!", add_special_tokens=False)
        expected = word_to_ids[w]
        def contains(seq, sub):
            for i in range(len(seq) - len(sub) + 1):
                if tuple(seq[i:i+len(sub)]) == sub:
                    return True
            return False
        if contains(ids1, expected) and contains(ids2, expected):
            stable_ctx += 1
    print(f"  Context stability: {stable_ctx}/{len(test_words)}")

    # --------------------------------------------------------------
    # [7] Save
    # --------------------------------------------------------------
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    alphabet = {
        "version": "v2",
        "model": MODEL_NAME,
        "method": "vocab-wide rare-token scan "
                  "(prose_rate + code_rate + lm_head_norm)",
        "size": len(selected_bare),
        "phonemes": selected_bare,
        "rarity_percentile": RARITY_PERCENTILE,
        "stability_threshold": STABILITY_THRESHOLD,
        "script_distribution": dict(script_counts),
        "first_char_distribution": dict(first_counts),
        "length_distribution": {
            str(k): v for k, v in Counter(len(b) for b in selected_bare).items()
        },
        "validation": {
            "sample_size": len(word_to_ids),
            "token_length_distribution": {str(k): v for k, v in lc.items()},
            "context_stable_fraction": stable_ctx / len(test_words),
        },
    }
    with OUTPUT.open("w") as f:
        json.dump(alphabet, f, indent=2, ensure_ascii=False)
    print(f"\nWrote {OUTPUT}")


if __name__ == "__main__":
    main()

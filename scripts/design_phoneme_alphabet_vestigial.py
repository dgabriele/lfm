#!/usr/bin/env python
"""Design a phoneme alphabet from Qwen's most vestigial but stable tokens.

"Vestigial" means: the token exists in Qwen's vocabulary but received
minimal gradient during pretraining — rarely produced, rarely used.
We identify such tokens directly via the L2 norm of their LM-head output
row: tokens Qwen was trained to rarely emit have small output norms.

The hypothesis underpinning this script: if we fine-tune on Neuroglot
using tokens Qwen barely used during pretraining, then

  (a) no capability Qwen has is meaningfully destroyed (there's little
      learned function at those embeddings to damage),
  (b) the token embeddings can be reassigned to Neuroglot meaning with
      minimal interference with everything else Qwen knows,
  (c) since Qwen was still exposed to these tokens in SOME context, its
      language-processing machinery still activates when it reads them
      — we retain architectural leverage for semantic processing.

We additionally require:

  - **Latin-script word-like form** (no punctuation/symbols): keeps Qwen
    in language-mode when reading Neuroglot rather than code-parsing mode.
  - **BPE concat stability**: deterministic tokenization under
    concatenation into multi-phoneme words.
  - **Rarity in our English+code corpora**: secondary safety check that
    these tokens aren't actually common in English despite low norm
    (defensive, since norm is a proxy).
"""

from __future__ import annotations

import json
import random
import unicodedata
from collections import Counter
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_NAME = "Qwen/Qwen2.5-0.5B"
OUTPUT = Path("data/phoneme_alphabet_vestigial.json")
TARGET_SIZE = 50
SEED = 42

MIN_LEN = 2
MAX_LEN = 5

# Vestigiality — bottom X% of vocab by LM-head output-weight L2 norm.
VESTIGIAL_PERCENTILE = 0.15

# Stability — min fraction of partners preserving BPE boundaries.
STABILITY_THRESHOLD = 0.85
STABILITY_PARTNERS = 60

# Defensive rarity check against English+code corpora (catches any
# high-frequency token mis-flagged by norm).
FREQ_PROSE_PATH = Path("data/translator/english_corpus.txt")
FREQ_CODE_PATH = Path("data/qwen_targets_cache/stack-smol-xl.jsonl")
FREQ_PROSE_LINES = 50_000
FREQ_CODE_LINES = 10_000
MAX_RATE_PER_MILLION = 1.0  # drop anything above 1 occurrence per million

# Diversity
PER_FIRST_CHAR_CAP = 6


def is_latin_wordlike(s: str) -> bool:
    """True if every char is a Latin letter (allowing common diacritics)."""
    if not s:
        return False
    for ch in s:
        cat = unicodedata.category(ch)
        # Ll=lowercase letter, Lu=uppercase, Lt=titlecase, Lo=other letter,
        # Lm=modifier letter, Mn=combining mark
        if cat[0] != "L":
            return False
        # Restrict script to Latin to keep Qwen in language mode (vs Greek/
        # Cyrillic triggering a different script-specific prior pathway).
        try:
            name = unicodedata.name(ch, "")
        except ValueError:
            return False
        if "LATIN" not in name:
            return False
    return True


def enumerate_candidates(tok) -> list[tuple[int, str]]:
    """Return (tid, bare) for Latin word-like Qwen tokens, both space-prefixed
    and bare forms present in vocab."""
    vocab = tok.get_vocab()
    cands: list[tuple[int, str]] = []
    for tok_str, tid in vocab.items():
        if not tok_str.startswith("Ġ"):
            continue
        bare = tok_str[1:]
        if not (MIN_LEN <= len(bare) <= MAX_LEN):
            continue
        if not is_latin_wordlike(bare):
            continue
        if bare not in vocab:
            continue
        cands.append((tid, bare))
    return cands


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
    """For each candidate (tid, bare), test left/right stability."""
    partner_ids = [
        tok.encode(" " + b, add_special_tokens=False) for b in partners
    ]
    partners_clean = [
        (b, ids[0]) for b, ids in zip(partners, partner_ids) if len(ids) == 1
    ]
    out: list[tuple[int, str, float, float]] = []
    n = len(partners_clean)
    for tid, bare in candidates:
        ls_ok = 0
        rs_ok = 0
        for b, b_id in partners_clean:
            ids = tok.encode(" " + bare + b, add_special_tokens=False)
            if len(ids) >= 2 and ids[0] == tid:
                ls_ok += 1
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

    print("\n[1] Enumerating Latin word-like vocab candidates...")
    candidates = enumerate_candidates(tok)
    print(f"  {len(candidates)} candidates (Latin-script, len {MIN_LEN}-{MAX_LEN})")
    by_len = Counter(len(b) for _, b in candidates)
    print(f"  by length: {dict(sorted(by_len.items()))}")

    print("\n[2] Computing LM-head output-weight norms...")
    with torch.no_grad():
        out_w = model.get_output_embeddings().weight  # (vocab, hidden)
        all_norms = out_w.norm(dim=-1).cpu().numpy()
    vocab_size = len(all_norms)
    print(f"  vocab={vocab_size} tokens. "
          f"norm range [{all_norms.min():.3f}, {all_norms.max():.3f}], "
          f"median={float(torch.tensor(all_norms).median()):.3f}")

    # Cutoff computed over ALL vocab (vestigiality is a whole-model property)
    sorted_all = sorted(all_norms)
    cutoff_all = sorted_all[int(vocab_size * VESTIGIAL_PERCENTILE)]
    print(f"  vestigial cutoff @ bottom {VESTIGIAL_PERCENTILE*100:.0f}%: "
          f"norm ≤ {cutoff_all:.3f}")

    # Apply cutoff to our Latin word-like candidates
    vestigial = [
        (tid, bare, float(all_norms[tid])) for tid, bare in candidates
        if all_norms[tid] <= cutoff_all
    ]
    vestigial.sort(key=lambda x: x[2])
    print(f"  {len(vestigial)} Latin candidates within vestigial tail")

    print("  example vestigial tokens (lowest norm first):")
    for tid, bare, n in vestigial[:20]:
        print(f"    {bare!r:>7}  norm={n:.3f}")

    print(f"\n[3] Defensive rarity check (English prose + Stack code)...")
    prose = count_corpus_tokens(tok, FREQ_PROSE_PATH, FREQ_PROSE_LINES, False)
    code = count_corpus_tokens(tok, FREQ_CODE_PATH, FREQ_CODE_LINES, True)
    prose_total = max(sum(prose.values()), 1)
    code_total = max(sum(code.values()), 1)
    print(f"  prose tokens: {prose_total:,}   code tokens: {code_total:,}")

    threshold = MAX_RATE_PER_MILLION / 1e6
    surviving = []
    flagged = []
    for tid, bare, n in vestigial:
        pr = prose.get(tid, 0) / prose_total
        cr = code.get(tid, 0) / code_total
        if max(pr, cr) <= threshold:
            surviving.append((tid, bare, n, pr, cr))
        else:
            flagged.append((tid, bare, pr, cr))
    print(f"  {len(vestigial)} → {len(surviving)} after dropping common-in-English/code")
    if flagged:
        flagged.sort(key=lambda x: -max(x[2], x[3]))
        print("  flagged (vestigial but common somewhere — mostly loanwords/names):")
        for tid, bare, pr, cr in flagged[:8]:
            print(f"    {bare!r:>7}  prose={pr*1e6:.1f}/M  code={cr*1e6:.1f}/M")

    print(f"\n[4] Concat stability test (threshold={STABILITY_THRESHOLD})...")
    partners = [b for _, b, _, _, _ in surviving[:STABILITY_PARTNERS]]
    if len(partners) < 10:
        print("  !!! Too few candidates for meaningful stability test. "
              "Consider relaxing VESTIGIAL_PERCENTILE.")
        return
    pair_candidates = [(tid, bare) for tid, bare, _, _, _ in surviving]
    stab = measure_stability(tok, pair_candidates, partners)
    stable = [
        (tid, bare, ls, rs) for tid, bare, ls, rs in stab
        if ls >= STABILITY_THRESHOLD and rs >= STABILITY_THRESHOLD
    ]
    stable.sort(key=lambda x: -(x[2] * x[3]))
    print(f"  {len(stable)} / {len(surviving)} pass stability")
    # Look up norm for sorting/display
    norm_lookup = {tid: n for tid, _, n, _, _ in surviving}
    print("  top-15 stable (shown with norm):")
    for tid, bare, ls, rs in stable[:15]:
        print(f"    {bare!r:>7}: stab={ls*rs:.2f}  norm={norm_lookup[tid]:.3f}")

    print(f"\n[5] Selecting diverse inventory (target={TARGET_SIZE}, "
          f"cap={PER_FIRST_CHAR_CAP} per first-char)...")
    selected: list[tuple[int, str, float]] = []
    first_counts: Counter[str] = Counter()
    # Sort by norm ascending (most vestigial first) among stable ones
    stable_by_norm = sorted(
        stable, key=lambda x: norm_lookup[x[0]],
    )
    for tid, bare, ls, rs in stable_by_norm:
        fc = bare[0].lower()
        if first_counts[fc] >= PER_FIRST_CHAR_CAP:
            continue
        selected.append((tid, bare, norm_lookup[tid]))
        first_counts[fc] += 1
        if len(selected) >= TARGET_SIZE:
            break

    selected_bare = [b for _, b, _ in selected]
    selected_norms = [n for _, _, n in selected]
    print(f"  Selected {len(selected_bare)} phonemes.")
    print(f"  Norm range: [{min(selected_norms):.3f}, {max(selected_norms):.3f}]")
    print(f"  Length dist: {dict(sorted(Counter(len(b) for b in selected_bare).items()))}")
    print(f"  First-char (lc): {dict(sorted(first_counts.items()))}")
    print("  Phonemes:")
    for tid, bare, n in selected:
        print(f"    {bare!r:>7}  tid={tid:>6}  norm={n:.3f}")

    # --------------------------------------------------------------
    # [6] Validation — 3-phoneme word tokenization stability
    # --------------------------------------------------------------
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

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    alphabet = {
        "version": "vestigial-v1",
        "model": MODEL_NAME,
        "method": "Latin-script word-like tokens, bottom-"
                  f"{VESTIGIAL_PERCENTILE*100:.0f}% by LM-head output norm, "
                  "filtered by English+code rarity, concat-stable",
        "size": len(selected_bare),
        "phonemes": selected_bare,
        "vestigial_percentile": VESTIGIAL_PERCENTILE,
        "norm_range": [min(selected_norms), max(selected_norms)],
        "stability_threshold": STABILITY_THRESHOLD,
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

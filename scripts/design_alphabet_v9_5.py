#!/usr/bin/env python
"""Design v9.5 alphabet — Qwen BPE pidgin with productive morphology.

v9.5 vs v9:
  - v9 restricted the alphabet to Ġ-prefixed (word-start) Qwen BPE tokens
    only, guaranteeing that any concatenation of agent-emitted tokens
    re-tokenizes back to itself under Qwen.  The cost was that the agent
    couldn't compose stems with suffixes (no `tastily`, `restful`,
    `consciousness` from morpheme parts) — every emission started a new
    word.
  - v9.5 admits BOTH Ġ-prefixed AND bare (word-internal) BPE tokens.
    Each token carries an `is_word_start` flag.  At render time:
        Ġ-prefixed tokens prepend a space (they start a new word)
        bare tokens concatenate directly to the previous (mid-word continuation)
    The agent gains productive morphology: [Ġtaste][ily] → "tastily",
    [Ġortho][graph][y] → "orthography", [Ġun][stop][able] → "unstoppable".

  Trade-off: when Qwen re-reads the rendered string, its greedy BPE
  re-splits it differently from the agent's intended boundaries (e.g.
  "tastily" → [t, ast, ily]).  But the SPELLING is preserved, which is
  what the downstream LLM consumes — boundaries are agent-internal.

Selection: same scoring as v9 (log freq + 0.6 × log productivity); now
both Ġ-prefixed and bare candidates compete in one pool.  Target ~10K
total (≈ 5K Ġ + 5K bare in practice).
"""

from __future__ import annotations

import json
import math
import unicodedata
from collections import Counter
from pathlib import Path

from transformers import AutoTokenizer

MODEL = "Qwen/Qwen2.5-0.5B"
ENGLISH_CORPUS = Path("data/translator/english_corpus.txt")
OUTPUT = Path("data/phoneme_alphabet_v9_5.json")

TARGET_SIZE = 50000
SCAN_LINES = 200_000
MIN_LEN = 1
MAX_LEN = 7
PRODUCTIVITY_WEIGHT = 0.6

def is_alpha_ascii_lower(s: str) -> bool:
    """True iff every char is a-z.  Enforces lowercase + ASCII-only:
      - Rejects punctuation (belongs to the rendering layer, not the
        agent's expressive vocabulary).
      - Rejects uppercase variants (orthographic artefact: sentence
        starts and acronyms don't belong in a phonetic alphabet).
      - Rejects Qwen's byte-level BPE mojibake tokens like ``âĢľ``
        ``âĢĻs`` (curly quotes encoded as UTF-8 bytes) — these are
        valid Latin chars by Unicode but aren't real alphabet units.
    """
    if not s:
        return False
    return all("a" <= c <= "z" for c in s)


def main() -> None:
    print(f"loading tokenizer: {MODEL}")
    tok = AutoTokenizer.from_pretrained(MODEL)

    print(f"counting Qwen BPE token frequencies in {ENGLISH_CORPUS} "
          f"(<= {SCAN_LINES} lines)...")
    counts: Counter[int] = Counter()
    n_lines = 0
    with ENGLISH_CORPUS.open() as f:
        for i, line in enumerate(f):
            if i >= SCAN_LINES:
                break
            text = line.strip()
            if not text:
                continue
            ids = tok.encode(text, add_special_tokens=False)
            counts.update(ids)
            n_lines += 1
    total_text_tokens = sum(counts.values())
    print(f"  scanned {n_lines:,} lines, {total_text_tokens:,} tokens, "
          f"{len(counts):,} unique types")

    # Productivity index: count vocab tokens that strictly extend each candidate.
    print("computing productivity index...")
    vocab = tok.get_vocab()
    bare_forms_all: list[str] = []
    for token_str in vocab:
        bare_forms_all.append(token_str[1:] if token_str.startswith("Ġ") else token_str)
    by_prefix: dict[str, list[str]] = {}
    for bf in bare_forms_all:
        if len(bf) >= 2:
            by_prefix.setdefault(bf[:2], []).append(bf)

    def productivity(bare: str) -> int:
        bucket = by_prefix.get(bare[:2], [])
        return sum(1 for vs in bucket if len(vs) > len(bare) and vs.startswith(bare))

    # Score candidates, collapsing case variants.  Each unique (lowered
    # bare, is_word_start) entry accumulates frequency across all its
    # case variants ("The"+"the"+"THE" → one entry).  The one-to-many
    # Qwen-id → alphabet-idx mapping is handled at transcode time.
    print("scoring candidates (lowercased + case-collapsed)...")
    grouped: dict[tuple[str, bool], dict] = {}
    skipped = Counter()
    for token_str, tid in vocab.items():
        if token_str.startswith("Ġ"):
            raw = token_str[1:]
            is_word_start = True
        else:
            raw = token_str
            is_word_start = False
        bare = raw.lower()
        if not (MIN_LEN <= len(bare) <= MAX_LEN):
            skipped["len"] += 1
            continue
        if not is_alpha_ascii_lower(bare):
            skipped["nonletter"] += 1
            continue
        freq = counts.get(tid, 0)
        key = (bare, is_word_start)
        if key not in grouped:
            grouped[key] = {"qwen_ids": [], "freq": 0}
        grouped[key]["qwen_ids"].append(tid)
        grouped[key]["freq"] += freq

    scored: list[tuple[float, int, str, bool, int, int]] = []
    for (bare, is_word_start), entry in grouped.items():
        if entry["freq"] < 3:
            skipped["lowfreq"] += 1
            continue
        prod = productivity(bare)
        score = math.log1p(entry["freq"]) + PRODUCTIVITY_WEIGHT * math.log1p(prod)
        # Representative qwen id = the most-frequent variant (first in list
        # approximately; exact choice doesn't matter — mapping is many-to-one).
        rep_tid = entry["qwen_ids"][0]
        scored.append((score, rep_tid, bare, is_word_start, entry["freq"], prod))
    print(f"  unique (case-collapsed) candidates: {len(scored):,}  skipped: {dict(skipped)}")

    scored.sort(key=lambda x: -x[0])
    selected = scored[:TARGET_SIZE]

    n_ws = sum(1 for s in selected if s[3])
    n_bare = len(selected) - n_ws
    cumulative = sum(s[4] for s in selected)
    print(
        f"  selected {len(selected):,}  ({n_ws:,} Ġ-prefixed + {n_bare:,} bare)  "
        f"covering {cumulative/total_text_tokens:.1%} of English text",
    )

    # Diagnostics: show some bare suffix tokens that made it in.
    print("\nsample bare (continuation) tokens in alphabet:")
    bare_in = [(s[2], s[4], s[5]) for s in selected if not s[3]]
    bare_in.sort(key=lambda x: -x[1])
    for b, freq, prod in bare_in[:20]:
        print(f"  {b!r:<12}  freq={freq:>8,}  prod={prod}")

    print("\nsample Ġ-prefixed (word-start) tokens (top 20):")
    ws_in = [(s[2], s[4], s[5]) for s in selected if s[3]]
    ws_in.sort(key=lambda x: -x[1])
    for b, freq, prod in ws_in[:20]:
        print(f"  {b!r:<12}  freq={freq:>8,}  prod={prod}")

    # Show a rendering example for productive morphology.
    by_bare = {(s[2], s[3]): s[1] for s in selected}
    print("\nproductive morphology check — can the agent compose these?")
    examples = [
        ("tastily", [("Ġtaste", True), ("ily", False)]),
        ("orthography", [("Ġortho", True), ("graph", False), ("y", False)]),
        ("consciousness", [("Ġconscious", True), ("ness", False)]),
        ("unstoppable", [("Ġun", True), ("stop", False), ("able", False)]),
        ("restful", [("Ġrest", True), ("ful", False)]),
    ]
    for word, parts in examples:
        bare_parts = [p[0][1:] if p[1] else p[0] for p in parts]
        reachable = all((b, ws) in by_bare for b, ws in zip(bare_parts, [p[1] for p in parts]))
        marker = "✓" if reachable else "✗"
        rendered = "".join(
            (" " + b if ws else b) for b, ws in zip(bare_parts, [p[1] for p in parts])
        ).strip()
        print(f"  {marker} {word!r:>16}  via {bare_parts}  →  {rendered!r}")

    # Build many-to-one map: every Qwen id (across case variants) that
    # collapsed into a selected alphabet entry points at that entry's index.
    qwen_to_alphabet: dict[int, int] = {}
    for alphabet_idx, s in enumerate(selected):
        bare, is_word_start = s[2], s[3]
        for qid in grouped[(bare, is_word_start)]["qwen_ids"]:
            qwen_to_alphabet[qid] = alphabet_idx

    # Write artifact.
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    alphabet = {
        "version": "v9.5-english-pidgin-morphology",
        "model": MODEL,
        "method": (
            "Top-N Qwen BPE tokens (Ġ-prefixed + bare), lowercased and "
            "ASCII-only, with case variants collapsed to a single entry.  "
            "Bare tokens enable productive morphological composition "
            "(stem + suffix → real word) at the cost of dropping the strict "
            "Qwen round-trip guarantee."
        ),
        "size": len(selected),
        "phonemes": [s[2] for s in selected],
        "qwen_token_ids": [s[1] for s in selected],       # representative (one per entry)
        "qwen_to_alphabet": qwen_to_alphabet,             # many-to-one (all case variants)
        "is_word_start": [s[3] for s in selected],
        "frequencies": {s[2] + ("[Ġ]" if s[3] else "[bare]"): s[4] for s in selected},
        "productivity": {s[2] + ("[Ġ]" if s[3] else "[bare]"): s[5] for s in selected},
        "coverage_pct": round(cumulative / total_text_tokens, 4),
        "render_mode": "qwen_subword_with_morphology",
        "stats": {
            "n_word_start": n_ws,
            "n_bare": n_bare,
            "scan_lines": n_lines,
            "total_tokens_scanned": total_text_tokens,
            "productivity_weight": PRODUCTIVITY_WEIGHT,
            "n_qwen_ids_mapped": len(qwen_to_alphabet),
        },
    }
    with OUTPUT.open("w", encoding="utf-8") as f:
        json.dump(alphabet, f, indent=2, ensure_ascii=False)
    print(f"\nwrote {OUTPUT}")


if __name__ == "__main__":
    main()

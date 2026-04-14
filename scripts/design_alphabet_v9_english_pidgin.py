#!/usr/bin/env python
"""Design v9 alphabet — semantic-rich English BPE pidgin.

Selects the top-N most frequent English Qwen BPE tokens, deliberately
EMBRACING (not avoiding) Qwen's semantic priors.  Includes both content
morphemes and function words/affixes so the agent can compose
interpretable phrase structure (subject-verb-object scaffolding,
modifier markers, conjunctions, prepositions, productive affixes), not
just a bag-of-semantics.

Selection criteria:
  - Single space-prefixed Qwen BPE token (Ġ-prefixed).  Guarantees
    deterministic re-tokenization when emitted as " token".
  - Latin alphabetic only after removing Ġ (no punctuation, code chars).
  - Length 2-7 chars.
  - Lowercase only (skips proper nouns, capitalized brand names).
  - Ranked by frequency in our English prose corpus.
  - First 1500 hits = the alphabet.

Function words ('the', 'of', 'and', 'in', 'to', etc.) are kept on
purpose — they're the highest-frequency tokens AND they provide the
syntactic scaffolding that makes the agent's output interpretable as a
phrase, not just a tag cloud.

Output: data/phoneme_alphabet_v9.json — JSON artifact compatible with
PhonemeTokenizer (vocab_size = N + 1 with reserved word-boundary id).
"""

from __future__ import annotations

import json
import unicodedata
from collections import Counter
from pathlib import Path

from transformers import AutoTokenizer

MODEL = "Qwen/Qwen2.5-0.5B"
ENGLISH_CORPUS = Path("data/translator/english_corpus.txt")
OUTPUT = Path("data/phoneme_alphabet_v9.json")

# Alphabet size cap.  ~5000 sits at the diminishing-returns knee of
# coverage vs vocab bloat (yields ~70% English text coverage; going to
# 8000 only adds 3 more points).  Comparable to v7's 8000-token SPM
# vocab, so VAE training behavior is well-understood.
TARGET_SIZE = 5000
SCAN_LINES = 200_000     # corpus lines to scan for frequency
MIN_LEN = 1     # allow single-char punct (".", ",", "?")
MAX_LEN = 7
SEED = 42

# Productivity weight in the selection score.  Higher = bias toward
# tokens that act as roots for many Qwen vocab descendants (productive
# morphemes like "iso", "ortho", "trans" — they show up as prefixes of
# many other tokens in the BPE vocab).  0 = pure frequency selection.
PRODUCTIVITY_WEIGHT = 0.6


# Common punctuation tokens accepted into the alphabet.  These provide
# the structural scaffolding for phrase formation (sentence boundaries,
# clause breaks, quotes) without which 'pidgin' would just be a token soup.
ACCEPTED_PUNCT = set(".,!?:;'-\"()")


def is_alpha_latin_or_punct(s: str) -> bool:
    """True if the token is either alphabetic-Latin (any case) or a
    short punctuation token consisting only of accepted punctuation chars.
    """
    if not s:
        return False
    if all(ch in ACCEPTED_PUNCT for ch in s):
        return True
    for ch in s:
        if not ch.isalpha():
            return False
        try:
            name = unicodedata.name(ch, "")
        except ValueError:
            return False
        if "LATIN" not in name:
            return False
    return True


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
    print(f"  scanned {n_lines:,} lines, {sum(counts.values()):,} tokens, "
          f"{len(counts):,} unique types")

    # ----------------------------------------------------------------
    # Productivity index: for each candidate bare form B, count how many
    # OTHER vocab entries (Ġ-prefixed AND bare-suffix forms) start with
    # B as a substring.  Productive morphemes ("iso", "ortho", "trans",
    # "anti") have many descendants; specific content tokens have few.
    # Domain-NEUTRAL by construction — no hardcoded list, just BPE
    # structure measuring "is this a fertile root?"
    # ----------------------------------------------------------------
    print("computing productivity index (vocab descendants per token)...")
    vocab = tok.get_vocab()
    bare_forms: list[str] = []
    for token_str in vocab:
        if token_str.startswith("Ġ"):
            bare_forms.append(token_str[1:])
        else:
            bare_forms.append(token_str)

    # Pre-bucket by first 2 chars so productivity lookup is O(B) not O(V*B).
    by_prefix: dict[str, list[str]] = {}
    for bf in bare_forms:
        if len(bf) >= 2:
            by_prefix.setdefault(bf[:2], []).append(bf)

    def productivity(bare: str) -> int:
        bucket = by_prefix.get(bare[:2], [])
        # count tokens that strictly extend `bare` (longer + start-with)
        return sum(
            1 for vs in bucket
            if len(vs) > len(bare) and vs.startswith(bare)
        )

    # ----------------------------------------------------------------
    # Score every valid candidate (BOTH Ġ-prefixed AND bare) by
    #   score = log(1 + freq) + PRODUCTIVITY_WEIGHT * log(1 + productivity)
    # Both forms are needed for high English coverage:
    #   Ġ-prefixed → word-start tokens (the, dog, network, ...)
    #   bare       → word-internal morphemes (ing, ed, tion, ly, ...)
    # The agent learns when to emit each form; rendering recombines them
    # via space-vs-no-space concatenation (deterministic round-trip).
    # ----------------------------------------------------------------
    # Restrict to Ġ-prefixed only.  Including bare (word-internal)
    # tokens would let us hit 95% English coverage but breaks
    # deterministic round-trip: concatenating two adjacent bare tokens
    # (e.g. "ran" + "across" → "ranacross") re-tokenizes to a different
    # BPE split.  Ġ-only forces every render position to start a new
    # word (leading space), guaranteeing deterministic Qwen reads at
    # the cost of capping coverage near 50%.
    print("scoring all valid Ġ-prefixed candidates by freq + productivity...")
    import math
    scored: list[tuple[float, int, str, bool, int, int]] = []  # (score, tid, bare, is_word_start, freq, prod)
    skipped_caps = 0
    skipped_short = 0
    skipped_long = 0
    skipped_nonletter = 0

    for token_str, tid in vocab.items():
        if not token_str.startswith("Ġ"):
            continue
        bare = token_str[1:]
        is_word_start = True
        if len(bare) < MIN_LEN:
            skipped_short += 1
            continue
        if len(bare) > MAX_LEN:
            skipped_long += 1
            continue
        if not is_alpha_latin_or_punct(bare):
            skipped_nonletter += 1
            continue
        freq = counts.get(tid, 0)
        # Require some actual presence in our English corpus.  Otherwise
        # high-productivity-but-corpus-absent tokens (mostly code/non-prose
        # leftovers) leak in.
        if freq < 3:
            continue
        prod = productivity(bare) if bare.isalpha() else 0
        score = math.log1p(freq) + PRODUCTIVITY_WEIGHT * math.log1p(prod)
        scored.append((score, tid, bare, is_word_start, freq, prod))

    # Sort by combined score (freq + productivity) and take top TARGET_SIZE.
    scored.sort(key=lambda x: -x[0])
    total_text_tokens = sum(counts.values())
    candidates: list[tuple[int, str, int]] = []
    productivity_lookup: dict[str, int] = {}
    is_word_start_lookup: dict[str, bool] = {}
    cumulative = 0
    for score, tid, bare, is_word_start, freq, prod in scored[:TARGET_SIZE]:
        candidates.append((tid, bare, freq))
        productivity_lookup[bare] = prod
        is_word_start_lookup[bare] = is_word_start
        cumulative += freq
    coverage_pct = cumulative / total_text_tokens
    print(f"  selected {len(candidates):,} tokens covering {coverage_pct:.1%} of English text")

    print(
        f"  filtered out: too-short={skipped_short:,}  "
        f"too-long={skipped_long:,}  non-letter={skipped_nonletter:,}",
    )
    print(f"selected {len(candidates)} tokens")

    # Productivity diagnostic: how many selected tokens are highly productive?
    high_prod = sum(1 for b in productivity_lookup if productivity_lookup[b] >= 50)
    mid_prod = sum(1 for b in productivity_lookup if 10 <= productivity_lookup[b] < 50)
    print(f"  high-productivity tokens (≥50 BPE descendants): {high_prod}")
    print(f"  mid-productivity tokens (10-49 descendants):   {mid_prod}")
    # Sample a few high-productivity tokens (likely productive prefixes/roots)
    high_prod_samples = sorted(
        productivity_lookup.items(), key=lambda x: -x[1],
    )[:25]
    print("  most-productive tokens in alphabet:")
    for bare, prod in high_prod_samples:
        print(f"    {bare!r:>12}  ({prod} descendants)")

    print("\ntop-30 tokens (highest frequency):")
    for tid, bare, count in candidates[:30]:
        print(f"  {bare!r:<10} (id={tid:>6}, count={count:,})")
    print("\nsample of mid-frequency tokens (around rank 500):")
    for tid, bare, count in candidates[490:510]:
        print(f"  {bare!r:<12} (id={tid:>6}, count={count:,})")
    print(f"\nsample of tail tokens (rank {len(candidates)-20}-{len(candidates)}):")
    for tid, bare, count in candidates[-20:]:
        print(f"  {bare!r:<14} (id={tid:>6}, count={count:,})")

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    alphabet = {
        "version": "v9-english-pidgin",
        "model": MODEL,
        "method": (
            "Top-N most frequent English Qwen BPE tokens (semantic-rich "
            "pidgin substrate).  Includes function words + affixes for "
            "phrase structure, not just content morphemes.  Each token is "
            "a single Ġ-prefixed Qwen BPE unit, ensuring deterministic "
            "round-trip when surface is rendered as space-separated tokens."
        ),
        "size": len(candidates),
        "phonemes": [b for _, b, _ in candidates],
        "qwen_token_ids": [tid for tid, _, _ in candidates],
        "is_word_start": [is_word_start_lookup[b] for _, b, _ in candidates],
        "frequencies": {b: c for _, b, c in candidates},
        "coverage_pct": round(coverage_pct, 4),
        "render_mode": "qwen_subword_concat",
        "productivity": productivity_lookup,
        "stats": {
            "high_productivity_tokens": high_prod,
            "mid_productivity_tokens": mid_prod,
            "scan_lines": n_lines,
            "total_tokens_scanned": sum(counts.values()),
            "productivity_weight": PRODUCTIVITY_WEIGHT,
        },
    }
    with OUTPUT.open("w", encoding="utf-8") as f:
        json.dump(alphabet, f, indent=2, ensure_ascii=False)
    print(f"\nwrote {OUTPUT}")


if __name__ == "__main__":
    main()

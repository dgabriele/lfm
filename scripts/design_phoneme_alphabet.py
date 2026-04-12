#!/usr/bin/env python
"""Design a phoneme-level ASCII alphabet that tokenizes cleanly via Qwen BPE.

The problem: our existing IPA/romanized output produces strings that
Qwen's BPE fragments into byte-fallback subwords, destroying the
topographic structure the VAE would otherwise carry.  We want a
phoneme-level alphabet where:

  1. Each phoneme renders to a short ASCII sequence
  2. Composing phonemes into "words" (via concatenation) yields strings
     that Qwen tokenizes *deterministically* (same word → same
     subwords, in any context)
  3. Constructed words don't collide with real English/other-language
     words, so Qwen treats them as novel rather than invoking existing
     semantic priors

This script searches Qwen's BPE vocabulary for 2-character ASCII pairs
that satisfy stability under concatenation, then applies a wordlist
filter to remove real words, and finally validates a sample of
multi-phoneme words for consistent tokenization.

Produces: ``data/phoneme_alphabet_v1.json`` — a JSON artifact containing
the chosen phoneme inventory and verification statistics.  Consumed by
the forthcoming ``PhonemeVAEGenerator`` tokenizer.
"""

from __future__ import annotations

import json
import random
import string
import sys
from pathlib import Path

from transformers import AutoTokenizer

LETTERS = string.ascii_lowercase
MODEL_NAME = "Qwen/Qwen2.5-0.5B"
OUTPUT = Path("data/phoneme_alphabet_v1.json")
TARGET_SIZE = 50  # desired phoneme inventory size
SEED = 42
# Stability — strict enough that tokenization is deterministic, lenient
# enough that we retain linguistically-neutral (non-morpheme) strings.
STABILITY_THRESHOLD = 0.85
# Per-letter diversity cap for first/last char (prevents alphabet concentration).
PER_LETTER_CAP = 6
# Corpora for frequency estimation (rarity filter).
# Prose + code: we want candidates rare in BOTH so Qwen has weak priors
# whether interpreting prose or code context.
FREQ_PROSE_PATH = Path("data/translator/english_corpus.txt")
FREQ_CODE_PATH = Path("data/qwen_targets_cache/stack-smol-xl.jsonl")
FREQ_PROSE_LINES = 100_000
FREQ_CODE_LINES = 20_000  # Stack lines are much longer on average
# Max frequency percentile for candidates — reject anything above this
# (i.e. keep only the rare tail).  0.5 = bottom half by token frequency.
FREQ_PERCENTILE_MAX = 0.5


TWO_LETTER_WORDS = {
    "am", "an", "as", "at", "be", "by", "do", "go", "he", "hi",
    "if", "in", "is", "it", "me", "my", "no", "of", "oh", "ok",
    "on", "or", "so", "to", "up", "us", "we", "ye", "ah", "eh",
    "ho", "um", "aw", "ow", "ex", "ox", "ma", "pa",
}

# TLA-style code/technical acronyms that slip through the frequency filter
# because they're individually rare but recognizable to Qwen from code
# training data.  Including image formats, file extensions, DB keywords,
# common variable-name abbreviations, and protocol/markup tokens.
CODE_ACRONYM_BLACKLIST = {
    # image / media formats
    "img", "jpg", "png", "gif", "svg", "mp3", "mp4", "wav",
    # protocols / web / encoding
    "ftp", "http", "url", "uri", "dns", "tcp", "udp", "ssl", "utf",
    # variable abbreviations
    "var", "obj", "arr", "fn", "fns", "ptr", "ref", "buf", "tmp",
    "ctx", "cfg", "cls", "mod", "msg", "msgs", "err", "errs",
    # query / db
    "qry", "sql", "qty", "idx", "ids", "dto", "dao",
    # geometry / graphics
    "fov", "pts", "pos", "vec", "mat", "pix", "rgb", "uv",
    # generic tech
    "api", "cpu", "gpu", "ram", "io", "os", "ui", "ux",
    "xml", "yml", "csv", "pdf", "tex",
    # programming morphemes
    "ved",
    # cryptic / encoding-ish
    "ucz", "ogl", "ogs",
}

# English morphemes / word fragments that carry strong semantic priors
# to Qwen even when individually rare as token-ids.  These are BPE-stable
# exactly because they're recognizable sub-word units in real English:
# suffixes (-tion, -ure, -ery, -ful), common roots (act-, serv-, cept-),
# plural/tense endings (-ves, -ing, -ed).
ENGLISH_MORPHEME_BLACKLIST = {
    # nominal suffixes
    "ion", "ure", "ery", "ful", "ity", "ism", "ist", "ous",
    "ory", "ary", "ics", "ies",
    # adjectival / adverbial suffixes
    "ive", "ous", "ily", "less", "able", "ible",
    # plural / inflection
    "ves", "ing", "est",
    # prepositional / directional fragments
    "ere", "urn", "ort", "ach", "omp", "umb", "ull",
    # common stems recognized by Qwen
    "act", "app", "add", "ask", "end", "use", "own", "our",
    "its", "get", "key", "any", "out", "job", "old", "let",
    "ace", "ade", "ath", "elf", "ime", "ile", "icy", "ith",
    "iss", "org", "opp", "ult", "erv", "obs", "occ", "ail",
    "aph", "cep", "pec", "ven", "vel", "ful", "fig", "jud",
    "jug", "zen", "ann", "ang", "ann", "att", "icy",
}


def find_single_token_pairs(tok, length: int) -> list[str]:
    """All ASCII strings of `length` letters that tokenize to one token with space prefix."""
    import itertools
    single: list[str] = []
    for combo in itertools.product(LETTERS, repeat=length):
        p = "".join(combo)
        ids = tok.encode(" " + p, add_special_tokens=False)
        if len(ids) == 1:
            single.append(p)
    return single


def measure_stability(
    tok, candidates: list[str], test_partners: list[str],
) -> list[tuple[str, float, float]]:
    """For each candidate, compute (left_stable, right_stable) fractions.

    left_stable: ' A+B' tokenizes with leading token == ' A'.
    right_stable: ' B+A' tokenizes as exactly [' B', 'A'] with trailing 'A' bare.
    """
    single_id = {p: tok.encode(" " + p, add_special_tokens=False)[0] for p in candidates}
    partner_ids = [tok.encode(" " + b, add_special_tokens=False)[0] for b in test_partners]
    out: list[tuple[str, float, float]] = []
    for a in candidates:
        a_id = single_id[a]
        ls_ok = 0
        rs_ok = 0
        for b, b_id in zip(test_partners, partner_ids):
            # Left-stability
            ids = tok.encode(" " + a + b, add_special_tokens=False)
            if len(ids) >= 2 and ids[0] == a_id:
                ls_ok += 1
            # Right-stability
            ids2 = tok.encode(" " + b + a, add_special_tokens=False)
            if len(ids2) == 2 and ids2[0] == b_id:
                toks = tok.convert_ids_to_tokens(ids2)
                if toks[1] == a:
                    rs_ok += 1
        n = len(test_partners)
        out.append((a, ls_ok / n, rs_ok / n))
    return out


def main() -> None:
    print(f"Loading tokenizer: {MODEL_NAME}")
    tok = AutoTokenizer.from_pretrained(MODEL_NAME)

    # --------------------------------------------------------------
    # Step 1: enumerate single-token 2-char and 3-char ASCII strings
    # --------------------------------------------------------------
    print("\n[1] Enumerating single-token ASCII strings...")
    pairs2 = find_single_token_pairs(tok, 2)
    print(f"  {len(pairs2)} / {26**2} 2-char single-token strings")
    pairs3 = find_single_token_pairs(tok, 3)
    print(f"  {len(pairs3)} / {26**3} 3-char single-token strings")

    # --------------------------------------------------------------
    # Step 2: stability under concatenation (strict threshold).
    # --------------------------------------------------------------
    print(f"\n[2] Testing stability (threshold={STABILITY_THRESHOLD})...")
    # Use a mixed partner set so 2-char phonemes see 3-char neighbors and
    # vice-versa — closer to real usage.
    test_partners = pairs2[:60] + pairs3[:40]

    all_stability: list[tuple[str, float, float]] = []
    for pool_name, pool in [("2-char", pairs2), ("3-char", pairs3)]:
        print(f"  measuring {pool_name} ({len(pool)} candidates)...")
        stab = measure_stability(tok, pool, test_partners)
        all_stability.extend(stab)

    all_stability.sort(key=lambda x: x[1] * x[2], reverse=True)
    print("  top-15 by stability product:")
    for p, ls, rs in all_stability[:15]:
        print(f"    {p!r:>6}: left={ls:.2f} right={rs:.2f}  prod={ls*rs:.2f}")

    stable_candidates = [
        p for p, ls, rs in all_stability
        if ls >= STABILITY_THRESHOLD and rs >= STABILITY_THRESHOLD
    ]
    n2 = sum(1 for p in stable_candidates if len(p) == 2)
    n3 = sum(1 for p in stable_candidates if len(p) == 3)
    print(
        f"  {len(stable_candidates)} candidates ≥ {STABILITY_THRESHOLD} "
        f"(2-char: {n2}, 3-char: {n3})",
    )

    # --------------------------------------------------------------
    # Step 3a: drop 2-letter English words (real words invoke priors).
    # --------------------------------------------------------------
    print("\n[3a] Filtering real 2-letter English words...")
    before = len(stable_candidates)
    stable_candidates = [
        p for p in stable_candidates if p not in TWO_LETTER_WORDS
    ]
    print(f"  {before} → {len(stable_candidates)} after 2-letter drop")

    # --------------------------------------------------------------
    # Step 3a-bis: drop TLA-style code/tech acronyms that survive the
    # frequency filter but carry strong Qwen priors from code training.
    # --------------------------------------------------------------
    before = len(stable_candidates)
    dropped_code = [p for p in stable_candidates if p in CODE_ACRONYM_BLACKLIST]
    stable_candidates = [
        p for p in stable_candidates if p not in CODE_ACRONYM_BLACKLIST
    ]
    print(f"  {before} → {len(stable_candidates)} after code-acronym drop "
          f"(removed: {dropped_code})")

    # --------------------------------------------------------------
    # Step 3a-ter: drop English morphemes (suffixes, roots, fragments).
    # These carry semantic priors despite being low-frequency as raw
    # token-ids — Qwen recognizes them as sub-word units of real English.
    # --------------------------------------------------------------
    before = len(stable_candidates)
    dropped_morph = [
        p for p in stable_candidates if p in ENGLISH_MORPHEME_BLACKLIST
    ]
    stable_candidates = [
        p for p in stable_candidates if p not in ENGLISH_MORPHEME_BLACKLIST
    ]
    print(f"  {before} → {len(stable_candidates)} after English-morpheme drop "
          f"(removed: {dropped_morph})")

    # --------------------------------------------------------------
    # Step 3b: frequency filter — prefer rare tokens in real English.
    #
    # High-BPE-stability units tend to be common English morphemes
    # ('add', 'app', 'ith', 'org'), which carry strong semantic priors
    # in Qwen.  We want BPE-stable AND semantically-neutral — so we
    # penalize candidates whose token id appears frequently in a real
    # English corpus.
    # --------------------------------------------------------------
    print(f"\n[3b] Measuring token frequencies (prose + code)...")
    from collections import Counter

    def count_tokens_from_file(path: Path, max_lines: int, is_jsonl: bool) -> Counter[int]:
        c: Counter[int] = Counter()
        with path.open() as fh:
            for i, line in enumerate(fh):
                if i >= max_lines:
                    break
                text = line.strip()
                if is_jsonl:
                    # Extract "text" field from JSONL
                    try:
                        text = json.loads(text).get("text", "")
                    except json.JSONDecodeError:
                        continue
                if not text:
                    continue
                # Cap individual sample length for speed
                if len(text) > 4000:
                    text = text[:4000]
                ids = tok.encode(text, add_special_tokens=False)
                c.update(ids)
        return c

    print(f"  prose: {FREQ_PROSE_PATH} (≤{FREQ_PROSE_LINES} lines)")
    prose_counts = count_tokens_from_file(FREQ_PROSE_PATH, FREQ_PROSE_LINES, is_jsonl=False)
    print(f"    {sum(prose_counts.values()):,} tokens")

    print(f"  code:  {FREQ_CODE_PATH} (≤{FREQ_CODE_LINES} lines)")
    code_counts = count_tokens_from_file(FREQ_CODE_PATH, FREQ_CODE_LINES, is_jsonl=True)
    print(f"    {sum(code_counts.values()):,} tokens")

    # Per-corpus rate filter: a candidate must be rare in BOTH corpora.
    # This catches prose-common ("and") AND code-common ("var") independently,
    # so technical acronyms rare overall-but-concentrated-in-code get flagged.
    prose_total = max(sum(prose_counts.values()), 1)
    code_total = max(sum(code_counts.values()), 1)

    cand_rates: list[tuple[str, float, float, int, int]] = []
    for p in stable_candidates:
        tid = tok.encode(" " + p, add_special_tokens=False)[0]
        pc = prose_counts.get(tid, 0)
        cc = code_counts.get(tid, 0)
        pr = pc / prose_total
        cr = cc / code_total
        cand_rates.append((p, pr, cr, pc, cc))

    # Cutoff = max_rate such that the bottom FREQ_PERCENTILE_MAX of candidates
    # pass, using max(prose_rate, code_rate) as the score (worst-case rarity).
    scores = sorted(max(pr, cr) for _, pr, cr, _, _ in cand_rates)
    cutoff_idx = int(len(scores) * FREQ_PERCENTILE_MAX)
    rate_cutoff = (
        scores[cutoff_idx] if cutoff_idx < len(scores) else scores[-1]
    )
    print(f"  Max-rate cutoff (≤ {FREQ_PERCENTILE_MAX*100:.0f}th pct): "
          f"{rate_cutoff*1e6:.1f} per million tokens")

    kept = [
        p for p, pr, cr, _, _ in cand_rates if max(pr, cr) <= rate_cutoff
    ]
    rejected = [
        (p, pr, cr, pc, cc) for p, pr, cr, pc, cc in cand_rates
        if max(pr, cr) > rate_cutoff
    ]
    rejected.sort(key=lambda x: -max(x[1], x[2]))
    print(f"  {len(stable_candidates)} → {len(kept)} after per-corpus rate filter")
    print("  Top-15 rejected (by worst-case rate):")
    print("    phoneme  prose_rate   code_rate   (prose_n / code_n)")
    for p, pr, cr, pc, cc in rejected[:15]:
        tag = "[prose]" if pr > cr else "[code]"
        print(
            f"    {p!r:>6}  {pr*1e6:>8.1f}/M  {cr*1e6:>8.1f}/M  "
            f"({pc:,}/{cc:,}) {tag}",
        )
    stable_candidates = kept

    # --------------------------------------------------------------
    # Step 4: select a diverse phoneme inventory (cap on first/last char).
    # --------------------------------------------------------------
    print(f"\n[4] Selecting diverse inventory (target={TARGET_SIZE}, cap={PER_LETTER_CAP})...")
    random.seed(SEED)
    selected: list[str] = []
    first_char_counts: dict[str, int] = {}
    last_char_counts: dict[str, int] = {}

    pool = [p for p, _, _ in all_stability if p in stable_candidates]

    for p in pool:
        a, b = p[0], p[-1]
        if (first_char_counts.get(a, 0) >= PER_LETTER_CAP
                or last_char_counts.get(b, 0) >= PER_LETTER_CAP):
            continue
        selected.append(p)
        first_char_counts[a] = first_char_counts.get(a, 0) + 1
        last_char_counts[b] = last_char_counts.get(b, 0) + 1
        if len(selected) >= TARGET_SIZE:
            break

    print(f"  Selected {len(selected)} phonemes:")
    print(f"    {selected}")
    print(f"  First-char distribution: {dict(sorted(first_char_counts.items()))}")
    print(f"  Last-char distribution:  {dict(sorted(last_char_counts.items()))}")
    n2_sel = sum(1 for p in selected if len(p) == 2)
    n3_sel = sum(1 for p in selected if len(p) == 3)
    print(f"  Length split: {n2_sel} two-char, {n3_sel} three-char")

    # --------------------------------------------------------------
    # Step 5: validation — sample 5000 random 3-phoneme words and
    # check deterministic tokenization.
    # --------------------------------------------------------------
    print("\n[5] Validating 3-phoneme word tokenization stability...")
    random.seed(SEED)
    sample_words: list[str] = []
    for _ in range(5000):
        word = "".join(random.choice(selected) for _ in range(3))
        sample_words.append(word)

    # Check: tokenizing " word" should produce ≤3 tokens, and tokenizing
    # the same word in two different contexts should give the same result.
    token_lengths: list[int] = []
    word_to_tokens: dict[str, tuple[int, ...]] = {}
    for w in sample_words:
        if w in word_to_tokens:
            continue
        ids = tuple(tok.encode(" " + w, add_special_tokens=False))
        word_to_tokens[w] = ids
        token_lengths.append(len(ids))

    print(f"  Unique words sampled: {len(word_to_tokens)}")
    print(f"  Token-length distribution:")
    from collections import Counter
    lc = Counter(token_lengths)
    for k in sorted(lc):
        print(f"    {k} tokens: {lc[k]} words ({100*lc[k]/len(token_lengths):.1f}%)")

    # Context-stability check: for 200 of these words, tokenize in two
    # different surrounding contexts and confirm the word's tokens are
    # identical modulo context.
    print("  Context stability (word tokens unchanged in different sentences):")
    test_words = list(word_to_tokens.keys())[:200]
    prefix1 = " the "
    prefix2 = " a very "
    stable_ctx = 0
    for w in test_words:
        ids1 = tok.encode(prefix1 + w + " ends here.", add_special_tokens=False)
        ids2 = tok.encode(prefix2 + w + " appears!", add_special_tokens=False)
        # Check: the word's token subsequence should appear in both
        # (we can't simply compare lists because prefixes differ;
        # instead, test that the word's expected tokens are a
        # consecutive subsequence of each encoding).
        expected = word_to_tokens[w]
        def contains(seq, sub):
            for i in range(len(seq) - len(sub) + 1):
                if tuple(seq[i:i+len(sub)]) == sub:
                    return True
            return False
        if contains(ids1, expected) and contains(ids2, expected):
            stable_ctx += 1
    print(f"    {stable_ctx}/{len(test_words)} words stable across contexts")

    # --------------------------------------------------------------
    # Step 6: save the alphabet to disk
    # --------------------------------------------------------------
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    alphabet = {
        "version": "v1",
        "model": MODEL_NAME,
        "size": len(selected),
        "phonemes": selected,
        "first_char_distribution": first_char_counts,
        "last_char_distribution": last_char_counts,
        "validation": {
            "sample_size": len(word_to_tokens),
            "token_length_distribution": dict(lc),
            "context_stable_fraction": stable_ctx / len(test_words),
        },
    }
    with OUTPUT.open("w") as f:
        json.dump(alphabet, f, indent=2)
    print(f"\nWrote {OUTPUT}")


if __name__ == "__main__":
    main()

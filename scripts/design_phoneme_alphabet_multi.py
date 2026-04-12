#!/usr/bin/env python
"""Design a phoneme alphabet from Qwen tokens characteristic of
non-dominant Latin-script languages.

Goal:

  Select tokens that (a) Qwen learned well enough to have compositional
  structure (avoiding representationally-noisy vestigial tokens) but
  (b) from languages Qwen has moderate rather than dominant pretraining
  signal for, AND (c) that don't appear in English word decomposition.

The result is a phoneme inventory where fine-tuning can leverage Qwen's
pretrained attention/FFN patterns for word-like composition (architectural
leverage), while the damage from reshaping embeddings lands on
capabilities — namely Czech/Finnish/Estonian/Hungarian/Turkish/Indonesian
fluency — that are not critical to the Neuroglot interpretation pipeline.

Source languages (positive signal — tokens common here are what we want):
  - Czech (ces) — IE Slavic, moderate Qwen resource
  - Polish (pol) — IE Slavic, moderate
  - Finnish (fin) — Uralic, Latin-script, lower
  - Estonian (est) — Uralic, Latin-script, lower
  - Hungarian (hun) — Uralic, Latin-script, lower
  - Turkish (tur) — Turkic, Latin-script, moderate
  - Indonesian (ind) — Austronesian, Latin-script, moderate

Exclusion signals (negative — tokens must be rare here):
  - English prose (we must not damage English output)
  - Stack code (we must not damage technical English)
  - German (deu) — high Qwen resource, preserve
  - Spanish (spa) — high Qwen resource, preserve
  - Portuguese (por) — high Qwen resource, preserve

A candidate passes if:
  1. Common enough in at least one source language (meaningful fragment somewhere)
  2. Rare in ALL of the exclusion sources
  3. BPE concat-stable
  4. Latin-script word-like form
"""

from __future__ import annotations

import json
import random
import unicodedata
from collections import Counter
from pathlib import Path

import torch
from transformers import AutoTokenizer

MODEL_NAME = "Qwen/Qwen2.5-0.5B"
OUTPUT = Path("data/phoneme_alphabet_multi.json")
TARGET_SIZE = 50
SEED = 42

MIN_LEN = 2
MAX_LEN = 3  # tightened: length 4-5 tokens fragment under concat
STABILITY_THRESHOLD = 0.5  # English tokens themselves median ~0.7, 25%ile ~0.6
STABILITY_PARTNERS = 60
PER_FIRST_CHAR_CAP = 6

# Drop tokens Qwen will recognize as code abbreviations regardless of
# their (low) rate in our prose/code sample.
CODE_ACRONYM_BLACKLIST = {
    "img", "jpg", "png", "gif", "svg", "mp3", "mp4", "wav",
    "ftp", "http", "url", "uri", "dns", "tcp", "udp", "ssl", "utf",
    "var", "obj", "arr", "ptr", "ref", "buf", "tmp",
    "ctx", "cfg", "cls", "mod", "msg", "err",
    "qry", "sql", "qty", "idx", "ids", "dto", "dao",
    "fov", "pts", "pos", "vec", "mat", "pix", "rgb",
    "api", "cpu", "gpu", "ram", "xml", "yml", "csv", "pdf", "tex",
    "ved", "ucz", "ogl", "ogs", "fk", "ik", "tk", "tl", "kl",
    "dv", "vk", "hk", "kk", "eb", "ql",
}

LINES_PER_LANG = 40_000  # per-language sample cap

SOURCE_LANGS = {
    "ces": "data/leipzig/ces_news_2022_100K/ces_news_2022_100K-sentences.txt",
    "pol": "data/leipzig/pol_news_2023_100K/pol_news_2023_100K-sentences.txt",
    "fin": "data/leipzig/fin_news_2022_100K/fin_news_2022_100K-sentences.txt",
    "est": "data/leipzig/est_news_2022_100K/est_news_2022_100K-sentences.txt",
    "hun": "data/leipzig/hun_news_2022_100K/hun_news_2022_100K-sentences.txt",
    "tur": "data/leipzig/tur_news_2023_100K/tur_news_2023_100K-sentences.txt",
    "ind": "data/leipzig/ind_news_2022_100K/ind_news_2022_100K-sentences.txt",
}

# Major Latin-script languages Qwen is well-trained on — we must not
# damage capabilities in these.  Tokens common in ANY of these fail.
EXCLUSION_LANGS = {
    "deu": "data/leipzig/deu_news_2020_300K/deu_news_2020_300K-sentences.txt",
    "spa": "data/leipzig/spa_news_2023_100K/spa_news_2023_100K-sentences.txt",
    "por": "data/leipzig/por_news_2022_100K/por_news_2022_100K-sentences.txt",
    "english": "data/translator/english_corpus.txt",
}

CODE_PATH = Path("data/qwen_targets_cache/stack-smol-xl.jsonl")
CODE_LINES = 10_000

# A token must appear at least this many times per million in ≥1 source
# language (to be a real fragment rather than a noise artifact).
MIN_SOURCE_RATE_PER_MILLION = 5.0
# A token must appear fewer than this many times per million in ALL
# exclusion sources (not used in primary-language morphology).
MAX_EXCLUSION_RATE_PER_MILLION = 2.0


def is_latin_wordlike(s: str) -> bool:
    if not s:
        return False
    for ch in s:
        if unicodedata.category(ch)[0] != "L":
            return False
        try:
            name = unicodedata.name(ch, "")
        except ValueError:
            return False
        if "LATIN" not in name:
            return False
    return True


def enumerate_candidates(tok) -> list[tuple[int, str]]:
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
        # Require lowercase — mixed case would invoke English priors
        # on capitalized words and fragment inconsistently.
        if bare != bare.lower():
            continue
        if bare.lower() in CODE_ACRONYM_BLACKLIST:
            continue
        if bare not in vocab:
            continue
        cands.append((tid, bare))
    return cands


def count_corpus_tokens(
    tok, path: Path, max_lines: int, is_jsonl: bool = False,
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
            else:
                # Leipzig format: "tab-separated id<tab>sentence" — strip numeric prefix
                parts = text.split("\t", 1)
                if len(parts) == 2:
                    text = parts[1]
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
    partner_ids = [tok.encode(" " + b, add_special_tokens=False) for b in partners]
    partners_clean = [
        (b, ids[0]) for b, ids in zip(partners, partner_ids) if len(ids) == 1
    ]
    out = []
    n = max(len(partners_clean), 1)
    for tid, bare in candidates:
        ls = rs = 0
        for b, b_id in partners_clean:
            ids = tok.encode(" " + bare + b, add_special_tokens=False)
            if len(ids) >= 2 and ids[0] == tid:
                ls += 1
            ids2 = tok.encode(" " + b + bare, add_special_tokens=False)
            if len(ids2) == 2 and ids2[0] == b_id:
                if tok.convert_ids_to_tokens(ids2)[1] == bare:
                    rs += 1
        out.append((tid, bare, ls / n, rs / n))
    return out


def main() -> None:
    print(f"Loading tokenizer: {MODEL_NAME}")
    tok = AutoTokenizer.from_pretrained(MODEL_NAME)

    print("\n[1] Enumerating Latin word-like Qwen vocab tokens...")
    candidates = enumerate_candidates(tok)
    print(f"  {len(candidates)} candidates (Latin-script, len {MIN_LEN}-{MAX_LEN})")

    # --------------------------------------------------------------
    # [2] Count tokens in each corpus
    # --------------------------------------------------------------
    print(f"\n[2] Counting tokens per corpus (≤{LINES_PER_LANG} lines each)...")

    source_counts: dict[str, tuple[Counter[int], int]] = {}
    for lang, path in SOURCE_LANGS.items():
        p = Path(path)
        if not p.exists():
            print(f"  [skip] {lang}: {path} not found")
            continue
        c = count_corpus_tokens(tok, p, LINES_PER_LANG)
        total = max(sum(c.values()), 1)
        source_counts[lang] = (c, total)
        print(f"  source {lang}: {total:,} tokens")

    exclusion_counts: dict[str, tuple[Counter[int], int]] = {}
    for lang, path in EXCLUSION_LANGS.items():
        p = Path(path)
        if not p.exists():
            print(f"  [skip] {lang}: {path} not found")
            continue
        c = count_corpus_tokens(tok, p, LINES_PER_LANG)
        total = max(sum(c.values()), 1)
        exclusion_counts[lang] = (c, total)
        print(f"  exclude {lang}: {total:,} tokens")

    if CODE_PATH.exists():
        code_c = count_corpus_tokens(tok, CODE_PATH, CODE_LINES, is_jsonl=True)
        code_total = max(sum(code_c.values()), 1)
        exclusion_counts["code"] = (code_c, code_total)
        print(f"  exclude code: {code_total:,} tokens")

    # --------------------------------------------------------------
    # [3] Filter: rare in exclusions, common in ≥1 source
    # --------------------------------------------------------------
    print(f"\n[3] Filtering candidates...")
    print(f"  rule: max_exclusion_rate ≤ {MAX_EXCLUSION_RATE_PER_MILLION}/M "
          f"AND max_source_rate ≥ {MIN_SOURCE_RATE_PER_MILLION}/M")

    kept: list[dict] = []
    for tid, bare in candidates:
        # Exclusion rates — max across all exclusion corpora
        max_excl = 0.0
        for lang, (c, total) in exclusion_counts.items():
            rate = c.get(tid, 0) / total
            if rate > max_excl:
                max_excl = rate
        if max_excl * 1e6 > MAX_EXCLUSION_RATE_PER_MILLION:
            continue
        # Source rates — max across all source corpora
        per_source: dict[str, float] = {}
        for lang, (c, total) in source_counts.items():
            per_source[lang] = c.get(tid, 0) / total
        max_src = max(per_source.values()) if per_source else 0.0
        if max_src * 1e6 < MIN_SOURCE_RATE_PER_MILLION:
            continue
        kept.append({
            "tid": tid, "bare": bare,
            "max_excl_rate": max_excl, "max_src_rate": max_src,
            "per_source_rate": per_source,
            "dominant_source": max(per_source, key=per_source.get) if per_source else "",
        })

    print(f"  {len(candidates)} → {len(kept)} after filter")

    # Show breakdown by dominant source language
    by_lang = Counter(k["dominant_source"] for k in kept)
    print(f"  dominant-source distribution: {dict(by_lang)}")

    print("  sample kept (top-15 by max_src_rate):")
    kept.sort(key=lambda k: -k["max_src_rate"])
    for k in kept[:15]:
        rates = " ".join(f"{lang}={r*1e6:.1f}" for lang, r in k["per_source_rate"].items() if r > 0)
        print(f"    {k['bare']!r:>7}  excl={k['max_excl_rate']*1e6:.2f}/M  "
              f"dominant={k['dominant_source']}  rates=[{rates}]")

    # --------------------------------------------------------------
    # [4] Concat stability
    # --------------------------------------------------------------
    print(f"\n[4] Concat stability test (threshold={STABILITY_THRESHOLD})...")
    # Partners are what Neuroglot phonemes will actually abut in use:
    # English common words (in carrier sentences) + other phoneme candidates.
    # Using minority-lang tokens as partners produces false-positive merges
    # because BPE has learned many cross-minority-lang adjacencies.
    english_partners = [
        "the", "of", "and", "to", "is", "it", "was", "for", "with",
        "that", "this", "on", "by", "at", "as", "be", "has", "an",
        "or", "not", "but", "he", "she", "we", "they", "which", "what",
        "who", "how", "when", "where", "why", "can", "will", "have",
        "there", "about", "some", "all", "said", "one", "two", "three",
    ]
    partners = english_partners + [k["bare"] for k in kept[:STABILITY_PARTNERS // 2]]
    pairs = [(k["tid"], k["bare"]) for k in kept]
    stab = measure_stability(tok, pairs, partners)
    stable_ids = {
        (tid, bare) for tid, bare, ls, rs in stab
        if ls >= STABILITY_THRESHOLD and rs >= STABILITY_THRESHOLD
    }
    stable_kept = [k for k in kept if (k["tid"], k["bare"]) in stable_ids]
    stability_lookup = {(tid, bare): (ls, rs) for tid, bare, ls, rs in stab}
    print(f"  {len(stable_kept)} / {len(kept)} pass stability")

    # Sort stable by score: high source rate + low exclusion rate
    for k in stable_kept:
        ls, rs = stability_lookup[(k["tid"], k["bare"])]
        k["stab_product"] = ls * rs
    stable_kept.sort(
        key=lambda k: (k["stab_product"], k["max_src_rate"]),
        reverse=True,
    )

    # --------------------------------------------------------------
    # [5] Select diverse inventory
    # --------------------------------------------------------------
    print(f"\n[5] Selecting diverse inventory (target={TARGET_SIZE}, cap={PER_FIRST_CHAR_CAP})...")
    # Also prefer source-language diversity
    per_first: Counter[str] = Counter()
    per_src: Counter[str] = Counter()
    selected: list[dict] = []
    SRC_CAP = TARGET_SIZE // len(SOURCE_LANGS) + 5  # rough balancer
    for k in stable_kept:
        fc = k["bare"][0].lower()
        if per_first[fc] >= PER_FIRST_CHAR_CAP:
            continue
        if per_src[k["dominant_source"]] >= SRC_CAP:
            continue
        selected.append(k)
        per_first[fc] += 1
        per_src[k["dominant_source"]] += 1
        if len(selected) >= TARGET_SIZE:
            break

    selected_bare = [k["bare"] for k in selected]
    print(f"  Selected {len(selected_bare)} phonemes.")
    print(f"  Source dist: {dict(per_src)}")
    print(f"  Length dist: {dict(sorted(Counter(len(b) for b in selected_bare).items()))}")
    print(f"  First-char dist: {dict(sorted(per_first.items()))}")
    print("  Phonemes (with dominant source):")
    for k in selected:
        rates = " ".join(f"{lang}={r*1e6:.1f}" for lang, r in k["per_source_rate"].items() if r > 0)
        print(f"    {k['bare']!r:>7}  dominant={k['dominant_source']}  rates=[{rates}]")

    # --------------------------------------------------------------
    # [6] Validation — 3-phoneme word tokenization
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
        "version": "multilingual-latin-v1",
        "model": MODEL_NAME,
        "method": "Latin word-like tokens from non-dominant Latin-script "
                  "languages (ces/pol/fin/est/hun/tur/ind), excluding "
                  "primary-language tokens (eng/deu/spa/por/code)",
        "size": len(selected_bare),
        "phonemes": selected_bare,
        "source_distribution": dict(per_src),
        "first_char_distribution": dict(per_first),
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

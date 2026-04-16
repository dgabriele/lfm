"""Empirical probe to validate + formalize the v14 sentence-level
corpus representation BEFORE we spend cycles re-parsing 5M sentences.

The output is a plain-text report combining:

  1. Aggregate stats across N sampled sentences (parser behavior on
     punctuation, root labels, top-level children, abbreviation splits,
     coordination patterns, sentence-length distribution, OOV-prone
     tokens, etc.)
  2. Categorized edge-case examples (5 each) so design decisions can be
     made against real cases rather than imagined ones.
  3. A handful of full-tree dumps for visual inspection.

Read the report end-to-end before committing to a wrapping spec.

Usage::

    poetry run python scripts/probe_v14_punctuation.py \\
        --chunks-dir data/datasets/english-constituents-v13/chunks \\
        --num-sentences 300 \\
        --out output/v14_probe.txt
"""

from __future__ import annotations

import argparse
import logging
import os
import random
import re
from collections import Counter, defaultdict
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger(__name__)


PUNCT_CHARS = set(".,;:!?—–-\"'(){}[]")


def sample_sentences(chunks_dir: Path, n: int, seed: int = 42) -> list[str]:
    """Reservoir-sample sentences across all chunks for a representative
    mix without loading everything into memory."""
    rng = random.Random(seed)
    sentences: list[str] = []
    paths = sorted(chunks_dir.glob("chunk_*.txt"))
    if not paths:
        raise SystemExit(f"no chunk_*.txt under {chunks_dir}")
    seen = 0
    for p in paths:
        with p.open() as f:
            for line in f:
                line = line.strip()
                if not line or len(line) < 20:
                    continue
                seen += 1
                if len(sentences) < n:
                    sentences.append(line)
                else:
                    j = rng.randrange(seen)
                    if j < n:
                        sentences[j] = line
    logger.info("sampled %d / %d sentences across %d chunks", len(sentences), seen, len(paths))
    return sentences


def tree_to_str(node, indent: int = 0) -> str:
    """Render a Stanza constituency node as PTB-style brackets, multi-line."""
    label = getattr(node, "label", "?")
    children = getattr(node, "children", []) or []
    pad = "  " * indent
    if not children:
        return f"{pad}({label})"
    parts = [f"{pad}({label}"]
    for c in children:
        parts.append(tree_to_str(c, indent + 1))
    parts.append(f"{pad})")
    return "\n".join(parts)


def collect_punct_parents(node, parent_label: str | None = None,
                          out: dict | None = None) -> dict[str, Counter]:
    """For every leaf punctuation token, record the label of its parent
    constituent.  Returns ``{punct_char: Counter(parent_label -> count)}``.
    """
    if out is None:
        out = defaultdict(Counter)
    label = getattr(node, "label", "")
    children = getattr(node, "children", []) or []
    if not children:
        # Leaf token.  In Stanza constituency trees a leaf is a word string
        # under a POS tag.  Punctuation gets POS tags like `.`, `,`, `:`,
        # ``, ``''`, etc.  We identify punct by the leaf string contents.
        text = (label or "").strip()
        if text and all(c in PUNCT_CHARS for c in text):
            out[text][parent_label or "<ROOT>"] += 1
        return out
    for c in children:
        collect_punct_parents(c, parent_label=label, out=out)
    return out


def top_level_children_of_s(node) -> list[str]:
    """Find labels of immediate children of the highest S in the tree
    (typically directly under ROOT).  Tells us what goes where if we
    keep "top-level constituents under S" wrapping."""
    label = getattr(node, "label", "")
    children = getattr(node, "children", []) or []
    if label == "S" and children:
        # Each child label is what we'd wrap directly under <S>.
        return [getattr(c, "label", "?") for c in children]
    for c in children:
        out = top_level_children_of_s(c)
        if out:
            return out
    return []


def walk_leaves(node):
    """Yield each leaf token string in the tree."""
    children = getattr(node, "children", []) or []
    if not children:
        yield (getattr(node, "label", "") or "").strip()
        return
    for c in children:
        yield from walk_leaves(c)


def has_pattern(node, predicate) -> bool:
    if predicate(node):
        return True
    for c in getattr(node, "children", []) or []:
        if has_pattern(c, predicate):
            return True
    return False


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--chunks-dir", type=Path,
                    default=Path("data/datasets/english-constituents-v13/chunks"))
    ap.add_argument("--num-sentences", type=int, default=200)
    ap.add_argument("--out", type=Path, default=Path("output/v14_probe.txt"))
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    sentences = sample_sentences(args.chunks_dir, args.num_sentences, args.seed)

    # Force Stanza backend (Benepar's English constituency model is what
    # we prefer in production but Stanza is the simpler dep here).
    os.environ.setdefault("LFM_FORCE_STANZA", "1")
    import stanza
    logger.info("loading Stanza English constituency pipeline...")
    stanza.download("en", processors="tokenize,pos,constituency", verbose=False)
    nlp = stanza.Pipeline(
        "en",
        processors="tokenize,pos,constituency",
        use_gpu=False,
        verbose=False,
    )

    # Aggregates ----------------------------------------------------
    punct_parents: dict[str, Counter] = defaultdict(Counter)
    top_level_seq_counter: Counter = Counter()
    sentences_with_period_inside_vp = 0
    sentences_with_comma_inside_vp = 0
    sentences_with_abbrev_split = 0
    n_root_s = 0
    n_root_other = 0
    root_label_counts: Counter = Counter()
    sample_dump: list[str] = []
    edge_cases: dict[str, list[str]] = defaultdict(list)

    # Coverage we want to know about for v14 design
    word_count_dist: Counter = Counter()
    token_count_dist: Counter = Counter()
    has_no_verb = 0
    has_quote = 0
    has_paren = 0
    has_emdash = 0
    has_semicolon = 0
    has_colon = 0
    has_relative_clause = 0
    has_top_level_coord = 0
    has_question_mark = 0
    has_exclamation = 0
    has_subordinator_lead = 0
    has_number = 0
    short_sentences = 0
    long_sentences = 0
    has_inline_url_remnant = 0
    has_dialogue_verb = 0
    contains_ellipsis = 0
    n_split_into_multiple = 0

    abbrev_re = re.compile(r"\b(?:Mr|Mrs|Ms|Dr|Prof|St|Jr|Sr|vs|etc|i\.e|e\.g)\.")
    url_residue_re = re.compile(r"https?://|www\.")
    dialogue_verb_re = re.compile(r"\b(said|says|asked|replied|told|exclaimed|whispered)\b", re.I)
    ellipsis_re = re.compile(r"\.\.\.|…")

    for i, raw in enumerate(sentences):
        try:
            doc = nlp(raw)
        except Exception as e:
            logger.warning("parse failed on: %s (%s)", raw[:80], e)
            continue
        # Lightweight surface-level signals computed from the raw text
        n_words = len(raw.split())
        word_count_dist[(n_words // 5) * 5] += 1  # 5-word buckets
        if n_words <= 3:
            short_sentences += 1
            if len(edge_cases["very_short"]) < 5:
                edge_cases["very_short"].append(raw[:200])
        if n_words >= 60:
            long_sentences += 1
            if len(edge_cases["very_long"]) < 5:
                edge_cases["very_long"].append(raw[:200])
        if '"' in raw or "“" in raw or "”" in raw:
            has_quote += 1
            if len(edge_cases["has_quote"]) < 5:
                edge_cases["has_quote"].append(raw[:200])
        if "(" in raw or ")" in raw:
            has_paren += 1
            if len(edge_cases["has_paren"]) < 5:
                edge_cases["has_paren"].append(raw[:200])
        if "—" in raw or "–" in raw:
            has_emdash += 1
            if len(edge_cases["has_emdash"]) < 5:
                edge_cases["has_emdash"].append(raw[:200])
        if ";" in raw:
            has_semicolon += 1
            if len(edge_cases["has_semicolon"]) < 5:
                edge_cases["has_semicolon"].append(raw[:200])
        if ":" in raw:
            has_colon += 1
            if len(edge_cases["has_colon"]) < 5:
                edge_cases["has_colon"].append(raw[:200])
        if "?" in raw:
            has_question_mark += 1
            if len(edge_cases["has_question_mark"]) < 5:
                edge_cases["has_question_mark"].append(raw[:200])
        if "!" in raw:
            has_exclamation += 1
            if len(edge_cases["has_exclamation"]) < 5:
                edge_cases["has_exclamation"].append(raw[:200])
        if any(ch.isdigit() for ch in raw):
            has_number += 1
        if url_residue_re.search(raw):
            has_inline_url_remnant += 1
            if len(edge_cases["url_remnant"]) < 5:
                edge_cases["url_remnant"].append(raw[:200])
        if dialogue_verb_re.search(raw):
            has_dialogue_verb += 1
            if len(edge_cases["dialogue_verb"]) < 5:
                edge_cases["dialogue_verb"].append(raw[:200])
        if ellipsis_re.search(raw):
            contains_ellipsis += 1
            if len(edge_cases["ellipsis"]) < 5:
                edge_cases["ellipsis"].append(raw[:200])

        # Some inputs may segment into multiple sentences (e.g. abbreviation issues)
        seg_count = len(doc.sentences)
        if seg_count > 1:
            n_split_into_multiple += 1
            if abbrev_re.search(raw):
                sentences_with_abbrev_split += 1
                if len(edge_cases["abbreviation_caused_split"]) < 5:
                    edge_cases["abbreviation_caused_split"].append(raw[:200])
            if len(edge_cases["multi_segment"]) < 5:
                edge_cases["multi_segment"].append(f"[{seg_count} segs] {raw[:200]}")

        for sent in doc.sentences:
            tree = sent.constituency
            root_label = getattr(tree, "label", "?")
            root_label_counts[root_label] += 1

            # Highest S identification — for ROOT-S sentences, the tree's
            # only child is S
            children = getattr(tree, "children", []) or []
            top_s = None
            for c in children:
                if getattr(c, "label", "") == "S":
                    top_s = c
                    break
            if top_s is not None:
                n_root_s += 1
                # Top-level structure
                tlc = [getattr(c, "label", "?") for c in (top_s.children or [])]
                top_level_seq_counter[tuple(tlc)] += 1
            else:
                n_root_other += 1

            # Punctuation per-mark parent labels
            collect_punct_parents(tree, parent_label=root_label, out=punct_parents)

            # Specific checks
            def parent_is_vp_with_period(node):
                lbl = getattr(node, "label", "")
                children_ = getattr(node, "children", []) or []
                if lbl == "VP":
                    for c in children_:
                        if not (getattr(c, "children", []) or []):
                            text = (getattr(c, "label", "") or "").strip()
                            if text == ".":
                                return True
                return False

            def parent_is_vp_with_comma(node):
                lbl = getattr(node, "label", "")
                children_ = getattr(node, "children", []) or []
                if lbl == "VP":
                    for c in children_:
                        if not (getattr(c, "children", []) or []):
                            text = (getattr(c, "label", "") or "").strip()
                            if text == ",":
                                return True
                return False

            if has_pattern(tree, parent_is_vp_with_period):
                sentences_with_period_inside_vp += 1
                if len(edge_cases["period_inside_vp"]) < 5:
                    edge_cases["period_inside_vp"].append(raw[:200])
            if has_pattern(tree, parent_is_vp_with_comma):
                sentences_with_comma_inside_vp += 1
                if len(edge_cases["comma_inside_vp"]) < 5:
                    edge_cases["comma_inside_vp"].append(raw[:200])

            # Tree-driven structural counts
            def has_label(n, label):
                if getattr(n, "label", "") == label:
                    return True
                for c in getattr(n, "children", []) or []:
                    if has_label(c, label):
                        return True
                return False

            if not has_pattern(tree, lambda n: getattr(n, "label", "").startswith("V")):
                has_no_verb += 1
                if len(edge_cases["no_verb"]) < 5:
                    edge_cases["no_verb"].append(raw[:200])
            if has_label(tree, "SBAR"):
                has_relative_clause += 1
            # Top-level coordination: child of S that itself is S
            for c in getattr(tree, "children", []) or []:
                if getattr(c, "label", "") == "S":
                    inner = getattr(c, "children", []) or []
                    inner_labels = [getattr(x, "label", "") for x in inner]
                    if inner_labels.count("S") >= 2:
                        has_top_level_coord += 1
                        if len(edge_cases["top_level_coord"]) < 5:
                            edge_cases["top_level_coord"].append(raw[:200])
                        break
            # Subordinator at sentence start (Although, While, Because, ...)
            first = (raw.split() or [""])[0].lower().rstrip(",")
            if first in {"although", "while", "because", "since", "if",
                         "when", "after", "before", "though", "as", "unless",
                         "until", "whereas"}:
                has_subordinator_lead += 1
                if len(edge_cases["subordinator_lead"]) < 5:
                    edge_cases["subordinator_lead"].append(raw[:200])

            # Token count from the parse (ignores tag-bracketing overhead)
            n_tokens = sum(1 for _ in walk_leaves(tree))
            token_count_dist[(n_tokens // 10) * 10] += 1

        # Save 8 representative full-tree dumps spread across the sample
        if i in (0, 5, 17, 42, 80, 120, 160, 199):
            sample_dump.append(f"=== sentence #{i} ===\n{raw}\n" +
                               "\n".join(tree_to_str(s.constituency) for s in doc.sentences) +
                               "\n")

    # Report -------------------------------------------------------
    args.out.parent.mkdir(parents=True, exist_ok=True)
    lines = []
    lines.append(f"v14 sentence-level corpus probe — {args.num_sentences} sampled sentences\n")

    lines.append("=== root labels ===")
    for lbl, ct in root_label_counts.most_common():
        lines.append(f"  {lbl:>8}  {ct}")
    lines.append(f"\n  ROOT→S sentences:     {n_root_s}")
    lines.append(f"  ROOT→other sentences: {n_root_other}")

    lines.append("\n=== punctuation parent-label distribution ===")
    for ch in sorted(punct_parents):
        total = sum(punct_parents[ch].values())
        top = ", ".join(f"{l}={c}" for l, c in punct_parents[ch].most_common(5))
        lines.append(f"  '{ch}'  N={total}  top parents: {top}")

    lines.append("\n=== top-level S children sequence (top 15) ===")
    for seq, ct in top_level_seq_counter.most_common(15):
        lines.append(f"  {ct:>4}  {seq}")

    lines.append("\n=== specific checks ===")
    lines.append(f"  period inside VP:               {sentences_with_period_inside_vp}")
    lines.append(f"  comma  inside VP:               {sentences_with_comma_inside_vp}")
    lines.append(f"  multi-segment splits (any):     {n_split_into_multiple}")
    lines.append(f"  abbreviation-driven splits:     {sentences_with_abbrev_split}")
    lines.append(f"  no verb (parser):               {has_no_verb}")
    lines.append(f"  has SBAR / relative clause:     {has_relative_clause}")
    lines.append(f"  top-level S-coordination:       {has_top_level_coord}")
    lines.append(f"  subordinator-led sentence:      {has_subordinator_lead}")
    lines.append("")
    lines.append("=== surface markers in raw text ===")
    lines.append(f"  has quote (\"/“/”):  {has_quote}")
    lines.append(f"  has parens:        {has_paren}")
    lines.append(f"  has em/en dash:    {has_emdash}")
    lines.append(f"  has semicolon:     {has_semicolon}")
    lines.append(f"  has colon:         {has_colon}")
    lines.append(f"  has question mark: {has_question_mark}")
    lines.append(f"  has exclamation:   {has_exclamation}")
    lines.append(f"  has digit:         {has_number}")
    lines.append(f"  has URL remnant:   {has_inline_url_remnant}")
    lines.append(f"  has dialogue verb: {has_dialogue_verb}")
    lines.append(f"  contains ellipsis: {contains_ellipsis}")
    lines.append(f"  very short (≤3 words): {short_sentences}")
    lines.append(f"  very long  (≥60 words): {long_sentences}")

    lines.append("\n=== word-count distribution (5-bucket) ===")
    for bucket in sorted(word_count_dist):
        lines.append(f"  {bucket:>3}-{bucket+4:<3}: {word_count_dist[bucket]}")
    lines.append("\n=== parse-token-count distribution (10-bucket) ===")
    for bucket in sorted(token_count_dist):
        lines.append(f"  {bucket:>3}-{bucket+9:<3}: {token_count_dist[bucket]}")

    for cat, exs in edge_cases.items():
        if not exs:
            continue
        lines.append(f"\n--- examples: {cat} ---")
        for e in exs:
            lines.append(f"  {e}")

    lines.append("\n=== sample full-tree dumps ===\n")
    lines.extend(sample_dump)

    args.out.write_text("\n".join(lines))
    logger.info("wrote %s", args.out)
    print("\n".join(lines[:60]))


if __name__ == "__main__":
    main()

"""Build a sentence-level v14 corpus with fully nested constituency tags.

Reads source sentences (chunks/chunk_*.txt — already segmented + cleaned
upstream by the v13 pipeline), parses each with Stanza, and emits one
fully-tagged sample per surviving sentence in the form::

    <S> <NP> the cat </NP> <VP> sat <PP> on <NP> the mat </NP> </PP> </VP> . </S>

Wrapping rules (validated against the empirical probe in
output/v14_probe.txt):

  * Wrap every parser node whose label is in WRAP_LABELS (phrase-level
    only — POS-level tags like NN/VBD never wrap).
  * Punctuation leaves are emitted as bare tokens at whatever tree level
    the parser put them — never artificially shoved inside another phrase.
    The parser already follows PTB convention so periods naturally land
    as siblings under <S>, not inside <VP>.
  * Drop filters (in order): non-S root, no verb anywhere in tree,
    multi-segment Stanza split (abbreviation confusion), ≤ MIN_WORDS or
    ≥ MAX_WORDS, lone-quote/paren remnants from upstream cleaning.

Usage::

    poetry run python scripts/build_v14_sentences.py \\
        --chunks-dir data/datasets/english-constituents-v13/chunks \\
        --num-sentences 200 \\
        --out output/v14_test_corpus.txt \\
        --report output/v14_build_report.txt
"""

from __future__ import annotations

import argparse
import logging
import os
import re
from collections import Counter
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger(__name__)


# Phrase-level tags we wrap.  POS tags (NN, VBD, etc.) are intentionally
# skipped — wrapping every word doubles token count for no structural
# payoff.  Punctuation POS tags (`,`, `.`, `:`, etc.) emit bare leaves.
WRAP_LABELS = frozenset({
    "S", "SBAR", "SBARQ", "SINV", "SQ",
    "NP", "VP", "PP",
    "ADJP", "ADVP", "NML",
    "WHNP", "WHADJP", "WHADVP", "WHPP",
    "FRAG", "PRN", "INTJ",
})

# Tags whose presence flags drop-the-sample (not a real declarative).
NON_S_ROOTS = frozenset({"FRAG", "INTJ", "X"})

# Word-count window (after sentence-level normalization).
MIN_WORDS = 4
MAX_WORDS = 60

# Quote / paren residue we don't want at sample level.  Upstream
# pipeline already strips most; this is a final safety net.
_RE_RESIDUAL_QUOTE = re.compile(r"['\"`“”‘’]")
_RE_RESIDUAL_PAREN = re.compile(r"[()\[\]{}]")


def _is_wrappable(node) -> bool:
    return getattr(node, "label", "") in WRAP_LABELS


def _walk_emit(node) -> list[str]:
    """Render one parser node into the v14 wire format.

    Recursion strategy:
      * Phrase-level node whose label is in WRAP_LABELS:
            emit  "<LABEL>" + children + "</LABEL>"
      * POS-level node (one leaf child):
            emit  the leaf string itself.  Punctuation leaves become
            bare tokens (".", ",", ";", ":", etc.).
      * Otherwise (rare: ROOT, X, …): just descend into children.
    """
    label = getattr(node, "label", "")
    children = getattr(node, "children", []) or []
    # Leaf token: parser stores it as label string with no children.
    if not children:
        return [label.strip()] if label.strip() else []
    if label in WRAP_LABELS:
        out = [f"<{label}>"]
        for c in children:
            out.extend(_walk_emit(c))
        out.append(f"</{label}>")
        return out
    # POS-tag node: a non-wrap label with a single leaf child.
    if len(children) == 1 and not (getattr(children[0], "children", []) or []):
        return _walk_emit(children[0])
    # Anything else (ROOT, X, …): descend transparently.
    out: list[str] = []
    for c in children:
        out.extend(_walk_emit(c))
    return out


def _has_verb(node) -> bool:
    label = getattr(node, "label", "")
    if label.startswith("V") and len(label) <= 4:  # VB, VBD, VBN, VP, VBG, VBZ, VBP
        return True
    for c in getattr(node, "children", []) or []:
        if _has_verb(c):
            return True
    return False


def _drop_reason(raw: str, doc) -> str | None:
    n_words = len(raw.split())
    if n_words < MIN_WORDS:
        return "too_short"
    if n_words > MAX_WORDS:
        return "too_long"
    if len(doc.sentences) > 1:
        return "multi_segment"
    if _RE_RESIDUAL_QUOTE.search(raw):
        return "residual_quote"
    if _RE_RESIDUAL_PAREN.search(raw):
        return "residual_paren"
    sent = doc.sentences[0]
    tree = sent.constituency
    # ROOT → first child should be S (or one of its variants).  Accept
    # SBARQ / SQ / SINV as legit "sentence" shapes.  Drop FRAG/INTJ/X.
    children = getattr(tree, "children", []) or []
    if not children:
        return "empty_tree"
    top = getattr(children[0], "label", "")
    if top in NON_S_ROOTS or not (top.startswith("S")):
        return "non_s_root"
    if not _has_verb(tree):
        return "no_verb"
    return None


def _emit_sample(doc) -> str:
    tree = doc.sentences[0].constituency
    children = getattr(tree, "children", []) or []
    # The wrapper at top is always the parser's S (or S-variant).  Walk
    # the tree starting from that single S node so the output begins
    # with <S> not <ROOT> noise.
    top = children[0]
    tokens = _walk_emit(top)
    return " ".join(tokens)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--chunks-dir", type=Path,
                    default=Path("data/datasets/english-constituents-v13/chunks"))
    ap.add_argument("--num-sentences", type=int, default=100,
                    help="How many source sentences to attempt (default: 100).")
    ap.add_argument("--out", type=Path, default=Path("output/v14_test_corpus.txt"))
    ap.add_argument("--report", type=Path, default=Path("output/v14_build_report.txt"))
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    # Reservoir sample so the test set looks like the real corpus.
    import random
    rng = random.Random(args.seed)
    sentences: list[str] = []
    seen = 0
    paths = sorted(args.chunks_dir.glob("chunk_*.txt"))
    if not paths:
        raise SystemExit(f"no chunk_*.txt under {args.chunks_dir}")
    for p in paths:
        with p.open() as f:
            for line in f:
                line = line.strip()
                if not line or len(line) < 20:
                    continue
                seen += 1
                if len(sentences) < args.num_sentences:
                    sentences.append(line)
                else:
                    j = rng.randrange(seen)
                    if j < args.num_sentences:
                        sentences[j] = line
    logger.info("sampled %d / %d sentences across %d chunks",
                len(sentences), seen, len(paths))

    os.environ.setdefault("LFM_FORCE_STANZA", "1")
    import stanza
    logger.info("loading Stanza...")
    stanza.download("en", processors="tokenize,pos,constituency", verbose=False)
    nlp = stanza.Pipeline(
        "en",
        processors="tokenize,pos,constituency",
        use_gpu=False,
        verbose=False,
    )

    out_lines: list[str] = []
    drops: Counter = Counter()
    sample_examples: dict[str, list[tuple[str, str]]] = {}

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.report.parent.mkdir(parents=True, exist_ok=True)

    for idx, raw in enumerate(sentences):
        try:
            doc = nlp(raw)
        except Exception as e:
            drops["parse_error"] += 1
            logger.warning("parse failed: %s (%s)", raw[:80], e)
            continue
        reason = _drop_reason(raw, doc)
        if reason is not None:
            drops[reason] += 1
            sample_examples.setdefault(reason, []).append((raw, ""))
            continue
        line = _emit_sample(doc)
        out_lines.append(line)
        if "kept" not in sample_examples:
            sample_examples["kept"] = []
        if len(sample_examples["kept"]) < 5:
            sample_examples["kept"].append((raw, line))

    # Write outputs --------------------------------------------------
    args.out.write_text("\n".join(out_lines) + "\n")

    report = []
    report.append(f"v14 sentence build — {len(sentences)} attempted")
    report.append(f"  kept:    {len(out_lines)}")
    for r, c in drops.most_common():
        report.append(f"  drop[{r}]: {c}")
    report.append(f"\nkept rate: {100*len(out_lines)/max(1,len(sentences)):.1f}%")
    if out_lines:
        lens = [len(s.split()) for s in out_lines]
        report.append("\n=== token-with-tags length distribution ===")
        report.append(f"  min={min(lens)} max={max(lens)} mean={sum(lens)/len(lens):.1f}")
        for thr in (32, 64, 96, 128, 160, 200):
            pct = 100 * sum(1 for L in lens if L <= thr) / len(lens)
            report.append(f"  ≤{thr:>3}: {pct:5.1f}%")

    report.append("\n=== example transformations ===")
    for cat, exs in sample_examples.items():
        report.append(f"\n--- {cat} ---")
        for raw, line in exs[:5]:
            report.append(f"raw:  {raw[:200]}")
            if line:
                report.append(f"out:  {line[:300]}")
            report.append("")
    args.report.write_text("\n".join(report))

    logger.info("wrote %d samples → %s", len(out_lines), args.out)
    logger.info("wrote report → %s", args.report)
    print("\n".join(report[:30]))


if __name__ == "__main__":
    main()

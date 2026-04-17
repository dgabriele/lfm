"""Stream English sentences from HuggingFace datasets.

Splits paragraph-level text into sentences, applies heuristic filters
(no digits, no quotes, no fragments, length bounds), and writes
passing sentences to a file. SpaCy NER is run in a second pass.

Usage:
    poetry run python scripts/stream_hf_sentences.py \
        --dataset allenai/c4 --config en \
        --output /tmp/hf_sentences.txt \
        --target 5000000 --smoke 500
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import nltk
nltk.download("punkt_tab", quiet=True)


def has_digits(s: str) -> bool:
    return bool(re.search(r'\d', s))


def has_direct_quotes(s: str) -> bool:
    """Reject actual quotation marks but allow apostrophes in contractions."""
    if re.search(r'["""]', s):
        return True
    if "``" in s or "''" in s:
        return True
    # Reject dialogue-style speech attribution with single quotes
    if re.search(r"'\s*[A-Z]", s) and re.search(
        r'\b(said|asked|told|replied|exclaimed|shouted)\b', s, re.IGNORECASE
    ):
        return True
    return False


def has_name_initials(s: str) -> bool:
    """Reject sentences with name-initial patterns like D.C., U.S., J.K."""
    return bool(re.search(r'\b[A-Z]\.([A-Z]\.)+', s))


def has_special_chars(s: str) -> bool:
    return bool(re.search(
        r'http|www\.|\.com|\.org|\.net|\.edu|\.gov|\.io'
        r'|\{|\}|\\[a-z]|<[a-z]|@|\||\[|\]'
        r'|\.php|\.html|\.pdf|\.jpg|\.png',
        s.lower(),
    ))


def is_fragment(s: str) -> bool:
    s = s.strip()
    if not s or not s[0].isalpha():
        return True
    if not s.endswith((".", "!", "?")):
        return True
    if re.search(r'\b(i\.e|e\.g|etc|et al|viz|cf|vs)\s*\.\s*$', s):
        return True
    if re.search(r'[)\]]\s*\.?\s*$', s):
        return True
    words = s.split()
    if len(words) < 3:
        return True
    return False


def length_ok(s: str, min_w: int = 8, max_w: int = 35) -> bool:
    return min_w <= len(s.split()) <= max_w


_SPAM_PATTERNS = re.compile(
    r'click here|sign up|subscribe|buy now|order now|free shipping'
    r'|add to cart|checkout|coupon|promo code|discount code'
    r'|call us|contact us today|get in touch|request a quote'
    r'|we (offer|provide|deliver|specialize|guarantee)'
    r'|our (team|staff|experts|professionals|company) (will|can|is)'
    r'|your (order|subscription|account|purchase|payment)'
    r'|terms and conditions|privacy policy|cookie policy'
    r'|all rights reserved|copyright \d|permission',
    re.IGNORECASE,
)

_MARKETING_PATTERNS = re.compile(
    r"don't miss|limited time|act now|hurry|exclusive offer"
    r'|best (deal|price|value|offer)|lowest price'
    r'|satisfaction guaranteed|money back|risk.free'
    r'|award.winning|world.class|top.rated|best.in.class'
    r'|trusted by|as seen on|featured in',
    re.IGNORECASE,
)


def is_spam_or_marketing(s: str) -> bool:
    if _SPAM_PATTERNS.search(s):
        return True
    if _MARKETING_PATTERNS.search(s):
        return True
    # Second-person promotional address ("you can", "you will", "your")
    # with commercial context words
    if re.search(r'\byou(r)?\b', s, re.IGNORECASE):
        commercial = re.findall(
            r'\b(buy|purchase|order|subscribe|visit|download|install'
            r'|website|site|shop|store|product|service|price|cost'
            r'|shipping|delivery|customer|client)\b',
            s, re.IGNORECASE,
        )
        if len(commercial) >= 2:
            return True
    return False


def passes_heuristic(s: str) -> bool:
    s = s.strip()
    if not s:
        return False
    if is_fragment(s):
        return False
    if not length_ok(s):
        return False
    if has_digits(s):
        return False
    if has_direct_quotes(s):
        return False
    if has_name_initials(s):
        return False
    if has_special_chars(s):
        return False
    if is_spam_or_marketing(s):
        return False
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="allenai/c4")
    ap.add_argument("--config", default="en")
    ap.add_argument("--output", type=Path, default=Path("/tmp/hf_sentences.txt"))
    ap.add_argument("--target", type=int, default=5_000_000,
                    help="Stop after collecting this many sentences")
    ap.add_argument("--smoke", type=int, default=0,
                    help="If >0, just print this many passing sentences and exit")
    ap.add_argument("--max-docs", type=int, default=0,
                    help="Process at most this many documents (0=unlimited)")
    ap.add_argument("--skip-docs", type=int, default=0,
                    help="Skip this many documents from the start of the stream")
    args = ap.parse_args()

    from datasets import load_dataset
    from nltk.tokenize import sent_tokenize

    print(f"Streaming from {args.dataset} (config={args.config})...")
    ds = load_dataset(args.dataset, args.config, split="train", streaming=True)

    collected = 0
    docs_seen = 0
    total_sents = 0
    out = None if args.smoke else open(args.output, "w")

    try:
        for row in ds:
            text = row.get("text", "")
            if not text:
                continue
            docs_seen += 1
            if args.skip_docs and docs_seen <= args.skip_docs:
                if docs_seen % 500_000 == 0:
                    print(f"  skipping... {docs_seen:,} / {args.skip_docs:,}")
                continue

            sents = sent_tokenize(text)
            for s in sents:
                s = s.strip()
                total_sents += 1
                if passes_heuristic(s):
                    if args.smoke:
                        print(f"  {collected+1}. {s}")
                        collected += 1
                        if collected >= args.smoke:
                            raise StopIteration
                    else:
                        out.write(s + "\n")
                        collected += 1

            if collected % 100_000 == 0 and collected > 0 and not args.smoke:
                pct = 100 * collected / total_sents if total_sents else 0
                print(f"  {collected:,} collected from {docs_seen:,} docs "
                      f"({total_sents:,} sentences seen, {pct:.1f}% pass rate)")

            if args.target and collected >= args.target:
                break
            if args.max_docs and docs_seen >= args.max_docs:
                break

    except StopIteration:
        pass
    finally:
        if out:
            out.close()

    pct = 100 * collected / total_sents if total_sents else 0
    print(f"\nDone: {collected:,} sentences from {docs_seen:,} docs "
          f"({total_sents:,} total, {pct:.1f}% pass rate)")
    if not args.smoke:
        print(f"Written to {args.output}")


if __name__ == "__main__":
    main()

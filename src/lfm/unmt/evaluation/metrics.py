"""Evaluation metrics for unsupervised NMT.

With no paired reference data available, we cannot compute standard
BLEU against ground truth.  The two diagnostics that matter in the
unsupervised regime are:

1. **Round-trip BLEU** — translate a monolingual sentence into the
   other language and back, and measure BLEU between the final output
   and the original.  High round-trip BLEU means the encoder preserves
   meaning across the round trip, which is a necessary (but not
   sufficient) condition for translation quality.
2. **Sample inspection** — a small set of held-out source sentences
   is translated and logged so the operator can eyeball the output.

Both are implemented here; the CLI ``translate`` command uses these
for its sanity output.

BLEU is the simple corpus-level variant using uniform n-gram weights
(1–4) and brevity penalty.  We avoid dragging in ``sacrebleu`` to
keep the dependency surface small.
"""

from __future__ import annotations

import math
from collections import Counter


def _ngrams(tokens: list[str], n: int) -> Counter:
    return Counter(tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1))


def corpus_bleu(
    references: list[list[str]],
    hypotheses: list[list[str]],
    max_n: int = 4,
) -> float:
    """Uniform-weight corpus BLEU with brevity penalty.

    Args:
        references: One list of reference tokens per example (single
            reference, not multi-reference).
        hypotheses: One list of hypothesis tokens per example.
        max_n: Maximum n-gram order (default 4 = BLEU-4).

    Returns:
        BLEU score in ``[0, 1]``.  ``0`` for the degenerate case of
        zero or empty hypotheses.
    """
    if len(references) != len(hypotheses):
        raise ValueError("references and hypotheses must be the same length")
    if not hypotheses:
        return 0.0

    clipped_counts = [0] * max_n
    total_counts = [0] * max_n
    ref_len = 0
    hyp_len = 0

    for ref_tokens, hyp_tokens in zip(references, hypotheses):
        ref_len += len(ref_tokens)
        hyp_len += len(hyp_tokens)
        for n in range(1, max_n + 1):
            hyp_ngrams = _ngrams(hyp_tokens, n)
            ref_ngrams = _ngrams(ref_tokens, n)
            for gram, count in hyp_ngrams.items():
                clipped_counts[n - 1] += min(count, ref_ngrams.get(gram, 0))
            total_counts[n - 1] += max(len(hyp_tokens) - n + 1, 0)

    if hyp_len == 0:
        return 0.0

    precisions: list[float] = []
    for clipped, total in zip(clipped_counts, total_counts):
        if total == 0:
            precisions.append(0.0)
        else:
            precisions.append(clipped / total)

    if min(precisions) == 0:
        # Geometric mean collapses.  Use smoothed log to avoid -inf.
        precisions = [max(p, 1e-12) for p in precisions]

    log_avg = sum(math.log(p) for p in precisions) / max_n
    geo_mean = math.exp(log_avg)

    brevity = 1.0 if hyp_len > ref_len else math.exp(1 - ref_len / max(hyp_len, 1))
    return brevity * geo_mean


def round_trip_bleu(
    translate_fn,
    sources: list[str],
    source_lang: str,
    target_lang: str,
) -> tuple[float, list[tuple[str, str, str]]]:
    """Compute round-trip BLEU via ``translate_fn``.

    ``translate_fn(texts, src, tgt)`` must translate a list of strings
    from ``src`` to ``tgt``.

    Returns:
        Tuple ``(bleu, samples)`` where ``samples`` is a list of
        ``(original, intermediate, round_trip)`` triples for the first
        few examples — useful for logging.
    """
    intermediate = translate_fn(sources, source_lang, target_lang)
    round_trip = translate_fn(intermediate, target_lang, source_lang)

    refs_tokens = [s.split() for s in sources]
    hyps_tokens = [s.split() for s in round_trip]
    bleu = corpus_bleu(refs_tokens, hyps_tokens)

    samples = list(zip(sources, intermediate, round_trip))[:5]
    return bleu, samples

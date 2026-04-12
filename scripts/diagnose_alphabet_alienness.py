#!/usr/bin/env python
"""Compare Qwen's response to pseudowords from different script alphabets.

We want a principled answer to: does Cyrillic/Greek give us a more alien
(lower-prior) substrate than ASCII for our Neuroglot alphabet?

Method
------
Sample pseudowords from four alphabets:

  1. ASCII-phoneme    — our current curated 50-phoneme set, composed
                        into random 3-phoneme words
  2. ASCII-random     — random 3-char strings over a-z (control for
                        "any ASCII alien text")
  3. Cyrillic-random  — random 3-char strings from Cyrillic letters,
                        excluding lookalikes that collide with Latin
  4. English-real     — 50 real English words (control showing what
                        "prior activated" looks like)

For each pseudoword, compute two signals embedded in a generic carrier
sentence ("The new word here is <W>. It means"):

  a) **word_nll**         — sum negative log-likelihood of the pseudoword
                            tokens given the prefix.  Higher = more
                            surprising = less prior.
  b) **next_entropy**     — Shannon entropy of Qwen's next-token
                            distribution after the pseudoword.  Higher =
                            Qwen less certain what comes next = more alien.
  c) **next_top5**        — top-5 most-likely next tokens, for qualitative
                            inspection of whether Qwen is activating
                            English morphology priors.

If Cyrillic-random has substantially higher word_nll AND higher
next_entropy than ASCII-phoneme, the script-mix alphabet is a better
substrate.  If they're comparable, ASCII is sufficient and we avoid the
multi-script risks.
"""

from __future__ import annotations

import json
import random
import string
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_NAME = "Qwen/Qwen2.5-0.5B"
ALPHABET_PATH = Path("data/phoneme_alphabet_v1.json")
N_WORDS = 50
WORD_PHONEMES = 3
CARRIER = "The new word here is"  # no trailing space — word adds its own
SEED = 42

# Cyrillic letters.  We exclude lookalikes that Qwen's tokenizer may
# normalise to Latin (а/е/о/р/с/у/х/к/н/м/т) — we want unambiguously
# Cyrillic code points.  Keep the 17 letters that are visually distinct
# from Latin.
CYRILLIC_DISTINCT = list("бгджзийлпфцчшщъыьэюя")

# Reference English words spanning common morphological patterns.
ENGLISH_WORDS = [
    "table", "problem", "company", "system", "water", "light", "number",
    "mother", "father", "school", "house", "reason", "student", "doctor",
    "history", "music", "nature", "country", "service", "morning",
    "evening", "family", "friend", "story", "market", "garden", "letter",
    "window", "picture", "kitchen", "summer", "winter", "bridge", "island",
    "village", "forest", "mountain", "valley", "river", "flower", "animal",
    "medicine", "science", "language", "machine", "culture", "journey",
    "chapter", "question", "silence",
]


def sample_ascii_phoneme_words(phonemes: list[str], n: int) -> list[str]:
    return ["".join(random.choice(phonemes) for _ in range(WORD_PHONEMES))
            for _ in range(n)]


def sample_ascii_random_words(n: int, char_len: int = 8) -> list[str]:
    return ["".join(random.choice(string.ascii_lowercase) for _ in range(char_len))
            for _ in range(n)]


def sample_cyrillic_random_words(n: int, char_len: int = 6) -> list[str]:
    # Cyrillic letters are 2-byte UTF-8; char_len=6 gives ~similar
    # byte-length to 8-char ASCII.
    return ["".join(random.choice(CYRILLIC_DISTINCT) for _ in range(char_len))
            for _ in range(n)]


@torch.no_grad()
def score_word(model, tok, prefix: str, word: str, device) -> dict:
    """Compute word NLL and next-token entropy + top-5 after the word."""
    prefix_ids = tok.encode(prefix, add_special_tokens=False)
    full_ids = tok.encode(prefix + " " + word, add_special_tokens=False)
    # Sanity: full_ids should start with prefix_ids.
    if full_ids[:len(prefix_ids)] != prefix_ids:
        # Boundary mismatch — re-align by finding longest common prefix.
        k = 0
        while k < min(len(prefix_ids), len(full_ids)) and prefix_ids[k] == full_ids[k]:
            k += 1
        prefix_len = k
    else:
        prefix_len = len(prefix_ids)
    word_ids = full_ids[prefix_len:]
    if not word_ids:
        return {"word_nll": float("nan"), "next_entropy": float("nan"),
                "n_word_tokens": 0, "top5": []}

    input_ids = torch.tensor([full_ids], device=device)
    logits = model(input_ids).logits[0].float()  # (seq, vocab), float32 for stability

    # NLL of each word token given everything before it.
    nll = 0.0
    for i, wid in enumerate(word_ids):
        pos = prefix_len + i - 1
        logp = F.log_softmax(logits[pos], dim=-1)
        nll -= logp[wid].item()

    # Entropy of next-token distribution AFTER the word.
    last = logits[-1]
    probs = F.softmax(last, dim=-1)
    entropy = -(probs * torch.log(probs + 1e-12)).sum().item()

    top5_ids = torch.topk(probs, k=5).indices.tolist()
    top5 = [(tok.decode([tid]).replace("\n", "\\n"), probs[tid].item())
            for tid in top5_ids]

    return {
        "word_nll": nll,
        "next_entropy": entropy,
        "n_word_tokens": len(word_ids),
        "top5": top5,
    }


def summarize(label: str, scores: list[dict]) -> dict:
    nlls = [s["word_nll"] for s in scores]
    ents = [s["next_entropy"] for s in scores]
    ntoks = [s["n_word_tokens"] for s in scores]
    nlls_per_tok = [n / max(t, 1) for n, t in zip(nlls, ntoks)]
    return {
        "label": label,
        "n": len(scores),
        "mean_word_nll": sum(nlls) / len(nlls),
        "mean_nll_per_token": sum(nlls_per_tok) / len(nlls_per_tok),
        "mean_n_tokens": sum(ntoks) / len(ntoks),
        "mean_next_entropy": sum(ents) / len(ents),
    }


def main() -> None:
    random.seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading {MODEL_NAME} on {device}...")
    tok = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, torch_dtype=torch.float16 if device.type == "cuda" else torch.float32,
    ).to(device).eval()

    with ALPHABET_PATH.open() as f:
        alphabet = json.load(f)
    phonemes = alphabet["phonemes"]
    print(f"Loaded {len(phonemes)} phonemes from {ALPHABET_PATH}")

    cohorts = {
        "ASCII-phoneme":   sample_ascii_phoneme_words(phonemes, N_WORDS),
        "ASCII-random":    sample_ascii_random_words(N_WORDS),
        "Cyrillic-random": sample_cyrillic_random_words(N_WORDS),
        "English-real":    random.sample(ENGLISH_WORDS, N_WORDS),
    }

    print("\nSampled 5 examples per cohort:")
    for name, words in cohorts.items():
        print(f"  {name:>17}: {words[:5]}")

    summary: list[dict] = []
    qualitative: dict[str, list] = {}
    print(f"\nScoring each cohort ({N_WORDS} words × carrier '{CARRIER}')...")
    for name, words in cohorts.items():
        scores = []
        for w in words:
            s = score_word(model, tok, CARRIER, w, device)
            s["word"] = w
            scores.append(s)
        stats = summarize(name, scores)
        summary.append(stats)
        qualitative[name] = scores[:3]  # save 3 examples per cohort
        print(
            f"  {name:>17}: mean_nll={stats['mean_word_nll']:.2f}  "
            f"nll/tok={stats['mean_nll_per_token']:.2f}  "
            f"mean_n_tok={stats['mean_n_tokens']:.2f}  "
            f"next_ent={stats['mean_next_entropy']:.3f}",
        )

    print("\nQualitative — top-5 next-token predictions after one word per cohort:")
    for name, scores in qualitative.items():
        s = scores[0]
        top5_str = ", ".join(f"{tstr!r}={p:.3f}" for tstr, p in s["top5"])
        print(f"  [{name}] word={s['word']!r}  top5: {top5_str}")

    # Interpretation rule of thumb:
    # - word_nll is in bits-equivalent.  Higher = more surprising.
    # - English-real should have the LOWEST word_nll (priors present).
    # - Maximally alien substrate should have the HIGHEST word_nll and
    #   HIGHEST next_entropy.
    print("\nInterpretation guide:")
    print("  - higher mean_word_nll  → pseudoword more surprising (weaker priors)")
    print("  - higher mean_next_entropy → Qwen less confident what follows (more alien)")

    # Sort cohorts by mean_nll_per_token for clear ranking
    ranked = sorted(summary, key=lambda s: s["mean_nll_per_token"], reverse=True)
    print("\nRanked by alienness (nll_per_token, high→low):")
    for r in ranked:
        print(f"  {r['label']:>17}: {r['mean_nll_per_token']:.2f} nll/tok  "
              f"ent={r['mean_next_entropy']:.3f}")

    output = Path("data/alphabet_alienness_diagnostic.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w") as f:
        json.dump({
            "model": MODEL_NAME,
            "carrier": CARRIER,
            "n_words_per_cohort": N_WORDS,
            "summary": summary,
            "qualitative_examples": qualitative,
        }, f, indent=2, default=str)
    print(f"\nWrote diagnostic to {output}")


if __name__ == "__main__":
    main()

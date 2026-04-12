#!/usr/bin/env python
"""Measure which Qwen attention mode each Neuroglot rendering triggers.

The phoneme VAE produces phoneme-id sequences; we can render them with
different within-word separators (empty / hyphen / pipe / space).  The
question this script answers empirically: does Qwen read pipe-separated
Neuroglot as LANGUAGE (our goal) or as CODE / STRUCTURED DATA (which
would defeat the architectural-leverage argument for using a Latin
word-like alphabet)?

Method: feed identical content in each rendering to Qwen with a prose
carrier sentence, then inspect Qwen's next-token distribution after the
sample.  Language-mode produces continuation candidates like ' the',
'.', ' is', etc.  Code-mode produces '|', ',', newlines, operators.

No speculation — we just print the empirical distributions.
"""

from __future__ import annotations

import json
import random
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = "Qwen/Qwen2.5-0.5B"
ALPHABET_PATH = Path("data/phoneme_alphabet_multi.json")
N_SAMPLES = 20
WORDS_PER_SAMPLE = 4       # Neuroglot words per sample
PHONEMES_PER_WORD = 3
SEED = 42

CARRIER = "The new word here is"  # sample is inserted after, prose context

# Classify top-token predictions as prose vs code
PROSE_TOKEN_HINTS = {
    ".", ",", "?", "!", ";", ":", "...", "'s",
    " the", " a", " an", " and", " or", " but", " is", " was", " are", " were",
    " of", " in", " on", " at", " to", " for", " with", " by", " from",
    " it", " he", " she", " they", " we", " you", " I",
    " this", " that", " these", " those",
    " which", " who", " what", " when", " where", " why", " how",
    "\n", "\n\n",
}
CODE_TOKEN_HINTS = {
    "|", "||", "&", "&&", "==", "!=", ">=", "<=", "->", "=>",
    "(", ")", "[", "]", "{", "}", "<", ">",
    "//", "/*", "*/", "#", ";;",
    "\\", "$", "`", "~",
    # variable/tech flavors
    " None", " null", " true", " false", " True", " False",
    " return", " def", " class", " import", " function",
}


def classify(tok_str: str) -> str:
    if tok_str in PROSE_TOKEN_HINTS:
        return "prose"
    if tok_str in CODE_TOKEN_HINTS:
        return "code"
    return "other"


def render_concat(words: list[list[str]]) -> str:
    return " ".join("".join(w) for w in words)


def render_hyphen(words: list[list[str]]) -> str:
    return " ".join("-".join(w) for w in words)


def render_pipe(words: list[list[str]]) -> str:
    return " ".join("|".join(w) for w in words)


def render_space(words: list[list[str]]) -> str:
    # Each phoneme as its own space-separated token; word boundaries lost
    return " ".join(ph for w in words for ph in w)


@torch.no_grad()
def score_sample(model, tok, prefix: str, sample: str, device) -> dict:
    """Compute word NLL + next-token distribution + prose/code hint counts."""
    prefix_ids = tok.encode(prefix, add_special_tokens=False)
    full_ids = tok.encode(prefix + " " + sample, add_special_tokens=False)
    prefix_len = len(prefix_ids)
    if full_ids[:prefix_len] != prefix_ids:
        k = 0
        while k < min(len(prefix_ids), len(full_ids)) and prefix_ids[k] == full_ids[k]:
            k += 1
        prefix_len = k

    input_ids = torch.tensor([full_ids], device=device)
    logits = model(input_ids).logits[0].float()

    word_ids = full_ids[prefix_len:]
    n_word_tokens = len(word_ids)
    nll = 0.0
    for i, wid in enumerate(word_ids):
        pos = prefix_len + i - 1
        logp = F.log_softmax(logits[pos], dim=-1)
        nll -= logp[wid].item()

    last = logits[-1]
    probs = F.softmax(last, dim=-1)
    entropy = -(probs * torch.log(probs + 1e-12)).sum().item()
    top20_ids = torch.topk(probs, k=20).indices.tolist()
    top20 = [(tok.decode([tid]).replace("\n", "\\n"), probs[tid].item())
             for tid in top20_ids]

    prose_mass = sum(p for s, p in top20 if classify(s) == "prose")
    code_mass = sum(p for s, p in top20 if classify(s) == "code")
    return {
        "sample": sample,
        "n_word_tokens": n_word_tokens,
        "nll_per_token": nll / max(n_word_tokens, 1),
        "entropy": entropy,
        "prose_mass": prose_mass,
        "code_mass": code_mass,
        "top5": top20[:5],
    }


def main() -> None:
    random.seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading {MODEL} on {device}...")
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, torch_dtype=(torch.float16 if device.type == "cuda" else torch.float32),
    ).to(device).eval()

    phonemes = json.loads(ALPHABET_PATH.read_text())["phonemes"]

    # Build N samples as lists of words (each word = list of phonemes)
    samples: list[list[list[str]]] = []
    for _ in range(N_SAMPLES):
        words = [
            [random.choice(phonemes) for _ in range(PHONEMES_PER_WORD)]
            for _ in range(WORDS_PER_SAMPLE)
        ]
        samples.append(words)

    cohorts = {
        "CONCAT   (ottellochogrvak)":  [render_concat(s) for s in samples],
        "HYPHEN   (ott-ell-och-ogr-vak)": [render_hyphen(s) for s in samples],
        "PIPE     (ott|ell|och|ogr|vak)": [render_pipe(s) for s in samples],
        "SPACE    (ott ell och ogr vak)": [render_space(s) for s in samples],
    }
    # English control
    ENGLISH = [
        "morning coffee turned cold", "the garden beyond the fence",
        "a very small blue house", "walking slowly down the street",
        "silent kitchen at dusk", "wind moved across the field",
        "children laughed in distance", "rain against the old window",
        "letters arriving in envelopes", "music drifting from upstairs",
    ] * 2
    cohorts["ENGLISH  (control)"] = ENGLISH[:N_SAMPLES]

    print(f"\nCarrier: {CARRIER!r}")
    print(f"Samples per cohort: {N_SAMPLES}\n")

    for name, cohort in cohorts.items():
        scores = [score_sample(model, tok, CARRIER, s, device) for s in cohort]
        mean_nll = sum(s["nll_per_token"] for s in scores) / len(scores)
        mean_ent = sum(s["entropy"] for s in scores) / len(scores)
        mean_prose = sum(s["prose_mass"] for s in scores) / len(scores)
        mean_code = sum(s["code_mass"] for s in scores) / len(scores)
        mean_toks = sum(s["n_word_tokens"] for s in scores) / len(scores)
        print(f"  {name}")
        print(f"    mean_nll/tok={mean_nll:.2f}  n_tokens={mean_toks:.1f}  "
              f"next_entropy={mean_ent:.2f}")
        print(f"    next-token mass: prose={mean_prose:.3f}  code={mean_code:.3f}  "
              f"(higher prose = more language-mode)")
        # Show example top-5 for first sample
        top5 = scores[0]["top5"]
        top5_str = ", ".join(f"{s!r}={p:.2f}" for s, p in top5)
        print(f"    example[0] top-5: {top5_str}")
        print()

    print("Interpretation: higher prose mass / lower code mass = Qwen reading")
    print("the sample in language mode (our goal).  If PIPE cohort shows code")
    print("mass materially higher than ENGLISH baseline, the pipe separator is")
    print("shifting Qwen out of language mode.")


if __name__ == "__main__":
    main()

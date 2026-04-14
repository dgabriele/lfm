"""Compare Qwen interpretation on raw IPA vs p2g-spelled variants.

Takes plain English sentences, converts each to (a) raw IPA (CMUDict
word-by-word joined with spaces) and (b) p2g-spelled approximate
English, then asks zero-shot Qwen-Instruct to describe each.  Shows
side-by-side what Qwen makes of the two surface forms.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_qwen_p2g_spelling import load_p2g, p2g_spell
import test_qwen_p2g_spelling as tm

from lfm.data.loaders.ipa import IPAConverter

# Use v11 p2g
tm.CKPT_PATH = "data/models/p2g_v11/latest.pt"

SENTENCES = [
    "The cat sat on the mat and watched the bird outside the window.",
    "Consciousness is a fundamental property of the physical universe.",
    "The doctor carefully examined the patient after the accident.",
    "She opened her laptop and began writing the quarterly report.",
    "Heavy rain flooded the streets of the small coastal village.",
    "The old professor delivered his final lecture on ancient Greek philosophy.",
    "Children laughed as they chased each other through the playground.",
    "Engineers redesigned the bridge to withstand stronger earthquakes.",
]


def to_ipa(conv, sentence):
    words = sentence.strip(".?!").split()
    out = []
    for w in words:
        clean = w.strip(",;:'\"()").lower()
        ipa = conv._english_word_to_ipa(clean)  # noqa: SLF001
        out.append(ipa if ipa else clean)
    return " ".join(out)


def to_p2g(conv, sentence, p2g, ipa_vocab, sp_vocab, cfg, device):
    words = sentence.strip(".?!").split()
    out = []
    for w in words:
        clean = w.strip(",;:'\"()").lower()
        ipa = conv._english_word_to_ipa(clean)  # noqa: SLF001
        if ipa is None:
            out.append(clean)
        else:
            out.append(p2g_spell(ipa, p2g, ipa_vocab, sp_vocab, cfg, device))
    return " ".join(out)


def interpret(model, tok, text):
    msg = [
        {"role": "system", "content": (
            "You will be given a short passage written with approximate "
            "phonetic English spelling.  Read it and describe, in ONE "
            "short English sentence, what the passage is about.  Do not "
            "try to rewrite it — just say what situation or topic it "
            "describes."
        )},
        {"role": "user", "content": f"{text}\n\nWhat is this about?"},
    ]
    prompt = tok.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)
    inputs = tok(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(
            **inputs, max_new_tokens=60, temperature=0.7, top_p=0.9,
            do_sample=True, repetition_penalty=1.1, pad_token_id=tok.pad_token_id,
        )
    return tok.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    conv = IPAConverter(drop_unconvertible=False)
    print("Loading p2g v11…")
    p2g, ipa_vocab, sp_vocab, cfg = load_p2g(device)
    print("Loading Qwen-Instruct…")
    qtok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")
    qmodel = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen2.5-0.5B-Instruct", torch_dtype=torch.bfloat16,
    ).to(device).eval()

    for s in SENTENCES:
        ipa = to_ipa(conv, s)
        spelled = to_p2g(conv, s, p2g, ipa_vocab, sp_vocab, cfg, device)
        q_ipa = interpret(qmodel, qtok, ipa)
        q_spelled = interpret(qmodel, qtok, spelled)
        print("\n" + "=" * 78)
        print(f"ORIG     : {s}")
        print(f"IPA      : {ipa}")
        print(f"P2G      : {spelled}")
        print(f"Qwen-IPA : {q_ipa}")
        print(f"Qwen-P2G : {q_spelled}")


if __name__ == "__main__":
    main()

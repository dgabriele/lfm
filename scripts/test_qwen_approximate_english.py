"""Quick test: can Qwen zero-shot interpret approximate-English spelling?

Takes a handful of normal English sentences, converts each word to IPA via
CMUDict (English only), romanizes IPA to ASCII, then asks Qwen-Instruct
to describe what the result is about.  If Qwen does well, the IPA →
romanize path is enough for the LFM project and we can skip the whole
BPE-vocabulary detour.
"""

from __future__ import annotations

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from lfm.data.loaders.ipa import IPAConverter
from lfm.translator.romanize import romanize_iso

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


def to_approx_english(conv: IPAConverter, text: str) -> str:
    """English → IPA (CMUDict) → ASCII romanization, word-by-word."""
    words = text.strip(".?!").split()
    out: list[str] = []
    for w in words:
        clean = w.strip(",;:'\"()").lower()
        ipa = conv._english_word_to_ipa(clean)  # noqa: SLF001
        if ipa is None:
            out.append(clean)  # fall back to original spelling
        else:
            out.append(romanize_iso(ipa))
    return " ".join(out)


def interpret(model, tok, text: str) -> str:
    messages = [
        {"role": "system", "content": (
            "You will be given a short passage written with approximate "
            "phonetic English spelling (words spelled close to how they "
            "sound, not standardly).  Read it and describe, in ONE short "
            "English sentence, what the passage is about.  Do not try to "
            "transcribe it back to standard English — just say what "
            "situation or topic it describes."
        )},
        {"role": "user", "content": f"{text}\n\nWhat is this about?"},
    ]
    prompt = tok.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True,
    )
    inputs = tok(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=60,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
            repetition_penalty=1.1,
            pad_token_id=tok.pad_token_id,
        )
    return tok.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()


def main() -> None:
    print("Loading Qwen-Instruct + IPA converter…")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    qtok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")
    qmodel = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen2.5-0.5B-Instruct", torch_dtype=torch.bfloat16,
    ).to(device).eval()
    conv = IPAConverter(drop_unconvertible=False)

    for s in SENTENCES:
        approx = to_approx_english(conv, s)
        interp = interpret(qmodel, qtok, approx)
        print("\n" + "=" * 66)
        print(f"ORIG  : {s}")
        print(f"APPROX: {approx}")
        print(f"QWEN  : {interp}")


if __name__ == "__main__":
    main()

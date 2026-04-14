"""Can Qwen interpret sentences spelled via our trained p2g seq2seq?

Uses the v10 p2g seq2seq (word-level IPA → English spelling) to render
each word of a normal English sentence into its approximate spelling,
then asks Qwen what the sentence is about.  Unlike the earlier
``romanize_iso`` path (literal IPA→ASCII that looks alien), this uses
a learned model that emits real-English-looking spellings.
"""

from __future__ import annotations

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from lfm.data.loaders.ipa import IPAConverter
from lfm.p2g.seq2seq import P2GSeq2Seq, P2GSeq2SeqConfig
from lfm.p2g.vocab import CharVocab

CKPT_PATH = "data/models/p2g_seq2seq/best.pt"

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


def load_p2g(device: torch.device):
    ck = torch.load(CKPT_PATH, map_location=device, weights_only=False)
    ipa_vocab = CharVocab(
        chars=ck["ipa_vocab"],
        char_to_id={c: i for i, c in enumerate(ck["ipa_vocab"])},
    )
    sp_vocab = CharVocab(
        chars=ck["sp_vocab"],
        char_to_id={c: i for i, c in enumerate(ck["sp_vocab"])},
    )
    cfg = ck["cfg"]
    model_cfg = P2GSeq2SeqConfig(
        input_vocab_size=ipa_vocab.size,
        output_vocab_size=sp_vocab.size,
        d_model=cfg["decoder_dim"],
        encoder_layers=cfg["encoder_layers"],
        decoder_layers=cfg["decoder_layers"],
        nhead=cfg["decoder_heads"],
        max_ipa_len=cfg["max_ipa_len"],
        max_spelling_len=cfg["max_spelling_len"],
        dropout=cfg["dropout"],
    )
    model = P2GSeq2Seq(model_cfg).to(device).eval()
    model.load_state_dict(ck["model_state"])
    return model, ipa_vocab, sp_vocab, cfg


def p2g_spell(word_ipa: str, model, ipa_vocab, sp_vocab, cfg, device):
    from lfm.p2g.vocab import PAD_ID
    enc = ipa_vocab.encode(word_ipa)[: cfg["max_ipa_len"]]
    ids = torch.full((1, cfg["max_ipa_len"]), PAD_ID, dtype=torch.long, device=device)
    ids[0, : len(enc)] = torch.tensor(enc, device=device)
    pred = model.generate(ids)[0]
    return sp_vocab.decode(pred)


def to_approx_english(conv, sentence, model, ipa_vocab, sp_vocab, cfg, device):
    words = sentence.strip(".?!").split()
    out: list[str] = []
    for w in words:
        clean = w.strip(",;:'\"()").lower()
        ipa = conv._english_word_to_ipa(clean)  # noqa: SLF001
        if ipa is None:
            out.append(clean)
        else:
            out.append(p2g_spell(ipa, model, ipa_vocab, sp_vocab, cfg, device))
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
    print("Loading p2g seq2seq…")
    p2g_model, ipa_vocab, sp_vocab, p2g_cfg = load_p2g(device)
    print("Loading Qwen-Instruct…")
    qtok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")
    qmodel = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen2.5-0.5B-Instruct", torch_dtype=torch.bfloat16,
    ).to(device).eval()

    for s in SENTENCES:
        approx = to_approx_english(conv, s, p2g_model, ipa_vocab, sp_vocab, p2g_cfg, device)
        interp = interpret(qmodel, qtok, approx)
        print("\n" + "=" * 66)
        print(f"ORIG  : {s}")
        print(f"APPROX: {approx}")
        print(f"QWEN  : {interp}")


if __name__ == "__main__":
    main()

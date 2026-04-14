#!/usr/bin/env python
"""Train an English IPA→spelling (p2g) seq2seq from CMUDict.

Pipeline:
  1. Load CMUDict (~123K English words with ARPABET pronunciations).
  2. Convert ARPABET → IPA via standard mapping table.
  3. Build (IPA-string, spelling-string) training pairs.
  4. Fine-tune ``zanegraper/t5-small-ipa-phoneme-to-text`` (which already
     knows the IPA→text task) on cleaner CMUDict pairs to improve
     real-word recovery + tighten OOD pseudoword behavior.
  5. Save to ``data/models/p2g_english/`` (HF format, drop-in usable).

Why fine-tune rather than train from scratch:
  - The base model already has the IPA character vocabulary embedded;
    starting from a working IPA→text mapping converges much faster than
    teaching it from random init.
  - We get ~30 min training instead of ~2-3hr.

Determinism: greedy decoding (do_sample=False) makes the trained model
deterministic at inference time.
"""

from __future__ import annotations

import logging
import random
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger(__name__)

BASE_MODEL = "zanegraper/t5-small-ipa-phoneme-to-text"
OUTPUT_DIR = Path("data/models/p2g_english")
EPOCHS = 3
BATCH_SIZE = 64
LR = 5e-5
WARMUP_FRACTION = 0.05
MAX_LEN = 32
SEED = 42


# ARPABET → IPA mapping (no stress markers in IPA).  Standard table.
ARPA_TO_IPA = {
    # Vowels (with stress markers stripped at use time)
    "AA": "ɑ", "AE": "æ", "AH": "ʌ", "AO": "ɔ",
    "AW": "aʊ", "AY": "aɪ", "EH": "ɛ", "ER": "ɝ",
    "EY": "eɪ", "IH": "ɪ", "IY": "i", "OW": "oʊ",
    "OY": "ɔɪ", "UH": "ʊ", "UW": "u",
    # Consonants
    "B": "b", "CH": "tʃ", "D": "d", "DH": "ð",
    "F": "f", "G": "ɡ", "HH": "h", "JH": "dʒ",
    "K": "k", "L": "l", "M": "m", "N": "n",
    "NG": "ŋ", "P": "p", "R": "ɹ", "S": "s",
    "SH": "ʃ", "T": "t", "TH": "θ", "V": "v",
    "W": "w", "Y": "j", "Z": "z", "ZH": "ʒ",
}


def arpa_to_ipa(arpa_phonemes: list[str]) -> str:
    """Convert ARPABET phoneme list to IPA string (stress markers dropped)."""
    out: list[str] = []
    for ph in arpa_phonemes:
        # Strip stress digit suffix (AH0, AH1, AH2 → AH)
        bare = ph.rstrip("0123456789")
        if bare in ARPA_TO_IPA:
            out.append(ARPA_TO_IPA[bare])
    return "".join(out)


class P2GDataset(Dataset):
    def __init__(self, pairs: list[tuple[str, str]], tokenizer, max_len: int):
        self.pairs = pairs
        self.tok = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        ipa, spelling = self.pairs[idx]
        # Encoder input: IPA string.  Decoder target: spelling.
        # Modern HF API uses text_target= instead of as_target_tokenizer().
        enc = self.tok(
            text=ipa,
            text_target=spelling,
            max_length=self.max_len, truncation=True,
            padding="max_length", return_tensors="pt",
        )
        labels = enc["labels"].squeeze(0)
        # Replace pad token id with -100 so loss ignores it
        labels[labels == self.tok.pad_token_id] = -100
        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "labels": labels,
        }


def main() -> None:
    torch.manual_seed(SEED)
    random.seed(SEED)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # ------ Load CMUDict and build pairs ------
    import nltk
    try:
        from nltk.corpus import cmudict
        cmu = cmudict.dict()
    except (LookupError, OSError):
        nltk.download("cmudict", quiet=True)
        from nltk.corpus import cmudict
        cmu = cmudict.dict()
    logger.info(f"CMUDict loaded: {len(cmu):,} words")

    pairs: list[tuple[str, str]] = []
    for word, prons in cmu.items():
        # Filter: lowercase letters only, length > 1, not all digits
        if not word.isalpha() or len(word) < 2 or not word.islower():
            continue
        # Take the first pronunciation (most common variant).  CMUDict
        # gives multiple for some words; we want one (IPA, spelling) per
        # word for deterministic mapping.
        ipa = arpa_to_ipa(prons[0])
        if not ipa or len(ipa) > MAX_LEN - 2 or len(word) > MAX_LEN - 2:
            continue
        pairs.append((ipa, word))
    logger.info(f"built {len(pairs):,} (IPA, spelling) pairs after filtering")

    # 95/5 split
    random.shuffle(pairs)
    split = int(len(pairs) * 0.95)
    train_pairs, val_pairs = pairs[:split], pairs[split:]
    logger.info(f"train={len(train_pairs):,}  val={len(val_pairs):,}")

    # ------ Model + tokenizer ------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"loading base model {BASE_MODEL} on {device}")
    tok = AutoTokenizer.from_pretrained(BASE_MODEL)
    model = AutoModelForSeq2SeqLM.from_pretrained(BASE_MODEL).to(device)

    train_ds = P2GDataset(train_pairs, tok, MAX_LEN)
    val_ds = P2GDataset(val_pairs, tok, MAX_LEN)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)

    # ------ Optimizer + scheduler ------
    optim = torch.optim.AdamW(model.parameters(), lr=LR)
    total_steps = len(train_loader) * EPOCHS
    sched = get_linear_schedule_with_warmup(
        optim, num_warmup_steps=int(WARMUP_FRACTION * total_steps),
        num_training_steps=total_steps,
    )

    # ------ Train ------
    log_every = max(50, len(train_loader) // 20)
    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        for step, batch in enumerate(train_loader):
            batch = {k: v.to(device) for k, v in batch.items()}
            out = model(**batch)
            loss = out.loss
            optim.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optim.step()
            sched.step()
            running_loss += loss.item()
            if (step + 1) % log_every == 0:
                avg = running_loss / log_every
                logger.info(
                    f"epoch={epoch}  step={step+1}/{len(train_loader)}  "
                    f"train_loss={avg:.3f}  lr={sched.get_last_lr()[0]:.2e}",
                )
                running_loss = 0.0

        # ------ Validate ------
        model.eval()
        val_loss = 0.0
        n = 0
        with torch.no_grad():
            for batch in val_loader:
                batch = {k: v.to(device) for k, v in batch.items()}
                out = model(**batch)
                val_loss += out.loss.item() * batch["input_ids"].size(0)
                n += batch["input_ids"].size(0)
        logger.info(f"epoch={epoch}  val_loss={val_loss/n:.3f}")

        # Sample qualitative outputs each epoch
        model.eval()
        with torch.no_grad():
            samples = [
                ("ðə", "the"),
                ("dɔɡ", "dog"),
                ("rʌnɪŋ", "running"),
                ("məʃin", "machine"),
                ("kɑnʃəsnəs", "consciousness"),
                # Pseudowords (no expected spelling)
                ("brɪndəlɪŋ", None),
                ("snorvɛnt", None),
                ("flubraʃiz", None),
                ("krimoɪstər", None),
                ("fəʃʌnəbəl", None),
            ]
            for ipa, expected in samples:
                ids = tok(ipa, return_tensors="pt").input_ids.to(device)
                gen = model.generate(ids, max_new_tokens=32, do_sample=False)
                pred = tok.decode(gen[0], skip_special_tokens=True)
                tag = f"(expected={expected!r})" if expected else "(pseudoword)"
                logger.info(f"  {ipa!r:>22} → {pred!r:<22} {tag}")

    # ------ Save ------
    logger.info(f"saving model to {OUTPUT_DIR}")
    model.save_pretrained(OUTPUT_DIR)
    tok.save_pretrained(OUTPUT_DIR)
    logger.info("done.")


if __name__ == "__main__":
    main()

"""Dataset for IPA -> English translation training."""

from __future__ import annotations

import json
from pathlib import Path

from torch import Tensor
from torch.utils.data import Dataset


class IPATranslationDataset(Dataset):
    """Dataset of (IPA, English) pairs formatted for causal LM training.

    Format: ``<ipa> {ipa_text} </ipa> <eng> {english_text} </eng>``

    Labels mask the prompt portion (``<ipa>...</ipa> <eng>``) with -100
    so the model only trains on generating the English output.

    Args:
        ipa_messages: List of IPA strings.
        english_texts: List of corresponding English strings.
        tokenizer: HuggingFace tokenizer instance.
        max_len: Maximum sequence length for tokenization.
    """

    def __init__(
        self,
        ipa_messages: list[str],
        english_texts: list[str],
        tokenizer,
        max_len: int = 256,
    ) -> None:
        self.pairs = list(zip(ipa_messages, english_texts))
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> dict[str, Tensor]:
        ipa, eng = self.pairs[idx]

        prompt = f"<ipa> {ipa} </ipa> <eng>"
        target = f" {eng} </eng>"
        full = prompt + target

        encoding = self.tokenizer(
            full,
            max_length=self.max_len,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )

        input_ids = encoding["input_ids"].squeeze(0)
        attention_mask = encoding["attention_mask"].squeeze(0)

        # Labels: mask prompt portion — only train on English output
        prompt_encoding = self.tokenizer(
            prompt,
            max_length=self.max_len,
            truncation=True,
            return_tensors="pt",
        )
        prompt_len = prompt_encoding["input_ids"].shape[1]

        labels = input_ids.clone()
        labels[:prompt_len] = -100
        labels[attention_mask == 0] = -100

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }

    @classmethod
    def from_jsonl(
        cls,
        path: str | Path,
        tokenizer,
        max_len: int = 256,
    ) -> IPATranslationDataset:
        """Load pairs from a JSONL file.

        Each line must have ``"ipa"`` and ``"english"`` keys.
        """
        ipa_messages = []
        english_texts = []
        with open(path, encoding="utf-8") as f:
            for line in f:
                record = json.loads(line)
                ipa_messages.append(record["ipa"])
                english_texts.append(record["english"])
        return cls(ipa_messages, english_texts, tokenizer, max_len)

"""Dataset + collate for p2g VAE training.

Loads (ipa, spelling) pairs from HDF5, maps chars to ids via CharVocab,
pads to fixed tensors.
"""

from __future__ import annotations

from pathlib import Path

import h5py
import torch
from torch.utils.data import Dataset

from .vocab import PAD_ID, CharVocab


class P2GDataset(Dataset):
    def __init__(
        self,
        h5_path: str | Path,
        ipa_vocab: CharVocab,
        sp_vocab: CharVocab,
        max_ipa_len: int,
        max_spelling_len: int,
    ) -> None:
        with h5py.File(h5_path, "r") as h:
            g = h["pairs"]
            self.ipa = [s.decode() if isinstance(s, bytes) else s for s in g["ipa"][:]]
            self.spelling = [
                s.decode() if isinstance(s, bytes) else s for s in g["spelling"][:]
            ]
            if "loss_weight" in g:
                self.loss_weights = g["loss_weight"][:].tolist()
            else:
                self.loss_weights = [1.0] * len(self.ipa)
        self.ipa_vocab = ipa_vocab
        self.sp_vocab = sp_vocab
        self.max_ipa_len = max_ipa_len
        self.max_spelling_len = max_spelling_len

    def __len__(self) -> int:
        return len(self.ipa)

    def __getitem__(self, idx: int) -> dict:
        ipa_ids = self.ipa_vocab.encode(self.ipa[idx])[: self.max_ipa_len]
        sp_ids = self.sp_vocab.encode(self.spelling[idx])[: self.max_spelling_len]
        return {
            "ipa_ids": ipa_ids,
            "spelling_ids": sp_ids,
            "spelling_len": len(sp_ids),
            "loss_weight": self.loss_weights[idx],
            "ipa_text": self.ipa[idx],
            "spelling_text": self.spelling[idx],
        }


def collate(batch: list[dict], max_ipa_len: int, max_spelling_len: int) -> dict:
    B = len(batch)
    ipa = torch.full((B, max_ipa_len), PAD_ID, dtype=torch.long)
    sp = torch.full((B, max_spelling_len), PAD_ID, dtype=torch.long)
    lens = torch.zeros(B, dtype=torch.long)
    for i, row in enumerate(batch):
        ipa_ids = row["ipa_ids"]
        sp_ids = row["spelling_ids"]
        ipa[i, : len(ipa_ids)] = torch.tensor(ipa_ids, dtype=torch.long)
        sp[i, : len(sp_ids)] = torch.tensor(sp_ids, dtype=torch.long)
        lens[i] = row["spelling_len"]
    weights = torch.tensor([r["loss_weight"] for r in batch], dtype=torch.float32)
    return {
        "ipa_ids": ipa,
        "spelling_ids": sp,
        "spelling_lens": lens,
        "loss_weight": weights,
        "ipa_text": [r["ipa_text"] for r in batch],
        "spelling_text": [r["spelling_text"] for r in batch],
    }


def build_vocabs(train_h5: str | Path) -> tuple[CharVocab, CharVocab]:
    """Construct IPA and spelling char vocabs from the train split."""
    with h5py.File(train_h5, "r") as h:
        g = h["pairs"]
        ipa_texts = [
            s.decode() if isinstance(s, bytes) else s for s in g["ipa"][:]
        ]
        sp_texts = [
            s.decode() if isinstance(s, bytes) else s for s in g["spelling"][:]
        ]
    return CharVocab.build(ipa_texts), CharVocab.build(sp_texts)
